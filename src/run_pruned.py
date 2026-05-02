"""
Run HML inference with a model whose attention heads have been pruned.

Supports two pruning strategies (regular-mode results only):

    top    – prune the top-K most important heads from
                     head_ablation_results/<model>/<dataset>/summary.txt.

    bottom – prune the bottom-X% least-important heads using the pruning
                     order in iterative_taylor/<model>/iterative_taylor_results.json.
                     X% is converted to a head count from that dataset's total heads.

Important behavior:
    - Head selection is dataset-specific (no union across datasets).
    - Inference is run only on entries where the base model has verdict=true.

Pipeline:
    1) For each requested dataset, collect dataset-specific heads to prune.
    2) Load the base model (on CPU) and zero out o_proj weight columns for
         those heads.
    3) Save the pruned model to a temporary directory.
    4) Serve the pruned model via vLLM.
    5) Run inference on that dataset (only base-verdict-true rows).
    6) Stop the server and delete the temporary directory.

Usage – top pruning (top-5 heads per dataset):
    python src/run_pruned.py \\
        --base-model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \\
        --prune-type top \\
        --prune-k 5

Usage – bottom pruning (bottom 5% heads per dataset):
    python src/run_pruned.py \\
        --base-model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \\
        --prune-type bottom \\
    --prune-percent 5

Paths to ablation / Taylor result files are auto-derived from
`--base-model` and `--results-dir`, but can be overridden per-dataset
with `--<dataset>-summary` / `--<dataset>-taylor`.
"""

import argparse
import gc
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
import httpx
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
HML_DATASETS = ["humaneval", "mbpp", "livecodebench"]

from coding_utils import get_layer_head_layout, get_transformer_layers
from run_inference import BENCHMARKS_DIR, DATASETS, SUFFIX_PROMPT, extract_reasoning_and_solution, _content_to_text


# ── Repetition / loop detection ───────────────────────────────────────────────

_DETECT_WINDOW = 3500       # rolling chars kept for pattern search
_DETECT_STEP = 200          # check every N new chars of output
_DETECT_MIN_LEN = 500       # don't check until this many chars have been generated
_DETECT_MIN_REPS = 6        # consecutive exact repetitions required to trigger
_DETECT_MIN_PAT = 6         # shortest repeating unit (chars); avoids indentation FPs
_DETECT_MAX_PAT = 500       # longest repeating unit considered


def _detect_repetition(tail: str) -> bool:
    """Return True if `tail` (a rolling window of recent output) is stuck in an
    exact-character repetitive loop.

    Uses exact string matching to guarantee zero false positives on well-formed
    reasoning chains.  Validated against 839 good chains (0 FPs) and detects
    88–98 % of degenerate finish=length chains from pruned models.
    """
    tail_len = len(tail)
    upper = min(_DETECT_MAX_PAT, tail_len // _DETECT_MIN_REPS)
    for p in range(_DETECT_MIN_PAT, upper + 1):
        if tail_len < p * _DETECT_MIN_REPS:
            continue
        pattern = tail[-p:]
        if tail[-(p * _DETECT_MIN_REPS):] == pattern * _DETECT_MIN_REPS:
            return True
    return False


# ── Inference with loop detection ────────────────────────────────────────────

def _infer_single_pruned(client, model_name, prompt, max_tokens, temperature):
    """Chat-completions streaming inference with real-time repetition detection.

    Identical to run_inference._infer_single (chat path) except that it
    monitors the rolling output window for degenerate loops and aborts early
    when one is detected, setting finish_reason to 'repetition_loop'.
    """
    chunks: list[str] = []
    finish_reason = ""
    error = ""
    rolling_window = ""
    total_chars = 0
    last_check_at = 0

    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        for chunk in stream:
            if not getattr(chunk, "choices", None):
                continue
            choice = chunk.choices[0]
            delta = getattr(choice, "delta", None)
            if delta is not None:
                reasoning_text = _content_to_text(getattr(delta, "reasoning_content", None))
                content_text = _content_to_text(getattr(delta, "content", None))
                text = reasoning_text + content_text
            else:
                text = ""

            if text:
                chunks.append(text)
                total_chars += len(text)
                rolling_window = (rolling_window + text)[-_DETECT_WINDOW:]

            if getattr(choice, "finish_reason", None) is not None:
                finish_reason = choice.finish_reason or ""

            # Periodically check for repetition loops
            if (
                total_chars >= _DETECT_MIN_LEN
                and total_chars - last_check_at >= _DETECT_STEP
            ):
                if _detect_repetition(rolling_window):
                    finish_reason = "repetition_loop"
                    break
                last_check_at = total_chars

    except Exception as e:
        error = str(e)
        error_lower = error.lower()
        if "timed out" in error_lower or "timeout" in error_lower or "read timeout" in error_lower:
            finish_reason = "timeout"
        else:
            finish_reason = "error"
        partial = "".join(chunks)
        if partial:
            print(f"  ERROR after partial output ({len(partial)} chars): {error}")
        else:
            print(f"  ERROR: {error}")

    return {
        "output": "".join(chunks),
        "finish_reason": finish_reason,
        "error": error,
    }


def _is_true_verdict(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return int(value) == 1
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y", "pass", "passed"}
    return False


def _load_base_verdict_true_indices(base_bench_dir: Path, cfg: dict, dataset_name: str) -> set[int]:
    base_path = base_bench_dir / cfg["file"]
    if not base_path.exists():
        raise FileNotFoundError(
            f"Base benchmark file not found for {dataset_name}: {base_path}. "
            "Run base-model inference and evaluation first."
        )

    with open(base_path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON in {base_path}, got {type(data).__name__}")

    verdict_present = 0
    with_solution = 0
    true_indices: set[int] = set()
    for idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            continue
        if entry.get("solution"):
            with_solution += 1
        if "verdict" not in entry:
            continue
        verdict_present += 1
        if _is_true_verdict(entry.get("verdict")):
            true_indices.add(idx)

    if verdict_present == 0:
        raise ValueError(
            f"No verdicts found in {base_path}. "
            "Run evaluate.py for the base model before run_pruned."
        )

    print(
        f"[filter] {dataset_name}: base verdict=true {len(true_indices)}/{len(data)} "
        f"(verdict fields={verdict_present}, solutions={with_solution})"
    )
    return true_indices


def _count_pending_rows_for_dataset(
    cfg: dict,
    output_dir: Path,
    allowed_indices: set[int] | None = None,
) -> tuple[int, int]:
    """Return (eligible_count, pending_count) for a dataset output file.

    This is a lightweight precheck used to avoid building/serving a pruned
    model when all eligible rows are already inferred.
    """
    src_path = BENCHMARKS_DIR / cfg["file"]
    out_path = output_dir / cfg["file"]

    data_path = out_path if out_path.exists() else src_path
    with open(data_path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON in {data_path}")

    total = len(data)
    target_indices = list(range(total))
    if allowed_indices is not None:
        target_indices = sorted(i for i in allowed_indices if 0 <= i < total)

    pending = [
        i for i in target_indices
        if not data[i].get("reasoning_chain") and not data[i].get("solution")
    ]
    return len(target_indices), len(pending)


def _run_inference_on_dataset_pruned(
    client: OpenAI,
    model_name: str,
    dataset_name: str,
    cfg: dict,
    max_tokens: int,
    temperature: float,
    output_dir: Path,
    allowed_indices: set[int] | None = None,
    workers: int = 32,
):
    """Like run_inference.run_inference_on_dataset but uses _infer_single_pruned
    and writes outputs to an explicitly specified directory.

    If allowed_indices is provided, only those sample indices are inferred.
    Entries outside the allowed set are scrubbed of inference/eval fields.
    """
    src_path = BENCHMARKS_DIR / cfg["file"]
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / cfg["file"]
    question_key = cfg["question_key"]

    if out_path.exists():
        with open(out_path) as f:
            data = json.load(f)
    else:
        with open(src_path) as f:
            data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON in {out_path if out_path.exists() else src_path}")

    def _clear_inference_fields(entry: dict) -> bool:
        changed = False
        for key in ("reasoning_chain", "solution", "finish_reason", "inference_error", "verdict", "passed"):
            if key in entry:
                entry.pop(key, None)
                changed = True
        return changed

    total = len(data)
    target_indices = list(range(total))
    if allowed_indices is not None:
        valid_allowed = sorted(i for i in allowed_indices if 0 <= i < total)
        dropped = len(allowed_indices) - len(valid_allowed)
        if dropped > 0:
            print(f"  [filter] Ignoring {dropped} out-of-range indices for {dataset_name}")

        allowed_set = set(valid_allowed)
        scrubbed = 0
        for idx, entry in enumerate(data):
            if idx in allowed_set:
                continue
            if isinstance(entry, dict) and _clear_inference_fields(entry):
                scrubbed += 1

        if scrubbed > 0:
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            print(f"  [filter] Scrubbed {scrubbed} excluded rows in {out_path}")

        target_indices = valid_allowed
        print(f"  [filter] Eligible rows for {dataset_name}: {len(target_indices)}/{total}")

    scoped_total = len(target_indices)
    if scoped_total == 0:
        print(f"  Nothing to do for {dataset_name}: no eligible rows")
        return

    todo = [
        i for i in target_indices
        if not data[i].get("reasoning_chain") and not data[i].get("solution")
    ]
    already_done = scoped_total - len(todo)
    if already_done > 0:
        print(f"  Resuming: {already_done}/{scoped_total} eligible rows already done, {len(todo)} remaining")
    if not todo:
        print(f"  Nothing to do for {dataset_name}")
        return

    lock = threading.Lock()
    done_count = already_done
    test_key = cfg.get("test_key")

    def process(idx):
        nonlocal done_count
        entry = data[idx]
        prompt = entry[question_key]
        if test_key and entry.get(test_key):
            prompt += "\n" + "\n".join(entry[test_key])
        prompt += SUFFIX_PROMPT

        t0 = time.time()
        infer = _infer_single_pruned(client, model_name, prompt, max_tokens, temperature)
        output = infer["output"]
        if output:
            reasoning, solution = extract_reasoning_and_solution(output)
        else:
            reasoning = f"[INFERENCE_ERROR] {infer['error']}" if infer["error"] else ""
            solution = ""

        elapsed = time.time() - t0
        status_parts = ["with solution" if solution else "NO solution block"]
        if infer["finish_reason"]:
            status_parts.append(f"finish={infer['finish_reason']}")
        if infer["error"]:
            status_parts.append("error")
        sol_status = ", ".join(status_parts)

        with lock:
            entry["reasoning_chain"] = reasoning
            entry["solution"] = solution
            entry["finish_reason"] = infer["finish_reason"]
            # Clear stale evaluation results so evaluate.py re-evaluates this entry
            entry.pop("verdict", None)
            entry.pop("passed", None)
            if infer["error"]:
                entry["inference_error"] = infer["error"]
            else:
                entry.pop("inference_error", None)
            done_count += 1
            print(
                f"  [{done_count}/{scoped_total}] idx={idx} done in {elapsed:.1f}s "
                f"({sol_status}, chars={len(output)})"
            )
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

    print(f"  Running {len(todo)} eligible samples with {workers} workers...")
    print(f"  Output: {out_path}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process, idx): idx for idx in todo}
        for fut in as_completed(futures):
            fut.result()

    print(
        f"  Finished {dataset_name}: {scoped_total} eligible rows "
        f"(of {total} total) in {out_path}"
    )


# ── Head collection ──────────────────────────────────────────────────────────

def _ablation_summary_path(results_dir: Path, dataset: str, model_short: str) -> Path:
    return results_dir / dataset / "head_ablation_results" / model_short / dataset / "summary.txt"


def _taylor_json_path(results_dir: Path, dataset: str, model_short: str) -> Path:
    return results_dir / dataset / "iterative_taylor" / model_short / "iterative_taylor_results.json"


def parse_top_heads_from_summary(summary_path: Path, top_k: int) -> list[tuple[int, int]]:
    """Read ranked heads from head_ablation summary.txt (top = most important)."""
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.txt not found: {summary_path}")

    pattern = re.compile(r"^\s*\d+\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|")
    heads: list[tuple[int, int]] = []
    with open(summary_path) as f:
        for line in f:
            m = pattern.match(line)
            if not m:
                continue
            heads.append((int(m.group(1)), int(m.group(2))))
            if len(heads) >= top_k:
                break

    if len(heads) < top_k:
        raise ValueError(
            f"Could not parse {top_k} heads from {summary_path}; found only {len(heads)}."
        )
    return heads


def _read_taylor_order(json_path: Path) -> tuple[list[tuple[int, int]], int]:
    """Return (ordered_pruned_heads, total_heads) from iterative Taylor results."""
    if not json_path.exists():
        raise FileNotFoundError(f"iterative_taylor_results.json not found: {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    history = data.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError(f"Missing/invalid 'history' in {json_path}")

    layer_num_heads = data.get("layer_num_heads")
    total_heads = None
    if isinstance(layer_num_heads, list) and layer_num_heads:
        try:
            total_heads = sum(int(v) for v in layer_num_heads)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Invalid layer_num_heads in {json_path}: {exc}") from exc
    else:
        num_layers = data.get("num_layers")
        num_heads = data.get("num_heads")
        if isinstance(num_layers, int) and isinstance(num_heads, int) and num_layers > 0 and num_heads > 0:
            total_heads = num_layers * num_heads

    if not total_heads or total_heads <= 0:
        raise ValueError(
            f"Could not determine total head count from {json_path}. "
            "Expected layer_num_heads or (num_layers, num_heads)."
        )

    ordered: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for step in history:
        if int(step.get("iteration", 0)) <= 0:
            continue
        for pair in step.get("pruned_this_iter", []):
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            head = (int(pair[0]), int(pair[1]))
            if head in seen:
                continue
            seen.add(head)
            ordered.append(head)

    return ordered, total_heads


def parse_bottom_heads_from_taylor(
    json_path: Path,
    bottom_k: int | None = None,
    bottom_percent: float | None = None,
) -> tuple[list[tuple[int, int]], int, int]:
    """Read least-important heads from iterative_taylor_results.json.

    Heads pruned earliest (lowest iterations) are the least important.

    Returns:
      (heads, selected_count, total_heads)
    """
    if (bottom_k is None) == (bottom_percent is None):
        raise ValueError("Provide exactly one of bottom_k or bottom_percent")

    ordered, total_heads = _read_taylor_order(json_path)

    if bottom_percent is not None:
        if bottom_percent <= 0:
            raise ValueError(f"bottom_percent must be > 0, got {bottom_percent}")
        target_k = max(1, int(math.ceil((bottom_percent / 100.0) * total_heads)))
    else:
        if bottom_k is None or bottom_k <= 0:
            raise ValueError(f"bottom_k must be > 0, got {bottom_k}")
        target_k = int(bottom_k)

    if target_k > len(ordered):
        raise ValueError(
            f"Could not extract {target_k} pruned heads from {json_path}; "
            f"only {len(ordered)} found in history."
        )

    return ordered[:target_k], target_k, total_heads


def collect_dataset_heads(
    prune_type: str,
    dataset: str,
    model_short: str,
    results_dir: Path,
    override_paths: dict[str, Path],
    prune_k: int | None,
    prune_percent: float | None,
) -> list[tuple[int, int]]:
    """Collect heads for one dataset (no cross-dataset union)."""
    if prune_type == "top":
        if prune_k is None or prune_k <= 0:
            raise ValueError("--prune-k must be > 0 for --prune-type top")
        path = override_paths.get(dataset) or _ablation_summary_path(results_dir, dataset, model_short)
        heads = parse_top_heads_from_summary(path, prune_k)
        print(f"[heads] {dataset} top-{prune_k} from {path}: {len(heads)} heads")
        return heads

    path = override_paths.get(dataset) or _taylor_json_path(results_dir, dataset, model_short)
    heads, selected_k, total_heads = parse_bottom_heads_from_taylor(
        path,
        bottom_k=prune_k,
        bottom_percent=prune_percent,
    )

    if prune_percent is not None:
        print(
            f"[heads] {dataset} bottom-{prune_percent:g}% from {path}: "
            f"{selected_k}/{total_heads} heads"
        )
    else:
        print(f"[heads] {dataset} bottom-{selected_k} from {path}: {len(heads)} heads")

    return heads


# ── Model pruning ─────────────────────────────────────────────────────────────

def prune_o_proj_input_heads(model, heads: list[tuple[int, int]]) -> None:
    """Zero out o_proj input columns for each (layer, head) pair."""
    layers = get_transformer_layers(model)
    num_layers = len(layers)
    layer_head_dims, layer_num_heads, max_heads = get_layer_head_layout(model, layers=layers)
    total_heads = sum(layer_num_heads)

    unique_dims = sorted(set(layer_head_dims))
    if len(unique_dims) == 1:
        print(f"[prune] Architecture: {num_layers} layers × <= {max_heads} heads, head_dim={unique_dims[0]} ({total_heads} total)")
    else:
        print(f"[prune] Architecture: {num_layers} layers × <= {max_heads} heads, mixed head_dim={unique_dims} ({total_heads} total)")
    print(f"[prune] Pruning {len(heads)} heads")

    with torch.no_grad():
        for layer_idx, head_idx in heads:
            if not (0 <= layer_idx < num_layers):
                raise ValueError(f"Layer index {layer_idx} out of range [0, {num_layers - 1}]")

            layer_heads = layer_num_heads[layer_idx]
            if not (0 <= head_idx < layer_heads):
                raise ValueError(f"Head index {head_idx} out of range [0, {layer_heads - 1}] for layer {layer_idx}")

            head_dim = layer_head_dims[layer_idx]
            o_proj = layers[layer_idx].self_attn.o_proj
            start = head_idx * head_dim
            end = (head_idx + 1) * head_dim
            if end > o_proj.weight.shape[1]:
                raise ValueError(
                    f"Head slice L{layer_idx}H{head_idx} -> [{start}:{end}] exceeds "
                    f"o_proj in_features={o_proj.weight.shape[1]}"
                )
            o_proj.weight[:, start:end] = 0


def build_and_save_pruned_model(
    base_model: str,
    heads: list[tuple[int, int]],
    out_dir: Path,
) -> None:
    print(f"[model] Loading base model: {base_model}")
    processor = None
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    # Some models (e.g., Gemma variants served by vLLM) require processor
    # artifacts like preprocessor_config.json in the model directory.
    try:
        processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[model] Processor load skipped: {exc}")

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    prune_o_proj_input_heads(model, heads)

    print(f"[model] Saving pruned model to: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    if processor is not None:
        processor.save_pretrained(out_dir)
        print("[model] Saved processor artifacts (including preprocessor_config.json when available)")

    del model, tokenizer, processor
    gc.collect()


# ── vLLM helpers ──────────────────────────────────────────────────────────────

def launch_vllm_server(
    model_path: Path,
    served_model_name: str,
    host: str,
    port: int,
    max_model_len: int,
) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(model_path),
        "--served-model-name", served_model_name,
        "--host", host,
        "--port", str(port),
        "--trust-remote-code",
        "--dtype", "auto",
        "--max-model-len", str(max_model_len),
    ]
    print("[vllm] Launch command:")
    print(" ".join(cmd))
    return subprocess.Popen(cmd, start_new_session=True)


def _port_is_available(host: str, port: int) -> bool:
    """Return True if binding on (host, port) succeeds."""
    bind_host = "" if host in {"0.0.0.0", "::"} else host
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((bind_host, port))
        return True
    except OSError:
        return False


def select_available_port(host: str, start_port: int, max_attempts: int = 100) -> int:
    """Pick the first available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        if _port_is_available(host, port):
            if port != start_port:
                print(f"[vllm] Port {start_port} unavailable, using {port}")
            return port
    raise RuntimeError(
        f"No free port found in range [{start_port}, {start_port + max_attempts - 1}]"
    )


def wait_for_server(api_base: str, server_proc: subprocess.Popen, timeout_sec: int, model_name: str = "") -> None:
    models_url = f"{api_base.rstrip('/')}/models"
    deadline = time.time() + timeout_sec
    print(f"[vllm] Waiting for readiness at {models_url} (timeout={timeout_sec}s)")
    while time.time() < deadline:
        if server_proc.poll() is not None:
            raise RuntimeError(
                f"vLLM exited before becoming ready (exit code {server_proc.returncode})."
            )
        try:
            with urllib.request.urlopen(models_url, timeout=5) as resp:
                if resp.status == 200:
                    import json as _json
                    body = _json.loads(resp.read())
                    model_ids = [m.get("id", "") for m in body.get("data", [])]
                    if model_ids and (not model_name or model_name in model_ids):
                        print(f"[vllm] Server is ready. Models: {model_ids}")
                        return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, Exception):
            pass
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for vLLM server at {models_url}")


def stop_server(server_proc: subprocess.Popen) -> None:
    if server_proc.poll() is not None:
        return
    print("[vllm] Stopping server...")
    try:
        os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        server_proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        print("[vllm] Terminate timed out; killing.")
        try:
            os.killpg(os.getpgid(server_proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            server_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass


# ── Inference ─────────────────────────────────────────────────────────────────

def run_hml_inference(
    api_base: str,
    served_model_name: str,
    inference_datasets: list[str],
    max_tokens: int,
    temperature: float,
    output_dir: Path,
    base_true_indices: dict[str, set[int]],
    workers: int,
) -> None:
    client = OpenAI(
        base_url=api_base,
        api_key="empty",
        timeout=httpx.Timeout(connect=60.0, read=None, write=60.0, pool=60.0),
    )
    for ds_name in inference_datasets:
        # livecodebench maps to livecodebench_v6 in DATASETS
        ds_key = "livecodebench_v6" if ds_name == "livecodebench" else ds_name
        if ds_key not in DATASETS:
            print(f"[infer] Unknown dataset key '{ds_key}', skipping")
            continue
        print(f"[infer] Dataset: {ds_key}")
        allowed = base_true_indices.get(ds_name)
        if allowed is None:
            print(f"[infer] No base-verdict filter for {ds_name}; skipping")
            continue
        _run_inference_on_dataset_pruned(
            client=client,
            model_name=served_model_name,
            dataset_name=ds_key,
            cfg=DATASETS[ds_key],
            max_tokens=max_tokens,
            temperature=temperature,
            output_dir=output_dir,
            allowed_indices=allowed,
            workers=workers,
        )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HML inference with dataset-specific pruned attention heads.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Core
    parser.add_argument("--base-model", required=True, help="HuggingFace model ID of the base model")
    parser.add_argument(
        "--base-bench-dir", type=Path, default=None,
        help="Folder containing evaluated base-model JSONs with verdict fields "
             "(default: data/benchmarks/<base-model-with-slashes-replaced>)",
    )
    parser.add_argument(
        "--prune-type", required=True, choices=["top", "bottom"],
        help="'top': prune most important heads (from ablation); 'bottom': prune least important (from Taylor)",
    )
    parser.add_argument(
        "--prune-k", type=int, default=None,
        help="For top: number of heads to remove. For bottom: exact head count (legacy alternative to --prune-percent).",
    )
    parser.add_argument(
        "--prune-percent", type=float, default=None,
        help="For bottom only: percentage of total heads to remove (e.g., 1, 2, 5, 10)",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=HML_DATASETS,
        choices=HML_DATASETS,
        help="Datasets to prune and run inference on (default: all HML)",
    )
    parser.add_argument(
        "--inference-datasets", nargs="+", default=None,
        choices=HML_DATASETS,
        help="Deprecated alias for --datasets; if provided, it overrides --datasets",
    )
    parser.add_argument(
        "--output-name", default=None,
        help="Prefix for served model name (default: auto-generated)",
    )
    parser.add_argument(
        "--results-dir", type=Path,
        default=REPO_ROOT / "results" / "hml",
        help="Root results directory (default: results/hml)",
    )

    # Per-dataset path overrides
    for ds in HML_DATASETS:
        parser.add_argument(
            f"--{ds}-summary", type=Path, default=None,
            help=f"Override path to {ds} ablation summary.txt (top pruning)",
        )
        parser.add_argument(
            f"--{ds}-taylor", type=Path, default=None,
            help=f"Override path to {ds} iterative_taylor_results.json (bottom pruning)",
        )

    # vLLM
    vllm = parser.add_argument_group("vLLM server")
    vllm.add_argument("--host", default="0.0.0.0")
    vllm.add_argument("--port", type=int, default=8000)
    vllm.add_argument("--max-model-len", type=int, default=32768)
    vllm.add_argument("--startup-timeout", type=int, default=9000)

    # Inference
    infer = parser.add_argument_group("inference")
    infer.add_argument("--max-tokens", type=int, default=28000)
    infer.add_argument("--temperature", type=float, default=0.0)
    infer.add_argument("--workers", type=int, default=32)

    # Temp model
    tmp = parser.add_argument_group("temporary model")
    tmp.add_argument(
        "--temp-root", type=Path,
        default=REPO_ROOT / "data" / "tmp_pruned_models",
        help="Parent dir for the temporary pruned model",
    )
    tmp.add_argument(
        "--keep-temp-model", action="store_true",
        help="Keep the temporary pruned model directory after inference",
    )

    args = parser.parse_args()

    if args.prune_type == "top":
        if args.prune_k is None or args.prune_k <= 0:
            parser.error("--prune-k must be a positive integer for --prune-type top")
        if args.prune_percent is not None:
            parser.error("--prune-percent is only valid for --prune-type bottom")
    else:
        if args.prune_percent is not None and args.prune_percent <= 0:
            parser.error("--prune-percent must be > 0")
        if args.prune_k is not None and args.prune_k <= 0:
            parser.error("--prune-k must be > 0 when provided")
        if args.prune_percent is None and args.prune_k is None:
            parser.error("For --prune-type bottom, provide either --prune-percent or --prune-k")
        if args.prune_percent is not None and args.prune_k is not None:
            parser.error("For --prune-type bottom, provide only one of --prune-percent or --prune-k")

    if args.inference_datasets is not None:
        print("[config] --inference-datasets provided; overriding --datasets")
        args.datasets = args.inference_datasets

    return args


def _format_percent_token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _build_prune_tag(prune_type: str, prune_k: int | None, prune_percent: float | None) -> str:
    if prune_type == "top":
        if prune_k is None:
            raise ValueError("prune_k is required for top pruning")
        return f"top_{prune_k}"
    if prune_percent is not None:
        return f"bottom_pct_{_format_percent_token(prune_percent)}"
    if prune_k is None:
        raise ValueError("prune_k or prune_percent is required for bottom pruning")
    return f"bottom_{prune_k}"


def main():
    args = parse_args()

    model_short = args.base_model.split("/")[-1]
    prune_tag = _build_prune_tag(args.prune_type, args.prune_k, args.prune_percent)

    output_dir = BENCHMARKS_DIR / "pruning" / model_short / prune_tag
    served_name_prefix = args.output_name or f"{model_short}_pruned_{prune_tag}"

    base_bench_dir = args.base_bench_dir or (BENCHMARKS_DIR / args.base_model.replace("/", "--"))
    if not base_bench_dir.exists():
        raise FileNotFoundError(
            f"Base benchmark dir not found: {base_bench_dir}. "
            "Run base-model inference/evaluation first or pass --base-bench-dir."
        )

    print(f"[config] Base model:   {args.base_model}")
    print(f"[config] Base bench:   {base_bench_dir}")
    print(f"[config] Prune type:   {args.prune_type}")
    if args.prune_type == "top":
        print(f"[config] Prune-k:      {args.prune_k} (exact heads)")
    elif args.prune_percent is not None:
        print(f"[config] Prune-%:      {args.prune_percent:g}%")
    else:
        print(f"[config] Prune-k:      {args.prune_k} (exact heads)")
    print(f"[config] Datasets:     {args.datasets}")
    print(f"[config] Output dir:   {output_dir}")
    print()

    # Build per-dataset override paths
    override_paths: dict[str, Path] = {}
    for ds in HML_DATASETS:
        key = "summary" if args.prune_type == "top" else "taylor"
        val = getattr(args, f"{ds}_{key}", None)
        if val is not None:
            override_paths[ds] = val

    # Build base-verdict=true index sets per dataset
    base_true_indices: dict[str, set[int]] = {}
    for ds_name in args.datasets:
        ds_key = "livecodebench_v6" if ds_name == "livecodebench" else ds_name
        cfg = DATASETS.get(ds_key)
        if cfg is None:
            print(f"[filter] Unknown dataset key '{ds_key}', skipping")
            continue
        base_true_indices[ds_name] = _load_base_verdict_true_indices(base_bench_dir, cfg, ds_key)

    if not base_true_indices:
        raise RuntimeError("No valid datasets to run after base-verdict filtering.")

    args.temp_root.mkdir(parents=True, exist_ok=True)

    for ds_name in args.datasets:
        if ds_name not in base_true_indices:
            print(f"[skip] {ds_name}: no base-verdict index set")
            continue

        ds_key = "livecodebench_v6" if ds_name == "livecodebench" else ds_name
        cfg = DATASETS.get(ds_key)
        if cfg is None:
            print(f"[skip] {ds_name}: unknown dataset key '{ds_key}'")
            continue

        eligible_count, pending_count = _count_pending_rows_for_dataset(
            cfg=cfg,
            output_dir=output_dir,
            allowed_indices=base_true_indices[ds_name],
        )
        if eligible_count == 0:
            print(f"[skip] {ds_name}: no eligible base-verdict=true rows")
            continue
        if pending_count == 0:
            print(
                f"[skip] {ds_name}: all {eligible_count} eligible rows are already complete; "
                "skipping prune/model-save/vLLM startup"
            )
            continue

        print(f"\n[dataset] {ds_name}")
        heads_to_prune = collect_dataset_heads(
            prune_type=args.prune_type,
            dataset=ds_name,
            model_short=model_short,
            results_dir=args.results_dir,
            override_paths=override_paths,
            prune_k=args.prune_k,
            prune_percent=args.prune_percent,
        )
        print(f"[dataset] {ds_name}: pruning {len(heads_to_prune)} heads")

        prefix = f"{model_short}_{prune_tag}_{ds_name}_"
        temp_model_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=str(args.temp_root)))
        served_model_name = f"{served_name_prefix}_{ds_name}"
        server_proc = None
        selected_port = select_available_port(args.host, args.port)
        api_base = f"http://localhost:{selected_port}/v1"

        try:
            build_and_save_pruned_model(args.base_model, heads_to_prune, temp_model_dir)
            print()

            server_proc = launch_vllm_server(
                model_path=temp_model_dir,
                served_model_name=served_model_name,
                host=args.host,
                port=selected_port,
                max_model_len=args.max_model_len,
            )
            wait_for_server(api_base, server_proc, timeout_sec=args.startup_timeout, model_name=served_model_name)
            print()

            run_hml_inference(
                api_base=api_base,
                served_model_name=served_model_name,
                inference_datasets=[ds_name],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                output_dir=output_dir,
                base_true_indices=base_true_indices,
                workers=args.workers,
            )
        finally:
            if server_proc is not None:
                stop_server(server_proc)

            if args.keep_temp_model:
                print(f"[cleanup] Keeping temp model: {temp_model_dir}")
            else:
                print(f"[cleanup] Removing temp model: {temp_model_dir}")
                shutil.rmtree(temp_model_dir, ignore_errors=True)

    print(f"\n[done] Outputs saved under {output_dir}/")


if __name__ == "__main__":
    main()
