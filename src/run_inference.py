"""
Run inference on HumanEval, MBPP, and LiveCodeBench v6 using a vLLM-served model.

Usage (connect to existing server):
    python src/run_inference.py \
        --model-name "Qwen/Qwen3-8B" \
        --api-base "http://localhost:8000/v1" \
        --max-tokens 4096

Usage (auto-start vLLM server):
    python src/run_inference.py \
        --model-name "Qwen/Qwen3-8B" \
        --serve \
        --port 8000 \
        --max-model-len 32768 \
        --max-tokens 4096

The script appends two new fields to each entry in the dataset JSON files:
  - "reasoning_chain": the text before the ```python block
  - "solution": the ```python ... ``` block (including tags), or "" if absent

It also appends metadata fields:
    - "finish_reason": model finish reason (e.g., "stop", "length")
    - "inference_error": present only when request/streaming fails
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from openai import OpenAI

BENCHMARKS_DIR = Path(__file__).resolve().parents[1] / "data" / "benchmarks"

DATASETS = {
    "humaneval": {
        "file": "humaneval.json",
        "question_key": "prompt",
    },
    "mbpp": {
        "file": "mbpp.json",
        "question_key": "text",
        "test_key": "test_list",
    },
    "livecodebench_v6": {
        "file": "livecodebench_v6.json",
        "question_key": "question_content",
    },
}

SUFFIX_PROMPT = (
    "\n\nSolve the question thinking step by step and give the final python "
    "code within ```python ....``` blocks."
)


def extract_reasoning_and_solution(output: str):
    """Split model output into reasoning chain and solution block."""
    # Find the last ```python ... ``` block
    pattern = r"(```python.*?```)"
    matches = list(re.finditer(pattern, output, re.DOTALL))
    if matches:
        last_match = matches[-1]
        reasoning = output[: last_match.start()].strip()
        solution = last_match.group(1)
    else:
        reasoning = output.strip()
        solution = ""
    return reasoning, solution


from coding_utils import bpe_decode as _bpe_decode


def _content_to_text(content) -> str:
    """Normalize OpenAI content payloads (string/list) into a plain string,
    decoding GPT-2/Llama byte-level BPE proxy characters if present."""
    if content is None:
        return ""
    if isinstance(content, str):
        return _bpe_decode(content)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return _bpe_decode("".join(parts))
    return _bpe_decode(str(content))


def _infer_single(client, model_name, prompt, max_tokens, temperature, use_completions=False):
    """Run a single inference call and return output + finish metadata.

    Returns:
        dict with keys: output, finish_reason, error
    """
    chunks = []
    finish_reason = ""
    error = ""

    try:
        if use_completions:
            stream = client.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )

            for chunk in stream:
                if not getattr(chunk, "choices", None):
                    continue
                choice = chunk.choices[0]
                text = getattr(choice, "text", None) or ""
                if text:
                    chunks.append(text)
                if getattr(choice, "finish_reason", None) is not None:
                    finish_reason = choice.finish_reason or ""
        else:
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
                    # Preserve both reasoning and normal content when present.
                    reasoning_text = _content_to_text(getattr(delta, "reasoning_content", None))
                    content_text = _content_to_text(getattr(delta, "content", None))
                    text = reasoning_text + content_text
                else:
                    text = ""
                if text:
                    chunks.append(text)
                if getattr(choice, "finish_reason", None) is not None:
                    finish_reason = choice.finish_reason or ""
    except Exception as e:
        error = str(e)
        # Override finish_reason — whatever vLLM sent last is unreliable when
        # the stream was cut short by an exception (e.g. ReadTimeout can arrive
        # after a "length" chunk, making it look like a normal stop).
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


def _model_dir(model_name: str) -> Path:
    """Return output directory for a model: data/benchmarks/<model-name>/."""
    safe_name = model_name.replace("/", "--")
    out = BENCHMARKS_DIR / safe_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_inference_on_dataset(
    client: OpenAI,
    model_name: str,
    dataset_name: str,
    cfg: dict,
    max_tokens: int,
    temperature: float,
    workers: int = 8,
    use_completions: bool = False,
):
    src_path = BENCHMARKS_DIR / cfg["file"]
    out_path = _model_dir(model_name) / cfg["file"]
    question_key = cfg["question_key"]

    # Load existing output (for resume) or copy from source
    if out_path.exists():
        with open(out_path) as f:
            data = json.load(f)
    else:
        with open(src_path) as f:
            data = json.load(f)

    total = len(data)
    todo = [i for i, entry in enumerate(data)
            if not entry.get("solution")]
    already_done = total - len(todo)
    if already_done > 0:
        print(f"  Resuming: {already_done}/{total} already done, {len(todo)} remaining")

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
        infer = _infer_single(client, model_name, prompt, max_tokens, temperature, use_completions)
        output = infer["output"]
        if output:
            reasoning, solution = extract_reasoning_and_solution(output)
        else:
            # Persist failure details so this sample is not silently lost.
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
                f"  [{done_count}/{total}] idx={idx} done in {elapsed:.1f}s "
                f"({sol_status}, chars={len(output)})"
            )
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

    print(f"  Running {len(todo)} samples with {workers} workers...")
    print(f"  Output: {out_path}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process, idx): idx for idx in todo}
        for fut in as_completed(futures):
            fut.result()

    print(f"  Finished {dataset_name}: {total} samples saved to {out_path}")


# ---------- vLLM server management ----------

def launch_vllm_server(
    model_name: str,
    host: str,
    port: int,
    max_model_len: int,
    extra_vllm_args: list | None = None,
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model", model_name,
        "--host", host,
        "--port", str(port),
        "--trust-remote-code",
        "--dtype", "auto",
        "--max-model-len", str(max_model_len),
    ]
    if extra_vllm_args:
        cmd.extend(extra_vllm_args)
    print("[vllm] Launch command:")
    print(" ".join(cmd))
    return subprocess.Popen(cmd, start_new_session=True)


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
        print("[vllm] Terminate timed out; killing server.")
        try:
            os.killpg(os.getpgid(server_proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            server_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass


def main():
    parser = argparse.ArgumentParser(description="Run inference on coding benchmarks")
    parser.add_argument("--model-name", required=True, help="Model name as served by vLLM")
    parser.add_argument("--api-base", default=None, help="vLLM API base URL (default: http://localhost:<port>/v1)")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=32, help="Parallel inference workers")
    parser.add_argument("--use-completions", action="store_true", help="Use completions API instead of chat")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()),
        default=list(DATASETS.keys()),
        help="Which datasets to run (default: all)",
    )

    # vLLM server management
    server_group = parser.add_argument_group("vLLM server (used when --serve is set)")
    server_group.add_argument(
        "--serve",
        action="store_true",
        help="Auto-start a vLLM server before inference and stop it afterward",
    )
    server_group.add_argument("--host", default="0.0.0.0", help="vLLM server host")
    server_group.add_argument("--port", type=int, default=8000, help="vLLM server port")
    server_group.add_argument("--max-model-len", type=int, default=32768, help="vLLM max model length")
    server_group.add_argument("--startup-timeout", type=int, default=9000, help="Seconds to wait for server readiness")

    args = parser.parse_args()

    api_base = args.api_base or f"http://localhost:{args.port}/v1"

    print(f"Model: {args.model_name}")
    print(f"API:   {api_base}")
    print(f"Max tokens: {args.max_tokens}, Temperature: {args.temperature}, Workers: {args.workers}")
    print()

    server_proc = None
    try:
        if args.serve:
            server_proc = launch_vllm_server(
                model_name=args.model_name,
                host=args.host,
                port=args.port,
                max_model_len=args.max_model_len,
            )
            wait_for_server(api_base, server_proc, timeout_sec=args.startup_timeout, model_name=args.model_name)
            print()

        client = OpenAI(
            base_url=api_base,
            api_key="empty",
            timeout=httpx.Timeout(connect=60.0, read=None, write=60.0, pool=60.0),
        )

        for name in args.datasets:
            cfg = DATASETS[name]
            print(f"=== {name} ===")
            run_inference_on_dataset(
                client, args.model_name, name, cfg,
                args.max_tokens, args.temperature,
                args.workers, args.use_completions,
            )
            print()

        print("All done.")

    finally:
        if server_proc is not None:
            stop_server(server_proc)


if __name__ == "__main__":
    main()
