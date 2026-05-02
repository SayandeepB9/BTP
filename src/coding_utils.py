"""
Shared utilities for coding dataset analysis scripts.

Supports both standard answer-token analysis and chain+code analysis.
"""

import gc
import os

# ── GPT-2/Llama byte-level BPE decode ────────────────────────────────────────
# vLLM may stream raw BPE token strings (e.g. Ġ for space, Ċ for newline)
# that get stored in JSON files. Decoding these before tokenizer.encode()
# ensures correct token sequences in entropy / ablation analyses.
def _build_bpe_remap() -> dict:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): bytes([b]) for b, c in zip(bs, cs) if c != b}

_BPE_REMAP: dict = _build_bpe_remap()


def bpe_decode(text: str) -> str:
    """Convert GPT-2/Llama BPE byte-proxy chars back to proper UTF-8 text.
    No-op for text that contains no proxy characters."""
    if not isinstance(text, str) or not any(ch in _BPE_REMAP for ch in text):
        return text
    buf = bytearray()
    for ch in text:
        if ch in _BPE_REMAP:
            buf.extend(_BPE_REMAP[ch])
        else:
            buf.extend(ch.encode("utf-8"))
    return buf.decode("utf-8", errors="replace")


import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ===================== Constants =====================

ALL_MODELS = [
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.1-8B",
    "google/gemma-4-E4B-it",
    "google/gemma-4-26B-A4B-it",
]
ALL_DATASETS = ["livecodebench", "mbpp", "humaneval", "codeforces"]
HML_DATASETS = ["humaneval", "mbpp", "livecodebench"]

# HML dataset file/field mapping
HML_DATASET_CONFIG = {
    "humaneval": {"file": "humaneval.json", "question_key": "prompt"},
    "mbpp": {"file": "mbpp.json", "question_key": "text", "test_key": "test_list"},
    "livecodebench": {"file": "livecodebench_v6.json", "question_key": "question_content"},
}

SUFFIX_PROMPT = (
    "\n\nSolve the question thinking step by step and give the final python "
    "code within ```python ....``` blocks."
)

# Codeforces rating range labels (used as directory names and CLI choices)
RATING_RANGES = ["<=1000", "1001-1500", "1501-2000", "2001-2500", "2501-3000", ">3000"]

# Token limits for chain and code
CHAIN_TOKEN_LIMIT = 1024
CODE_TOKEN_LIMIT = 1024


def _get_text_config(model):
    """Return nested text config when present (Gemma4), else top-level config."""
    cfg = model.config
    return getattr(cfg, "text_config", None) or cfg


def get_num_hidden_layers(model, layers=None) -> int:
    """Return decoder layer count across model config variants."""
    text_cfg = _get_text_config(model)
    num_layers = getattr(text_cfg, "num_hidden_layers", None)
    if num_layers is not None:
        return int(num_layers)

    cfg_num_layers = getattr(model.config, "num_hidden_layers", None)
    if cfg_num_layers is not None:
        return int(cfg_num_layers)

    if layers is None:
        layers = get_transformer_layers(model)
    return int(len(layers))


def get_num_attention_heads(model) -> int:
    """Return configured attention head count from text or top-level config."""
    text_cfg = _get_text_config(model)
    num_heads = getattr(text_cfg, "num_attention_heads", None)
    if num_heads is not None:
        return int(num_heads)

    cfg_num_heads = getattr(model.config, "num_attention_heads", None)
    if cfg_num_heads is not None:
        return int(cfg_num_heads)

    raise ValueError("Model config missing num_attention_heads.")


def _resolve_layer_type_head_dim(text_cfg, layer_idx: int):
    """Resolve Gemma4 per-layer head dim when layer_types metadata exists."""
    layer_types = getattr(text_cfg, "layer_types", None)
    if not layer_types or layer_idx < 0 or layer_idx >= len(layer_types):
        return None

    layer_type = layer_types[layer_idx]
    base_head_dim = getattr(text_cfg, "head_dim", None)
    global_head_dim = getattr(text_cfg, "global_head_dim", None)

    if layer_type == "full_attention" and global_head_dim is not None:
        return int(global_head_dim)
    if base_head_dim is not None:
        return int(base_head_dim)
    return None


def get_attention_head_dim(model, layers=None, layer_idx=None) -> int:
    """Resolve attention head width for o_proj input slicing.

    Supports mixed-head-dim architectures (for example Gemma4) by accepting an
    optional layer index.
    """
    config = model.config
    text_cfg = _get_text_config(model)

    # Gemma4-style per-layer attention type metadata.
    if layer_idx is not None:
        layer_type_dim = _resolve_layer_type_head_dim(text_cfg, int(layer_idx))
        if layer_type_dim is not None:
            return int(layer_type_dim)

    cfg_head_dim = getattr(text_cfg, "head_dim", None)
    if cfg_head_dim is not None:
        return int(cfg_head_dim)
    cfg_head_dim = getattr(config, "head_dim", None)
    if cfg_head_dim is not None:
        return int(cfg_head_dim)

    num_heads = get_num_attention_heads(model)

    if layers is None:
        try:
            layers = get_transformer_layers(model)
        except Exception:
            layers = None

    # If layer-specific inference is requested and projection width is available,
    # infer using configured num_heads.
    if layer_idx is not None and layers is not None:
        li = int(layer_idx)
        if 0 <= li < len(layers):
            attn = getattr(layers[li], "self_attn", None)
            o_proj = getattr(attn, "o_proj", None)
            in_features = getattr(o_proj, "in_features", None)
            if in_features is not None and num_heads > 0 and in_features % num_heads == 0:
                return int(in_features // num_heads)

    if layers:
        first_attn = getattr(layers[0], "self_attn", None)
        o_proj = getattr(first_attn, "o_proj", None)
        in_features = getattr(o_proj, "in_features", None)
        if in_features is not None and num_heads > 0 and in_features % num_heads == 0:
            return int(in_features // num_heads)

    hidden_size = getattr(text_cfg, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None or num_heads <= 0 or hidden_size % num_heads != 0:
        raise ValueError(
            f"Cannot infer head_dim from hidden_size={hidden_size} and "
            f"num_attention_heads={num_heads}."
        )
    return int(hidden_size // num_heads)


def get_layer_head_layout(model, layers=None):
    """Return per-layer head dimensions and head counts.

    Returns:
        (layer_head_dims, layer_num_heads, max_num_heads)
    """
    if layers is None:
        layers = get_transformer_layers(model)

    num_layers = len(layers)
    cfg_num_heads = get_num_attention_heads(model)

    layer_head_dims = []
    layer_num_heads = []
    for li in range(num_layers):
        hdim = int(get_attention_head_dim(model, layers=layers, layer_idx=li))
        layer_head_dims.append(hdim)

        attn = getattr(layers[li], "self_attn", None)
        o_proj = getattr(attn, "o_proj", None)
        in_features = getattr(o_proj, "in_features", None)

        if in_features is not None and hdim > 0 and in_features % hdim == 0:
            layer_num_heads.append(int(in_features // hdim))
        else:
            layer_num_heads.append(int(cfg_num_heads))

    if not layer_num_heads:
        raise ValueError("Could not resolve per-layer head layout.")

    max_num_heads = max(layer_num_heads)
    return layer_head_dims, layer_num_heads, max_num_heads


def get_model_final_norm(model):
    """Return the decoder final norm module when available."""
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "norm")
    ):
        return model.model.language_model.norm
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f
    if hasattr(model, "transformer") and hasattr(model.transformer, "norm"):
        return model.transformer.norm
    return None


# ===================== Prompt Building =====================

def build_cot_prompt(question: str, answer: str) -> str:
    """Build a CoT-style full sequence for analysis."""
    return f"Question: {question}{SUFFIX_PROMPT}\n{answer}"


def build_chain_code_tokens(question: str, chain: str, code: str, tokenizer, max_seq_len: int = None) -> tuple:
    """Build token ids for Chain+Code analysis with exact code boundaries.

    Returns:
        (full_ids, code_start_idx, code_len)
    """
    prefix_text = f"Question: {question}{SUFFIX_PROMPT}\n"
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)

    chain_ids = tokenizer.encode(chain, add_special_tokens=False)
    if len(chain_ids) > CHAIN_TOKEN_LIMIT:
        chain_ids = chain_ids[-CHAIN_TOKEN_LIMIT:]

    code_ids = tokenizer.encode(code, add_special_tokens=False)
    if len(code_ids) > CODE_TOKEN_LIMIT:
        code_ids = code_ids[:CODE_TOKEN_LIMIT]

    # Truncate prefix from the front if total exceeds max_seq_len
    if max_seq_len is not None:
        total = len(prefix_ids) + len(chain_ids) + len(code_ids)
        if total > max_seq_len:
            excess = total - max_seq_len
            prefix_ids = prefix_ids[excess:]

    code_start_idx = len(prefix_ids) + len(chain_ids)
    full_ids = prefix_ids + chain_ids + code_ids
    return full_ids, code_start_idx, len(code_ids)


def build_chain_code_prompt(question: str, chain: str, code: str, tokenizer) -> tuple:
    """Backward-compatible text builder around token-id construction.

    Returns:
        (full_prompt_text, code_start_idx)
    """
    full_ids, code_start_idx, _ = build_chain_code_tokens(question, chain, code, tokenizer)
    full_text = tokenizer.decode(full_ids, skip_special_tokens=True)
    return full_text, code_start_idx


# ===================== Rating Range Helpers =====================

def parse_rating_range(rating_range: str):
    """Return (low, high) inclusive integer bounds for a rating range string."""
    if rating_range.startswith("<="):
        return (0, int(rating_range[2:]))
    if rating_range.startswith(">"):
        return (int(rating_range[1:]) + 1, float("inf"))
    low, high = rating_range.split("-")
    return (int(low), int(high))


def rating_range_to_outdir(rating_range: str, base_out_dir: str) -> str:
    """Return results/cf/<rating_range>/<base_out_dir>.

    If base_out_dir starts with results/, the prefix is stripped to avoid duplicates.
    """
    normalized = base_out_dir.replace("\\", "/")
    if normalized.startswith("results/"):
        normalized = normalized[len("results/"):]
    normalized = normalized.strip("/")
    if normalized:
        return os.path.join("results", "cf", rating_range, normalized)
    return os.path.join("results", "cf", rating_range)


def hml_dataset_to_outdir(dataset_name: str, base_out_dir: str) -> str:
    """Return results/hml/<dataset_name>/<base_out_dir>.

    Mirrors rating_range_to_outdir but for HML experiments.
    """
    normalized = base_out_dir.replace("\\", "/")
    if normalized.startswith("results/"):
        normalized = normalized[len("results/"):]
    normalized = normalized.strip("/")
    if normalized:
        return os.path.join("results", "hml", dataset_name, normalized)
    return os.path.join("results", "hml", dataset_name)


# ===================== Dataset Loading =====================

def _resolve_cf_path(cf_data_path: str) -> str:
    """Allow passing either a basename or a full/relative path."""
    if os.path.isabs(cf_data_path) or os.path.exists(cf_data_path):
        return cf_data_path
    return os.path.join("data", "datasets", cf_data_path)


def load_coding_dataset(
    name: str,
    num_samples: int,
    rating_range=None,
    cf_data_path: str = "cf_dataset.parquet",
    no_cot: bool = False,
    tokenizer=None,
    max_seq_len: int = None,
):
    """Load a coding dataset and return a list of sample dicts."""
    samples = []

    if name == "livecodebench":
        ds = load_dataset("cassanof/livecodebench_lite_filtered", split="test")
        for i, row in enumerate(ds):
            if i >= num_samples:
                break
            question = row.get("question", "").strip()
            starter = row.get("starter_code", "") or ""
            answer = starter.strip() if starter.strip() else "# solution\npass"
            samples.append({"question": question, "answer": answer})

    elif name == "mbpp":
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        for i, row in enumerate(ds):
            if i >= num_samples:
                break
            samples.append({"question": row["prompt"], "answer": row["code"]})

    elif name == "humaneval":
        ds = load_dataset("openai/openai_humaneval", split="test")
        for i, row in enumerate(ds):
            if i >= num_samples:
                break
            samples.append({
                "question": row["prompt"],
                "answer": row["canonical_solution"],
            })

    elif name == "codeforces":
        import pandas as pd

        cf_data_path = _resolve_cf_path(cf_data_path)
        if no_cot:
            cf_data_path = cf_data_path.replace(".parquet", "_no_cot.parquet")

        if not os.path.exists(cf_data_path):
            raise FileNotFoundError(
                f"Codeforces dataset not found at '{cf_data_path}'. "
                "Prepare the dataset first."
            )

        df = pd.read_parquet(cf_data_path)

        if rating_range is not None:
            if isinstance(rating_range, str):
                low, high = parse_rating_range(rating_range)
                df = df[(df["rating"] >= (low or 0)) & (df["rating"] <= (high or 9999))]
            elif isinstance(rating_range, list):
                mask = None
                for rr in rating_range:
                    low, high = parse_rating_range(rr)
                    range_mask = (df["rating"] >= (low or 0)) & (df["rating"] <= (high or 9999))
                    mask = range_mask if mask is None else (mask | range_mask)
                df = df[mask]

        df = df.dropna(subset=["rating", "prompt", "generation"])
        df = df.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)
        range_str = rating_range if isinstance(rating_range, str) else f"{len(rating_range)} ranges"
        print(f"[dataset] CF after dedup: {len(df)} unique problems in {range_str}")

        ans_token_cap = 1024
        import random
        # Shuffle first, then iterate until we collect enough valid samples
        df = df.sample(frac=1, random_state=random.getstate()[1][0]).reset_index(drop=True)

        valid_rows = []
        skipped = 0
        for _, row in df.iterrows():
            if len(valid_rows) >= num_samples:
                break
            gen = row["generation"]
            if tokenizer is not None:
                gen_ids = tokenizer.encode(gen, add_special_tokens=False)
                if len(gen_ids) > ans_token_cap:
                    gen = tokenizer.decode(gen_ids[:ans_token_cap], skip_special_tokens=True)
            if tokenizer is not None and max_seq_len is not None:
                tlen = len(tokenizer.encode(build_cot_prompt(row["prompt"], gen), add_special_tokens=False))
                if tlen > max_seq_len:
                    skipped += 1
                    continue
            valid_rows.append({
                "question": row["prompt"],
                "answer": gen,
                "id": row["id"],
                "rating": int(row["rating"]),
            })

        print(f"[dataset] CF: collected {len(valid_rows)} valid samples (skipped {skipped} over max_seq_len)")
        samples.extend(valid_rows)

    else:
        raise ValueError(f"Unknown dataset: {name}")

    suffix = f" (rating_range={rating_range})" if rating_range else ""
    print(f"[dataset] Loaded {len(samples)} samples from {name}{suffix} (out of {num_samples} requested)")
    return samples


def load_coding_dataset_chain_code(
    name: str,
    num_samples: int,
    rating_range=None,
    cf_data_path: str = "cf_dataset_chain_code.parquet",
    tokenizer=None,
    max_seq_len: int = None,
):
    """Load a coding dataset with chain+code separation."""
    samples = []

    if name != "codeforces":
        raise ValueError(f"Dataset {name} not supported for chain+code analysis")

    import pandas as pd

    cf_data_path = _resolve_cf_path(cf_data_path)
    if not os.path.exists(cf_data_path):
        raise FileNotFoundError(
            f"Codeforces chain+code dataset not found at '{cf_data_path}'. "
            "Prepare the chain+code dataset first."
        )

    df = pd.read_parquet(cf_data_path)

    if rating_range is not None:
        if isinstance(rating_range, str):
            low, high = parse_rating_range(rating_range)
            df = df[(df["rating"] >= (low or 0)) & (df["rating"] <= (high or 9999))]
        elif isinstance(rating_range, list):
            mask = None
            for rr in rating_range:
                low, high = parse_rating_range(rr)
                range_mask = (df["rating"] >= (low or 0)) & (df["rating"] <= (high or 9999))
                mask = range_mask if mask is None else (mask | range_mask)
            df = df[mask]

    df = df.dropna(subset=["rating", "prompt", "chain", "code"])
    df = df.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)
    range_str = rating_range if isinstance(rating_range, str) else f"{len(rating_range)} ranges"
    print(f"[dataset] CF after dedup: {len(df)} unique problems in {range_str}")

    if tokenizer is None:
        raise ValueError("tokenizer is required for chain+code dataset")

    import random
    # Shuffle first, then iterate until we collect enough valid samples
    df = df.sample(frac=1, random_state=random.getstate()[1][0]).reset_index(drop=True)

    valid_rows = []
    skipped = 0
    for _, row in df.iterrows():
        if len(valid_rows) >= num_samples:
            break
        question = row["prompt"]
        chain = row["chain"]
        code = row["code"]

        full_ids, code_start_idx, code_len = build_chain_code_tokens(question, chain, code, tokenizer)
        if max_seq_len is not None and len(full_ids) > max_seq_len:
            skipped += 1
            continue

        valid_rows.append({
            "question": question,
            "chain": chain,
            "code": code,
            "code_start_idx": code_start_idx,
            "code_len": code_len,
            "id": row["id"],
            "rating": int(row["rating"]),
        })

    print(f"[dataset] CF chain_code: collected {len(valid_rows)} valid samples (skipped {skipped} over max_seq_len)")
    samples.extend(valid_rows)

    suffix = f" (rating_range={rating_range})" if rating_range else ""
    print(f"[dataset] Loaded {len(samples)} samples from {name}{suffix} (out of {num_samples} requested)")
    return samples


# ===================== HML Dataset Loading =====================

def strip_python_tags(solution: str) -> str:
    """Remove ```python and ``` tags to get raw code."""
    code = solution.strip()
    if code.startswith("```python"):
        code = code[len("```python"):]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()


def load_hml_dataset(
    name: str,
    num_samples: int,
    hml_model_dir: str,
    seed: int = 42,
    no_cot: bool = False,
    tokenizer=None,
    max_seq_len: int = None,
):
    """Load HML dataset from model inference results.

    For regular (cot) mode: answer = reasoning_chain + code
    For no_cot mode: answer = code only
    """
    import json
    import random as _random

    if name not in HML_DATASET_CONFIG:
        raise ValueError(f"Unknown HML dataset: {name}. Must be one of {list(HML_DATASET_CONFIG.keys())}")

    cfg = HML_DATASET_CONFIG[name]
    filepath = os.path.join(hml_model_dir, cfg["file"])
    question_key = cfg["question_key"]
    test_key = cfg.get("test_key")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"HML dataset not found: {filepath}")

    with open(filepath) as f:
        data = json.load(f)

    # Filter entries that passed evaluation (verdict=1) and have a solution
    correct_entries = [e for e in data if e.get("verdict") == 1 and e.get("solution")]
    print(f"[dataset] HML {name}: {len(correct_entries)}/{len(data)} entries with verdict=1")

    # Fall back to all entries with a solution if too few correct ones
    if len(correct_entries) < 20:
        valid_entries = [e for e in data if e.get("solution")]
        print(f"[dataset] HML {name}: < 20 correct — falling back to all {len(valid_entries)} entries with solution")
    else:
        valid_entries = correct_entries

    # Deterministic sampling
    rng = _random.Random(seed)
    rng.shuffle(valid_entries)

    samples = []
    skipped = 0
    for entry in valid_entries:
        if len(samples) >= num_samples:
            break

        question = entry[question_key]
        if test_key and entry.get(test_key):
            question += "\n" + "\n".join(entry[test_key])
        chain = bpe_decode(entry.get("reasoning_chain", "") or "")
        solution = bpe_decode(entry.get("solution", "") or "")

        if not solution:
            skipped += 1
            continue

        if no_cot:
            answer = solution
        else:
            answer = (chain + "\n" + solution).strip() if chain else solution

        if tokenizer is not None and max_seq_len is not None:
            prefix = f"Question: {question}{SUFFIX_PROMPT}\n"
            prefix_len = len(tokenizer.encode(prefix, add_special_tokens=False))
            sol_len = len(tokenizer.encode(solution, add_special_tokens=False))
            tlen = len(tokenizer.encode(build_cot_prompt(question, answer), add_special_tokens=False))
            if tlen > max_seq_len and chain and not no_cot:
                # Truncate chain from the front to fit, preserving question and solution
                avail = max_seq_len - prefix_len - sol_len
                if avail > 0:
                    chain_ids = tokenizer.encode(chain, add_special_tokens=False)[-avail:]
                    chain = tokenizer.decode(chain_ids, skip_special_tokens=True)
                else:
                    chain = ""
                answer = (chain + "\n" + solution).strip() if chain else solution

        samples.append({"question": question, "answer": answer})

    mode_str = "no_cot" if no_cot else "regular"
    print(f"[dataset] HML {name} ({mode_str}): {len(samples)} samples (skipped {skipped}, seed={seed})")
    return samples


def load_hml_dataset_chain_code(
    name: str,
    num_samples: int,
    hml_model_dir: str,
    seed: int = 42,
    tokenizer=None,
    max_seq_len: int = None,
):
    """Load HML dataset with chain+code separation for code-only metrics."""
    import json
    import random as _random

    if name not in HML_DATASET_CONFIG:
        raise ValueError(f"Unknown HML dataset: {name}")

    if tokenizer is None:
        raise ValueError("tokenizer is required for chain+code dataset")

    cfg = HML_DATASET_CONFIG[name]
    filepath = os.path.join(hml_model_dir, cfg["file"])
    question_key = cfg["question_key"]
    test_key = cfg.get("test_key")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"HML dataset not found: {filepath}")

    with open(filepath) as f:
        data = json.load(f)

    correct_entries = [e for e in data if e.get("verdict") == 1 and e.get("solution")]
    print(f"[dataset] HML {name}: {len(correct_entries)}/{len(data)} entries with verdict=1")

    if len(correct_entries) < 20:
        valid_entries = [e for e in data if e.get("solution")]
        print(f"[dataset] HML {name}: < 20 correct — falling back to all {len(valid_entries)} entries with solution")
    else:
        valid_entries = correct_entries

    rng = _random.Random(seed)
    rng.shuffle(valid_entries)

    samples = []
    skipped = 0
    for entry in valid_entries:
        if len(samples) >= num_samples:
            break

        question = entry[question_key]
        if test_key and entry.get(test_key):
            question += "\n" + "\n".join(entry[test_key])
        chain = bpe_decode(entry.get("reasoning_chain", "") or "")
        solution = bpe_decode(entry.get("solution", "") or "")

        if not solution:
            skipped += 1
            continue

        full_ids, code_start_idx, code_len = build_chain_code_tokens(question, chain, solution, tokenizer, max_seq_len=max_seq_len)

        samples.append({
            "question": question,
            "chain": chain,
            "code": solution,
            "code_start_idx": code_start_idx,
            "code_len": code_len,
        })

    print(f"[dataset] HML {name} (chain_code): {len(samples)} samples (skipped {skipped}, seed={seed})")
    return samples


# ===================== Model Loading =====================

def load_model_and_tokenizer(model_name: str):
    """Load a HuggingFace CausalLM in bfloat16 with device_map=auto."""
    print(f"[model] Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_transformer_layers(model):
    """Extract transformer decoder layers from common CausalLM model wrappers."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if (
        hasattr(model, "model")
        and hasattr(model.model, "language_model")
        and hasattr(model.model.language_model, "layers")
    ):
        return model.model.language_model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError(f"Cannot find transformer layers in {type(model)}")


def free_model(model, tokenizer):
    """Delete model + tokenizer and release GPU memory."""
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ===================== Per-token Loss =====================

def compute_per_token_loss(logits, label_ids):
    """Compute per-token cross-entropy loss."""
    logits_flat = logits.float().view(-1, logits.size(-1))
    labels_flat = label_ids.view(-1)
    return F.cross_entropy(logits_flat, labels_flat, reduction="none").cpu().numpy()


# ===================== Hook Management =====================

def remove_all_hooks(model):
    """Remove all forward/backward hooks registered on any submodule."""
    for module in model.modules():
        module._forward_hooks.clear()
        module._forward_pre_hooks.clear()
        module._backward_hooks.clear()
