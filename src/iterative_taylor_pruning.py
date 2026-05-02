"""
Iterative Taylor Attribution Head Pruning
==========================================
Each iteration:
  1. Attach learnable mask (ones) to all *surviving* heads.
  2. Forward+backward over the cached 180-sample dataset.
  3. Rank surviving heads by |grad|; remove the worst `prune_per_iter`.
  4. Evaluate and record the loss after pruning.

Repeat for `num_iters` iterations.
"""

import argparse
import hashlib
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from coding_utils import (
    build_chain_code_tokens,
    build_cot_prompt,
    free_model,
    get_layer_head_layout,
    get_num_hidden_layers,
    get_transformer_layers,
    hml_dataset_to_outdir,
    load_hml_dataset,
    load_hml_dataset_chain_code,
    load_model_and_tokenizer,
    remove_all_hooks,
    HML_DATASETS,
)


BATCH_SIZE = 32
# Use micro-batches for gradient pass to reduce peak activation memory.
GRAD_BATCH_SIZE = 1
MAX_SEQ_LEN = 3072
# Optional cap on answer/code tokens used for loss to reduce lm_head memory.
MAX_LOSS_TOKENS = None
CACHE_DIR = os.path.join("data", "cached_datasets")


# ===================== Dataset Loading =====================

def _cache_path(model_name, num_samples_per_type=10, seed=42):
    key = f"mixed_{model_name}_{num_samples_per_type}_{seed}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return os.path.join(CACHE_DIR, f"mixed_dataset_{num_samples_per_type}spt_seed{seed}_{h}.json")


def load_cached_dataset(model_name, num_samples_per_type=10, seed=42):
    cache_file = _cache_path(model_name, num_samples_per_type, seed)
    if not os.path.exists(cache_file):
        raise FileNotFoundError(
            f"Cached dataset not found at {cache_file}. "
            f"Run optimize_head_subset_cmaes.py first to build and cache the dataset."
        )
    with open(cache_file, "r") as f:
        mixed = json.load(f)
    print(f"Loaded {len(mixed)} samples from {cache_file}")
    return mixed


def load_hml_samples(dataset_name, hml_model_dir, num_samples, tokenizer, mode, seed=42):
    """Load HML dataset as a list for a single mode (regular or chain_code)."""
    if mode == "regular":
        raw = load_hml_dataset(
            dataset_name, num_samples, hml_model_dir,
            seed=seed, tokenizer=tokenizer, max_seq_len=MAX_SEQ_LEN,
        )
        return [{"type": "regular", "question": s["question"], "answer": s["answer"]} for s in raw]
    elif mode == "chain_code":
        raw = load_hml_dataset_chain_code(
            dataset_name, num_samples, hml_model_dir,
            seed=seed, tokenizer=tokenizer, max_seq_len=MAX_SEQ_LEN,
        )
        return [{"type": "chain_code", "question": s["question"], "chain": s["chain"], "code": s["code"]} for s in raw]
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ===================== Loss Evaluation (no grad) =====================

@torch.no_grad()
def evaluate_loss(model, tokenizer, layers, layer_head_dims, layer_num_heads, alive_heads, mixed_dataset):
    """Compute average loss with only alive_heads active (rest zeroed)."""
    num_layers = len(layers)

    # Register ablation hooks for dead heads
    hooks = []
    for layer_idx in range(num_layers):
        n_heads_layer = layer_num_heads[layer_idx]
        dead = [h for h in range(n_heads_layer) if (layer_idx, h) not in alive_heads]
        if dead:
            head_dim = layer_head_dims[layer_idx]
            keep_heads = torch.ones(n_heads_layer, device=model.device, dtype=model.dtype)
            keep_heads[dead] = 0

            def make_hook(keep_row, nhd, hdim):
                def hook_fn(module, args):
                    x = args[0]
                    bs, seq_len, proj_dim = x.shape
                    expected = nhd * hdim
                    if proj_dim != expected:
                        raise RuntimeError(
                            f"o_proj input dim mismatch: got {proj_dim}, "
                            f"expected num_heads*head_dim={expected}."
                        )
                    x = x.reshape(bs, seq_len, nhd, hdim)
                    x = x * keep_row.to(dtype=x.dtype)[None, None, :, None]
                    x = x.reshape(bs, seq_len, expected)
                    return (x,)
                return hook_fn

            o_proj = layers[layer_idx].self_attn.o_proj
            hooks.append(o_proj.register_forward_pre_hook(
                make_hook(keep_heads, n_heads_layer, head_dim)
            ))

    total_loss, n_valid = _run_forward(model, tokenizer, mixed_dataset, grad=False)

    for h in hooks:
        h.remove()
    remove_all_hooks(model)

    return total_loss / max(n_valid, 1)


# ===================== Taylor Scoring (one pass with grad) =====================

def taylor_score(
    model,
    tokenizer,
    layers,
    layer_head_dims,
    layer_num_heads,
    max_num_heads,
    alive_heads,
    mixed_dataset,
):
    """Attach mask to alive heads, forward+backward, return importance dict."""
    num_layers = len(layers)

    # Build mask: only alive heads get a learnable entry
    mask = torch.nn.Parameter(
        torch.ones(num_layers, max_num_heads, device=model.device, dtype=model.dtype)
    )

    # Freeze model
    for p in model.parameters():
        p.requires_grad = False

    # Hook: multiply by mask row, but force dead heads to 0
    hooks = []
    for layer_idx in range(num_layers):
        n_heads_layer = layer_num_heads[layer_idx]
        head_dim = layer_head_dims[layer_idx]
        alive_in_layer = [h for h in range(n_heads_layer) if (layer_idx, h) in alive_heads]
        alive_head_mask = torch.zeros(n_heads_layer, device=model.device, dtype=model.dtype)
        if alive_in_layer:
            alive_head_mask[alive_in_layer] = 1

        def make_hook(mask_row, alive_row, nhd, hdim):
            def hook_fn(module, args):
                x = args[0]
                bs, seq_len, proj_dim = x.shape
                expected = nhd * hdim
                if proj_dim != expected:
                    raise RuntimeError(
                        f"o_proj input dim mismatch: got {proj_dim}, "
                        f"expected num_heads*head_dim={expected}."
                    )
                x = x.reshape(bs, seq_len, nhd, hdim)
                effective_mask = mask_row * alive_row
                x = x * effective_mask[None, None, :, None]
                x = x.reshape(bs, seq_len, expected)
                return (x,)
            return hook_fn

        o_proj = layers[layer_idx].self_attn.o_proj
        hooks.append(o_proj.register_forward_pre_hook(
            make_hook(mask[layer_idx, :n_heads_layer], alive_head_mask, n_heads_layer, head_dim)
        ))

    # Forward + backward
    was_training = model.training
    model.train()
    try:
        total_loss, n_valid = _run_forward(model, tokenizer, mixed_dataset, grad=True, mask=mask)
    finally:
        if not was_training:
            model.eval()
        for h in hooks:
            h.remove()
        remove_all_hooks(model)

    # Extract importance for alive heads only.
    # Use absolute-gradient accumulator to avoid micro-batch cancellation.
    grad_tensor = mask.abs_grad if hasattr(mask, "abs_grad") else mask.grad
    if grad_tensor is None:
        raise RuntimeError("Taylor scoring failed: mask gradient is missing.")

    grad = grad_tensor.detach().float().cpu()
    alive_by_layer = {}
    for layer, head in alive_heads:
        alive_by_layer.setdefault(layer, []).append(head)

    importance = {}
    for layer_idx, heads in alive_by_layer.items():
        heads = sorted(heads)
        for head in heads:
            importance[(layer_idx, head)] = abs(float(grad[layer_idx, head]))

    avg_loss = total_loss / max(n_valid, 1)
    return importance, avg_loss

# ===================== Shared Forward Logic =====================

def _run_forward(model, tokenizer, mixed_dataset, grad=False, mask=None):
    """Run forward (and optionally backward) over the dataset. Returns (total_loss, n_valid).

    OOM handling: on torch.cuda.OutOfMemoryError the global BATCH_SIZE is halved and
    the failing batch is retried. During grad=True runs, we use a smaller
    GRAD_BATCH_SIZE micro-batch to reduce peak activation memory.
    """
    global BATCH_SIZE
    import gc

    samples_by_type = {"chain_code": [], "regular": [], "no_cot": []}
    for sample in mixed_dataset:
        samples_by_type[sample["type"]].append(sample)

    total_loss = 0.0
    n_valid = 0
    needs_mm_token_type_ids = False

    def _cap_target_span(start_idx, span_len):
        if MAX_LOSS_TOKENS is None or span_len <= MAX_LOSS_TOKENS:
            return start_idx, span_len
        capped_len = int(MAX_LOSS_TOKENS)
        # Keep suffix tokens to preserve the end-of-answer/code behavior.
        return start_idx + (span_len - capped_len), capped_len

    def _forward_hidden_states(batch_tokens, batch_mask):
        nonlocal needs_mm_token_type_ids
        if needs_mm_token_type_ids:
            return model.model(
                batch_tokens,
                attention_mask=batch_mask,
                mm_token_type_ids=torch.zeros_like(batch_tokens, dtype=torch.long),
                return_dict=False,
            )[0]

        try:
            return model.model(batch_tokens, attention_mask=batch_mask, return_dict=False)[0]
        except ValueError as e:
            # Gemma4 requires mm_token_type_ids in training mode; use all-zero ids
            # to indicate pure-text tokens during Taylor gradient passes.
            if "mm_token_type_ids" in str(e):
                needs_mm_token_type_ids = True
                return model.model(
                    batch_tokens,
                    attention_mask=batch_mask,
                    mm_token_type_ids=torch.zeros_like(batch_tokens, dtype=torch.long),
                    return_dict=False,
                )[0]
            raise

    # --- chain_code ---
    chain_batch = []
    for sample in samples_by_type["chain_code"]:
        full_ids, code_start_idx, code_len = build_chain_code_tokens(
            sample["question"], sample["chain"], sample["code"], tokenizer,
            max_seq_len=MAX_SEQ_LEN,
        )
        if code_start_idx > 0 and code_start_idx < len(full_ids) and code_len > 0:
            chain_batch.append((full_ids, code_start_idx, code_len))

    batch_start = 0
    while batch_start < len(chain_batch):
        current_batch_size = min(BATCH_SIZE, GRAD_BATCH_SIZE) if grad else BATCH_SIZE
        batch_end = min(batch_start + current_batch_size, len(chain_batch))
        batch = chain_batch[batch_start:batch_end]
        max_len = max(len(ids) for ids, _, _ in batch)
        pad_id = tokenizer.pad_token_id or 0

        padded_ids, attention_masks, adjusted_ranges = [], [], []
        for ids, code_start, code_len in batch:
            pad_len = max_len - len(ids)
            padded_ids.append([pad_id] * pad_len + ids)
            attention_masks.append([0] * pad_len + [1] * len(ids))
            adjusted_ranges.append((code_start + pad_len, code_len))

        batch_tokens = torch.tensor(padded_ids, device=model.device)
        batch_mask = torch.tensor(attention_masks, device=model.device, dtype=torch.bool)

        try:
            if grad:
                hidden_states = _forward_hidden_states(batch_tokens, batch_mask)
                batch_loss_sum = None
                batch_loss_count = 0
                for i, (code_start, code_len) in enumerate(adjusted_ranges):
                    code_start, code_len = _cap_target_span(code_start, code_len)
                    sliced_hidden = hidden_states[i : i + 1, code_start - 1 : code_start - 1 + code_len, :]
                    if sliced_hidden.shape[1] != code_len:
                        continue
                    code_logits = model.lm_head(sliced_hidden)
                    code_tokens = batch_tokens[i : i + 1, code_start : code_start + code_len]
                    loss = F.cross_entropy(
                        code_logits.reshape(-1, code_logits.size(-1)),
                        code_tokens.reshape(-1),
                    )
                    batch_loss_sum = loss if batch_loss_sum is None else (batch_loss_sum + loss)
                    batch_loss_count += 1
                    n_valid += 1
                if batch_loss_count > 0:
                    final_batch_loss = batch_loss_sum / batch_loss_count
                    final_batch_loss.backward()
                    if mask is not None and mask.grad is not None:
                        if not hasattr(mask, "abs_grad"):
                            mask.abs_grad = torch.zeros_like(mask)
                        mask.abs_grad += mask.grad.abs()
                        mask.grad.zero_()
                    total_loss += float(final_batch_loss.detach()) * batch_loss_count
                del hidden_states
            else:
                with torch.no_grad():
                    hidden_states = _forward_hidden_states(batch_tokens, batch_mask)
                    for i, (code_start, code_len) in enumerate(adjusted_ranges):
                        code_start, code_len = _cap_target_span(code_start, code_len)
                        sliced_hidden = hidden_states[i : i + 1, code_start - 1 : code_start - 1 + code_len, :]
                        if sliced_hidden.shape[1] != code_len:
                            continue
                        code_logits = model.lm_head(sliced_hidden)
                        code_tokens = batch_tokens[i : i + 1, code_start : code_start + code_len]
                        per_token = F.cross_entropy(
                            code_logits.reshape(-1, code_logits.size(-1)),
                            code_tokens.reshape(-1),
                            reduction="none",
                        )
                        total_loss += float(per_token.mean())
                        n_valid += 1
                del hidden_states

            torch.cuda.empty_cache()
            batch_start = batch_end

        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "out of memory" in str(e).lower() and BATCH_SIZE > 1:
                BATCH_SIZE = max(1, BATCH_SIZE // 2)
                print(f"[OOM] Reducing GLOBAL batch size to {BATCH_SIZE} and retrying chain_code batch {batch_start}...")
                gc.collect()
                torch.cuda.empty_cache()
            else:
                raise

    # --- regular and no_cot ---
    for sample_type in ["regular", "no_cot"]:
        tokenized = []
        for sample in samples_by_type[sample_type]:
            text = build_cot_prompt(sample["question"], sample["answer"])
            tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            answer_tokens = tokenizer(sample["answer"], return_tensors="pt", add_special_tokens=False)["input_ids"][0]
            ans_len = len(answer_tokens)
            if ans_len > 0 and len(tokens) > ans_len:
                tokenized.append((tokens.tolist(), answer_tokens.tolist(), ans_len))

        batch_start = 0
        while batch_start < len(tokenized):
            current_batch_size = min(BATCH_SIZE, GRAD_BATCH_SIZE) if grad else BATCH_SIZE
            batch_end = min(batch_start + current_batch_size, len(tokenized))
            batch = tokenized[batch_start:batch_end]
            max_len = max(len(ids) for ids, _, _ in batch)
            pad_id = tokenizer.pad_token_id or 0

            padded_ids, attention_masks, answer_data = [], [], []
            for ids, ans_ids, ans_len in batch:
                pad_len = max_len - len(ids)
                padded_ids.append([pad_id] * pad_len + ids)
                attention_masks.append([0] * pad_len + [1] * len(ids))
                answer_data.append((ans_ids, ans_len))

            batch_tokens = torch.tensor(padded_ids, device=model.device)
            batch_mask = torch.tensor(attention_masks, device=model.device, dtype=torch.bool)

            try:
                if grad:
                    hidden_states = _forward_hidden_states(batch_tokens, batch_mask)
                    batch_loss_sum = None
                    batch_loss_count = 0
                    for i, (ans_ids, ans_len) in enumerate(answer_data):
                        if MAX_LOSS_TOKENS is not None and ans_len > MAX_LOSS_TOKENS:
                            ans_ids = ans_ids[-MAX_LOSS_TOKENS:]
                            ans_len = MAX_LOSS_TOKENS
                        sliced_hidden = hidden_states[i : i + 1, -ans_len - 1 : -1, :]
                        if sliced_hidden.shape[1] != ans_len:
                            continue
                        ans_logits = model.lm_head(sliced_hidden)
                        ans_target = torch.tensor(ans_ids, device=model.device)
                        loss = F.cross_entropy(
                            ans_logits.reshape(-1, ans_logits.size(-1)),
                            ans_target.reshape(-1),
                        )
                        batch_loss_sum = loss if batch_loss_sum is None else (batch_loss_sum + loss)
                        batch_loss_count += 1
                        n_valid += 1
                    if batch_loss_count > 0:
                        final_batch_loss = batch_loss_sum / batch_loss_count
                        final_batch_loss.backward()
                        if mask is not None and mask.grad is not None:
                            if not hasattr(mask, "abs_grad"):
                                mask.abs_grad = torch.zeros_like(mask)
                            mask.abs_grad += mask.grad.abs()
                            mask.grad.zero_()
                        total_loss += float(final_batch_loss.detach()) * batch_loss_count
                    del hidden_states
                else:
                    with torch.no_grad():
                        hidden_states = _forward_hidden_states(batch_tokens, batch_mask)
                        for i, (ans_ids, ans_len) in enumerate(answer_data):
                            if MAX_LOSS_TOKENS is not None and ans_len > MAX_LOSS_TOKENS:
                                ans_ids = ans_ids[-MAX_LOSS_TOKENS:]
                                ans_len = MAX_LOSS_TOKENS
                            sliced_hidden = hidden_states[i : i + 1, -ans_len - 1 : -1, :]
                            if sliced_hidden.shape[1] != ans_len:
                                continue
                            ans_logits = model.lm_head(sliced_hidden)
                            ans_target = torch.tensor(ans_ids, device=model.device)
                            per_token = F.cross_entropy(
                                ans_logits.reshape(-1, ans_logits.size(-1)),
                                ans_target.reshape(-1),
                                reduction="none",
                            )
                            total_loss += float(per_token.mean())
                            n_valid += 1
                    del hidden_states

                torch.cuda.empty_cache()
                batch_start = batch_end

            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                if "out of memory" in str(e).lower() and BATCH_SIZE > 1:
                    BATCH_SIZE = max(1, BATCH_SIZE // 2)
                    print(f"[OOM] Reducing GLOBAL batch size to {BATCH_SIZE} and retrying {sample_type} batch {batch_start}...")
                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    raise

    return total_loss, n_valid


# ===================== Pruning Loop =====================

def run_pruning(model, tokenizer, dataset, out_dir, num_iters, prune_per_iter, model_name, dataset_name=None, mode=None):
    """Run iterative Taylor pruning on a single dataset. Returns history list."""
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    torch.set_grad_enabled(False)

    layers = get_transformer_layers(model)
    num_layers = get_num_hidden_layers(model, layers=layers)
    layer_head_dims, layer_num_heads, max_heads = get_layer_head_layout(model, layers=layers)
    total_heads = sum(layer_num_heads)

    unique_dims = sorted(set(layer_head_dims))
    if len(unique_dims) == 1:
        print(f"Architecture: {num_layers} layers x <= {max_heads} heads, head_dim={unique_dims[0]} ({total_heads} total)")
    else:
        print(f"Architecture: {num_layers} layers x <= {max_heads} heads, mixed head_dim={unique_dims} ({total_heads} total)")
    print(f"Pruning plan: {prune_per_iter} heads/iter x {num_iters} iters = {prune_per_iter * num_iters} heads removed")

    alive_heads = set((l, h) for l in range(num_layers) for h in range(layer_num_heads[l]))

    print("\nComputing baseline loss (all heads active)...")
    baseline_loss = evaluate_loss(model, tokenizer, layers, layer_head_dims, layer_num_heads, alive_heads, dataset)
    print(f"Baseline loss: {baseline_loss:.6f}")

    history = [{
        "iteration": 0,
        "alive_heads": total_heads,
        "pruned_this_iter": [],
        "loss": baseline_loss,
    }]

    print(f"\n{'Iter':>4} | {'Alive':>6} | {'Loss':>10} | {'Delta':>10} | Pruned heads")
    print("-" * 75)
    print(f"   0 | {total_heads:6d} | {baseline_loss:10.6f} | {'':>10} | (baseline)")

    # Build a lightweight args-like object for _save_results
    class _SaveCtx:
        pass
    ctx = _SaveCtx()
    ctx.out_dir = out_dir
    ctx.model = model_name
    ctx.num_iters = num_iters
    ctx.prune_per_iter = prune_per_iter
    ctx.dataset = dataset_name
    ctx.mode = mode

    start_time = time.time()

    for iteration in range(1, num_iters + 1):
        if len(alive_heads) <= prune_per_iter:
            print(f"\nStopping early: only {len(alive_heads)} heads remain.")
            break

        torch.set_grad_enabled(True)
        importance, _ = taylor_score(
            model,
            tokenizer,
            layers,
            layer_head_dims,
            layer_num_heads,
            max_heads,
            alive_heads,
            dataset,
        )
        torch.set_grad_enabled(False)

        ranked = sorted(importance.items(), key=lambda x: x[1])
        to_prune = [head for head, _ in ranked[:prune_per_iter]]

        for head in to_prune:
            alive_heads.discard(head)

        loss = evaluate_loss(model, tokenizer, layers, layer_head_dims, layer_num_heads, alive_heads, dataset)
        delta = loss - baseline_loss

        pruned_str = ", ".join(f"L{l}H{h}" for l, h in to_prune)
        print(f"{iteration:4d} | {len(alive_heads):6d} | {loss:10.6f} | {delta:+10.6f} | {pruned_str}")

        history.append({
            "iteration": iteration,
            "alive_heads": len(alive_heads),
            "pruned_this_iter": [[l, h] for l, h in to_prune],
            "loss": loss,
        })

        if iteration % 10 == 0:
            _save_results(ctx, history, alive_heads, baseline_loss, num_layers, layer_num_heads, layer_head_dims, start_time)

    elapsed = time.time() - start_time
    _save_results(ctx, history, alive_heads, baseline_loss, num_layers, layer_num_heads, layer_head_dims, start_time)

    print(f"\n{'='*75}")
    print(f"Completed in {elapsed:.1f}s")
    print(f"Heads remaining: {len(alive_heads)} / {total_heads}")
    print(f"Final loss: {history[-1]['loss']:.6f}  (baseline: {baseline_loss:.6f})")
    print(f"{'='*75}")

    print(f"\n{'Iter':>4} | {'Alive':>6} | {'Loss':>10} | {'Delta':>10}")
    print("-" * 40)
    for entry in history:
        it = entry["iteration"]
        al = entry["alive_heads"]
        lo = entry["loss"]
        delta_str = f"{lo - baseline_loss:+10.6f}" if it > 0 else f"{'':>10}"
        print(f"{it:4d} | {al:6d} | {lo:10.6f} | {delta_str}")

    # Clear hooks to prepare for next run
    remove_all_hooks(model)
    return history


# ===================== Main =====================

HML_MODES = ["regular", "chain_code"]

def main():
    global BATCH_SIZE, GRAD_BATCH_SIZE, MAX_SEQ_LEN, MAX_LOSS_TOKENS

    parser = argparse.ArgumentParser(description="Iterative Taylor attribution head pruning")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B",
                        help='Model name, or "all" for both HML models')
    parser.add_argument("--num_iters", type=int, default=100, help="Number of pruning iterations")
    parser.add_argument("--prune_per_iter", type=int, default=5, help="Heads to prune per iteration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_samples", type=int, default=100, help="Max samples per type for HML mode")
    parser.add_argument("--dataset", type=str, default=None, choices=HML_DATASETS,
                        help="HML dataset (enables HML mode)")
    parser.add_argument("--hml_model_dir", type=str, default=None,
                        help="Path to HML model results dir (e.g. data/benchmarks/Qwen--Qwen3-4B)")
    parser.add_argument("--mode", type=str, default=None, choices=HML_MODES,
                        help="Run only this mode (default: both regular and chain_code for HML)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: auto-resolved)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Batch size for eval/no-grad forward passes")
    parser.add_argument("--grad_batch_size", type=int, default=GRAD_BATCH_SIZE,
                        help="Micro-batch size for grad pass (lower reduces VRAM)")
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN,
                        help="Maximum sequence length during tokenization")
    parser.add_argument("--max_loss_tokens", type=int, default=MAX_LOSS_TOKENS,
                        help="Optional cap on target answer/code tokens used for loss")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch_size must be >= 1")
    if args.grad_batch_size < 1:
        parser.error("--grad_batch_size must be >= 1")
    if args.max_seq_len < 1:
        parser.error("--max_seq_len must be >= 1")
    if args.max_loss_tokens is not None and args.max_loss_tokens < 1:
        parser.error("--max_loss_tokens must be >= 1 when provided")

    BATCH_SIZE = args.batch_size
    GRAD_BATCH_SIZE = args.grad_batch_size
    MAX_SEQ_LEN = args.max_seq_len
    MAX_LOSS_TOKENS = args.max_loss_tokens

    is_hml = args.hml_model_dir is not None
    if is_hml and args.dataset is None:
        parser.error("--dataset is required when using --hml_model_dir")

    # Resolve model list
    if args.model == "all":
        models = ["Qwen/Qwen3-4B", "meta-llama/Llama-3.2-3B-Instruct"]
    else:
        models = [args.model]

    # Resolve mode list
    if is_hml:
        modes = [args.mode] if args.mode else HML_MODES
    else:
        modes = [None]  # CF mode: single pass with mixed cached dataset

    for model_name in models:
        model_short = model_name.split("/")[-1]

        if is_hml:
            # Resolve hml_model_dir per model when --model all
            if args.model == "all":
                hml_model_dir = os.path.join(
                    "data", "benchmarks", model_name.replace("/", "--")
                )
            else:
                hml_model_dir = args.hml_model_dir
        else:
            hml_model_dir = None

        print(f"\n{'='*75}")
        print(f"Loading model: {model_name}")
        print(f"{'='*75}")
        model, tokenizer = load_model_and_tokenizer(model_name)

        for mode in modes:
            if is_hml:
                mode_suffix = "" if mode == "regular" else f"_{mode}"
                out_dir = args.out_dir or (
                    hml_dataset_to_outdir(args.dataset, f"iterative_taylor{mode_suffix}")
                    + "/" + model_short
                )

                print(f"\n{'='*75}")
                print(f"  Dataset: {args.dataset} | Mode: {mode} | Model: {model_short}")
                print(f"  Output: {out_dir}")
                print(f"{'='*75}")

                dataset = load_hml_samples(
                    args.dataset, hml_model_dir, args.num_samples,
                    tokenizer, mode=mode, seed=args.seed,
                )
                run_pruning(
                    model, tokenizer, dataset, out_dir,
                    args.num_iters, args.prune_per_iter, model_name,
                    dataset_name=args.dataset, mode=mode,
                )
            else:
                out_dir = args.out_dir or os.path.join("results", "iterative_taylor", model_short)
                dataset = load_cached_dataset(model_name, num_samples_per_type=10, seed=args.seed)
                run_pruning(
                    model, tokenizer, dataset, out_dir,
                    args.num_iters, args.prune_per_iter, model_name,
                )

        free_model(model, tokenizer)
        print(f"\n=== Done with {model_short}! ===")


def _save_results(args, history, alive_heads, baseline_loss, num_layers, layer_num_heads, layer_head_dims, start_time):
    os.makedirs(args.out_dir, exist_ok=True)
    max_heads = max(layer_num_heads)

    # JSON with full history
    json_path = os.path.join(args.out_dir, "iterative_taylor_results.json")
    with open(json_path, "w") as f:
        result = {
            "model": args.model,
            "num_iters": args.num_iters,
            "prune_per_iter": args.prune_per_iter,
            "baseline_loss": baseline_loss,
            "num_layers": num_layers,
            "num_heads": max_heads,
            "layer_num_heads": list(layer_num_heads),
            "layer_head_dims": list(layer_head_dims),
            "elapsed_seconds": time.time() - start_time,
            "history": history,
            "alive_heads": sorted([[l, h] for l, h in alive_heads]),
        }
        if getattr(args, "dataset", None):
            result["dataset"] = args.dataset
        if getattr(args, "mode", None):
            result["mode"] = args.mode
        json.dump(result, f, indent=2)

    # NPZ for easy numpy loading
    npz_path = os.path.join(args.out_dir, "iterative_taylor_results.npz")
    iters = [e["iteration"] for e in history]
    alive_counts = [e["alive_heads"] for e in history]
    losses = [e["loss"] for e in history]
    np.savez_compressed(
        npz_path,
        iterations=np.array(iters),
        alive_counts=np.array(alive_counts),
        losses=np.array(losses),
        alive_heads=np.array(sorted([[l, h] for l, h in alive_heads]), dtype=np.int32),
        baseline_loss=np.array([baseline_loss]),
        layer_num_heads=np.array(layer_num_heads, dtype=np.int32),
        layer_head_dims=np.array(layer_head_dims, dtype=np.int32),
    )
    print(f"  [checkpoint] Saved to {json_path} and {npz_path}")


if __name__ == "__main__":
    main()
