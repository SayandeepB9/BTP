"""
Attention Head Ablation on Coding Datasets
==========================================
Supports three modes:
  - regular: standard CoT answer-token ablation
  - no_cot: answer-token ablation on no-cot dataset variant
  - chain_code: code-token-only ablation on chain+code dataset
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from coding_utils import (
    ALL_DATASETS,
    ALL_MODELS,
    HML_DATASETS,
    RATING_RANGES,
    build_chain_code_tokens,
    build_cot_prompt,
    compute_per_token_loss,
    free_model,
    get_layer_head_layout,
    get_num_hidden_layers,
    get_transformer_layers,
    load_coding_dataset,
    load_coding_dataset_chain_code,
    load_hml_dataset,
    load_hml_dataset_chain_code,
    load_model_and_tokenizer,
    rating_range_to_outdir,
    remove_all_hooks,
)


# ===================== Performance Tuning =====================

BATCH_SIZE = 32
TOP_K_HEADS = 100
MAX_SEQ_LEN = 3072


# ===================== Hook Factories =====================

def make_batched_ablate_pre_hook(head_indices, head_dim):
    """Pre-hook for o_proj: zeroes a different head for each sequence in a batch."""
    def hook_fn(module, args):
        x = args[0].clone()
        for b, hi in enumerate(head_indices):
            start = hi * head_dim
            end = (hi + 1) * head_dim
            x[b, :, start:end] = 0
        return (x,)

    return hook_fn


def make_ablate_heads_pre_hook(head_indices, head_dim):
    """Pre-hook for o_proj: zeroes out multiple heads on every sequence."""
    def hook_fn(module, args):
        x = args[0].clone()
        for hi in head_indices:
            start = hi * head_dim
            end = (hi + 1) * head_dim
            x[:, :, start:end] = 0
        return (x,)

    return hook_fn


def _iter_valid_heads(layer_num_heads):
    """Yield (layer_idx, head_idx) for all valid heads."""
    for layer_idx, n_heads in enumerate(layer_num_heads):
        for head_idx in range(n_heads):
            yield (layer_idx, head_idx)


def _rank_valid_heads(ablation_result, layer_num_heads):
    """Return valid heads sorted by descending importance."""
    ranked = [
        (li, hi, ablation_result[li, hi].item())
        for li, hi in _iter_valid_heads(layer_num_heads)
    ]
    ranked.sort(key=lambda x: x[2], reverse=True)
    return ranked


# ===================== Helpers =====================

def _save_phase1_outputs(
    out_dir,
    model_short,
    dataset_name,
    num_layers,
    layer_num_heads,
    layer_head_dims,
    tot,
    ablation_result,
    per_head_token_deltas,
    title_suffix="",
    x_label="Per-token delta loss",
):
    run_dir = os.path.join(out_dir, model_short, dataset_name)
    os.makedirs(run_dir, exist_ok=True)
    npz_path = os.path.join(run_dir, "ablation.npz")

    save_dict = {
        "ablation_mean": ablation_result.numpy(),
        "num_layers": np.array([num_layers]),
        "num_heads": np.array([max(layer_num_heads)]),
        "layer_num_heads": np.array(layer_num_heads, dtype=np.int32),
        "layer_head_dims": np.array(layer_head_dims, dtype=np.int32),
        "num_samples": np.array([tot]),
    }
    for li in range(num_layers):
        for hi in range(layer_num_heads[li]):
            vals = per_head_token_deltas[li][hi]
            if vals:
                save_dict[f"delta_L{li}_H{hi}"] = np.array(vals, dtype=np.float32)

    np.savez_compressed(npz_path, **save_dict)

    ranked_heads = _rank_valid_heads(ablation_result, layer_num_heads)
    topk = ranked_heads[: min(20, len(ranked_heads))]

    print("\nTop-20 most important heads:")
    print(f"{'Rank':>4} | {'Layer':>5} | {'Head':>4} | {'Mean Delta Loss':>14}")
    print("-" * 45)

    summary_lines = [f"\n--- RESULTS FOR {model_short} ON {dataset_name}{title_suffix} ---"]
    summary_lines.append("Top-20 most important heads:")
    summary_lines.append("Rank | Layer | Head |  Mean Delta Loss")
    summary_lines.append("-" * 45)

    for rank, (li, hi, val) in enumerate(topk):
        line = f"{rank + 1:4d} | {li:5d} | {hi:4d} | {val:14.4f}"
        print(line)
        summary_lines.append(line)

    summary_file = os.path.join(run_dir, "summary.txt")
    with open(summary_file, "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    return run_dir


def _save_phase2_outputs(run_dir, model_short, dataset_name, top_k, sorted_heads, knockout_avg_losses, knockout_per_token_all, title_suffix="", x_label="Per-token CE loss"):
    summary_lines = [f"\nProgressive top-{top_k} knockout{title_suffix}:"]
    for k in range(top_k + 1):
        if k == 0:
            line = f"k={k:3d} | baseline | avg_loss={knockout_avg_losses[k]:.4f}"
        else:
            li, hi, imp = sorted_heads[k - 1]
            line = f"k={k:3d} | +L{li}H{hi} (imp={imp:.4f}) | avg_loss={knockout_avg_losses[k]:.4f}"
        summary_lines.append(line)

    summary_file = os.path.join(run_dir, "summary.txt")
    with open(summary_file, "a") as f:
        f.write("\n".join(summary_lines) + "\n" + "-" * 50 + "\n")

    ko_path = os.path.join(run_dir, "knockout.npz")
    np.savez_compressed(ko_path, avg_losses=np.array(knockout_avg_losses))
    print(f"[save] {ko_path}")


# ===================== Regular/No-CoT =====================

def run_phase1_regular(model, tokenizer, data, num_layers, layer_num_heads, layer_head_dims, layers, model_short, dataset_name, out_dir):
    global BATCH_SIZE

    max_heads = max(layer_num_heads)
    total_heads = sum(layer_num_heads)
    print(f"\n[phase1] Per-head ablation: {num_layers} layers, {total_heads} total heads")

    ablation_result = torch.zeros((num_layers, max_heads))
    per_head_token_deltas = [[[] for _ in range(max_heads)] for _ in range(num_layers)]
    global_top_heads = None
    valid_heads = list(_iter_valid_heads(layer_num_heads))
    tot = 0

    for sample in tqdm(data, desc=f"Phase1 {model_short}/{dataset_name}"):
        text = build_cot_prompt(sample["question"], sample["answer"])
        answer = sample["answer"]

        tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
        answer_tokens = tokenizer(answer, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)

        ans_len = answer_tokens.shape[1]
        if ans_len == 0 or tokens.shape[1] <= ans_len:
            continue
        tot += 1

        with torch.no_grad():
            original_logits = model(tokens).logits
        base_ans_logits = original_logits[:, -ans_len - 1 : -1, :]
        base_per_token = compute_per_token_loss(base_ans_logits, answer_tokens)
        base_mean = base_per_token.mean()

        if base_mean < 1e-8:
            continue

        if tot <= 6:
            target_heads = valid_heads
        elif tot == 7:
            print("\n[phase1] 6 samples reached. Isolating Top heads ...")
            top_cap = min(TOP_K_HEADS, len(valid_heads))
            global_top_heads = [
                (li, hi)
                for li, hi, _ in _rank_valid_heads(ablation_result, layer_num_heads)[:top_cap]
            ]
            target_heads = global_top_heads
        else:
            target_heads = global_top_heads

        layer_to_heads = defaultdict(list)
        for l, h in target_heads:
            layer_to_heads[l].append(h)

        contribution = torch.zeros((num_layers, max_heads))
        batched_tokens = tokens.expand(BATCH_SIZE, -1)
        batched_answer_tokens = answer_tokens.expand(BATCH_SIZE, -1)

        for layer_idx, head_list in layer_to_heads.items():
            head_dim = layer_head_dims[layer_idx]
            o_proj = layers[layer_idx].self_attn.o_proj
            chunk_start_h = 0
            while chunk_start_h < len(head_list):
                chunk_end_h = min(chunk_start_h + BATCH_SIZE, len(head_list))
                current_batch_size = chunk_end_h - chunk_start_h
                head_indices = head_list[chunk_start_h:chunk_end_h]

                run_tokens = batched_tokens[:current_batch_size]
                run_answer_tokens = batched_answer_tokens[:current_batch_size]

                hook = o_proj.register_forward_pre_hook(make_batched_ablate_pre_hook(head_indices, head_dim))
                try:
                    with torch.no_grad():
                        ablated_logits = model(run_tokens).logits
                    abl_ans_logits = ablated_logits[:, -ans_len - 1 : -1, :]

                    abl_logits_flat = abl_ans_logits.float().reshape(-1, abl_ans_logits.size(-1))
                    labels_flat = run_answer_tokens.reshape(-1)
                    abl_per_token_flat = F.cross_entropy(abl_logits_flat, labels_flat, reduction="none").cpu().numpy()
                    abl_per_token = abl_per_token_flat.reshape(current_batch_size, ans_len)

                    for b, head_idx in enumerate(head_indices):
                        delta_per_token = abl_per_token[b] - base_per_token
                        per_head_token_deltas[layer_idx][head_idx].extend(delta_per_token.tolist())
                        contribution[layer_idx, head_idx] = float(delta_per_token.mean())

                    hook.remove()
                    chunk_start_h = chunk_end_h

                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    hook.remove()
                    if "out of memory" in str(e).lower() and BATCH_SIZE > 1:
                        BATCH_SIZE = max(1, BATCH_SIZE // 2)
                        print(f"\n[OOM] Reducing GLOBAL batch size to {BATCH_SIZE} and retrying layer {layer_idx} chunk {chunk_start_h}...")
                        batched_tokens = tokens.expand(BATCH_SIZE, -1)
                        batched_answer_tokens = answer_tokens.expand(BATCH_SIZE, -1)
                        import gc; gc.collect()
                        torch.cuda.empty_cache()
                    else:
                        print(f"\n[ERROR] Failed at layer {layer_idx}, chunk {chunk_start_h}-{chunk_end_h}: {e}")
                        raise

        ablation_result += contribution
        torch.cuda.empty_cache()

    ablation_result /= max(tot, 1)
    print(f"[phase1] Done. {tot} valid samples.")
    remove_all_hooks(model)

    run_dir = _save_phase1_outputs(
        out_dir,
        model_short,
        dataset_name,
        num_layers,
        layer_num_heads,
        layer_head_dims,
        tot,
        ablation_result,
        per_head_token_deltas,
        title_suffix="",
        x_label="Per-token delta loss",
    )
    return ablation_result, run_dir


def run_phase2_regular(model, tokenizer, data, num_layers, layer_num_heads, layer_head_dims, layers, ablation_result, top_k, model_short, dataset_name, run_dir):
    print(f"\n[phase2] Progressive top-{top_k} knockout")

    sorted_heads = _rank_valid_heads(ablation_result, layer_num_heads)
    top_k = min(top_k, len(sorted_heads))

    knockout_avg_losses = []
    knockout_per_token_all = []

    for k in tqdm(range(0, top_k + 1), desc=f"Phase2 {model_short}/{dataset_name}"):
        current_heads = sorted_heads[:k]
        layer_to_heads = {}
        for li, hi, _ in current_heads:
            layer_to_heads.setdefault(li, []).append(hi)

        hooks = []
        for li, hlist in layer_to_heads.items():
            o_proj = layers[li].self_attn.o_proj
            hooks.append(o_proj.register_forward_pre_hook(make_ablate_heads_pre_hook(hlist, layer_head_dims[li])))

        total_loss = 0.0
        n_valid = 0
        all_per_token = []

        for sample in data:
            text = build_cot_prompt(sample["question"], sample["answer"])
            answer = sample["answer"]

            tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
            answer_tokens = tokenizer(answer, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)

            ans_len = answer_tokens.shape[1]
            if ans_len == 0 or tokens.shape[1] <= ans_len:
                continue

            with torch.no_grad():
                logits = model(tokens).logits
            ans_logits = logits[:, -ans_len - 1 : -1, :]
            per_token = compute_per_token_loss(ans_logits, answer_tokens)
            all_per_token.extend(per_token.tolist())
            total_loss += per_token.mean()
            n_valid += 1

        avg_loss = total_loss / max(n_valid, 1)
        knockout_avg_losses.append(avg_loss)
        knockout_per_token_all.append(all_per_token)

        if k > 0:
            li, hi, imp = sorted_heads[k - 1]
            print(f"  k={k:3d} | +L{li}H{hi} (imp={imp:.4f}) | avg_loss={avg_loss:.4f}")
        else:
            print(f"  k=  0 | baseline | avg_loss={avg_loss:.4f}")

        for h in hooks:
            h.remove()

    remove_all_hooks(model)
    print("[phase2] Done.")

    _save_phase2_outputs(
        run_dir,
        model_short,
        dataset_name,
        top_k,
        sorted_heads,
        knockout_avg_losses,
        knockout_per_token_all,
        title_suffix="",
        x_label="Per-token CE loss",
    )


# ===================== Chain+Code =====================

def run_phase1_chain_code(model, tokenizer, data, num_layers, layer_num_heads, layer_head_dims, layers, model_short, dataset_name, out_dir):
    global BATCH_SIZE

    max_heads = max(layer_num_heads)
    total_heads = sum(layer_num_heads)
    print(f"\n[phase1] Per-head ablation: {num_layers} layers, {total_heads} total heads")
    print("[phase1] Metrics calculated for CODE ONLY")

    ablation_result = torch.zeros((num_layers, max_heads))
    per_head_token_deltas = [[[] for _ in range(max_heads)] for _ in range(num_layers)]
    global_top_heads = None
    valid_heads = list(_iter_valid_heads(layer_num_heads))
    tot = 0

    for sample in tqdm(data, desc=f"Phase1 {model_short}/{dataset_name}"):
        full_ids, code_start_idx, code_len = build_chain_code_tokens(sample["question"], sample["chain"], sample["code"], tokenizer, max_seq_len=MAX_SEQ_LEN)
        tokens = torch.tensor([full_ids], device=model.device)

        if code_start_idx <= 0 or code_start_idx >= tokens.shape[1]:
            continue
        code_tokens = tokens[:, code_start_idx : code_start_idx + code_len]
        code_len = code_tokens.shape[1]
        if code_len == 0:
            continue
        tot += 1

        with torch.no_grad():
            original_logits = model(tokens).logits

        code_logits = original_logits[:, code_start_idx - 1 : code_start_idx - 1 + code_len, :]
        if code_logits.shape[1] != code_len:
            continue

        base_per_token = compute_per_token_loss(code_logits, code_tokens)
        base_mean = base_per_token.mean()
        if base_mean < 1e-8:
            continue

        if tot <= 6:
            target_heads = valid_heads
        elif tot == 7:
            top_cap = min(TOP_K_HEADS, len(valid_heads))
            print(f"\n[phase1] 6 samples reached. Isolating Top-{top_cap} heads ...")
            global_top_heads = [
                (li, hi)
                for li, hi, _ in _rank_valid_heads(ablation_result, layer_num_heads)[:top_cap]
            ]
            target_heads = global_top_heads
        else:
            target_heads = global_top_heads

        layer_to_heads = defaultdict(list)
        for l, h in target_heads:
            layer_to_heads[l].append(h)

        contribution = torch.zeros((num_layers, max_heads))
        batched_tokens = tokens.expand(BATCH_SIZE, -1)
        batched_code_tokens = code_tokens.expand(BATCH_SIZE, -1)

        for layer_idx, head_list in layer_to_heads.items():
            head_dim = layer_head_dims[layer_idx]
            o_proj = layers[layer_idx].self_attn.o_proj
            chunk_start = 0

            while chunk_start < len(head_list):
                chunk_end = min(chunk_start + BATCH_SIZE, len(head_list))
                current_batch_size = chunk_end - chunk_start
                head_indices = head_list[chunk_start:chunk_end]

                run_tokens = batched_tokens[:current_batch_size]
                run_code_tokens = batched_code_tokens[:current_batch_size]

                hook = o_proj.register_forward_pre_hook(make_batched_ablate_pre_hook(head_indices, head_dim))

                try:
                    with torch.no_grad():
                        ablated_logits = model(run_tokens).logits

                    abl_code_logits = ablated_logits[:, code_start_idx - 1 : code_start_idx - 1 + code_len, :]
                    abl_logits_flat = abl_code_logits.float().reshape(-1, abl_code_logits.size(-1))
                    labels_flat = run_code_tokens.reshape(-1)
                    abl_per_token_flat = F.cross_entropy(abl_logits_flat, labels_flat, reduction="none").cpu().numpy()
                    abl_per_token = abl_per_token_flat.reshape(current_batch_size, code_len)

                    for b, head_idx in enumerate(head_indices):
                        delta_per_token = abl_per_token[b] - base_per_token
                        per_head_token_deltas[layer_idx][head_idx].extend(delta_per_token.tolist())
                        contribution[layer_idx, head_idx] = float(delta_per_token.mean())

                    hook.remove()
                    chunk_start = chunk_end

                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    hook.remove()
                    if "out of memory" in str(e).lower() and BATCH_SIZE > 1:
                        BATCH_SIZE = max(1, BATCH_SIZE // 2)
                        print(f"\n[OOM] Reducing GLOBAL batch size to {BATCH_SIZE} and retrying...")
                        batched_tokens = tokens.expand(BATCH_SIZE, -1)
                        batched_code_tokens = code_tokens.expand(BATCH_SIZE, -1)
                        torch.cuda.empty_cache()
                    else:
                        print(f"\n[ERROR] Failed at layer {layer_idx}, chunk {chunk_start}-{chunk_end}: {e}")
                        raise

        ablation_result += contribution
        torch.cuda.empty_cache()

    ablation_result /= max(tot, 1)
    print(f"[phase1] Done. {tot} valid samples.")
    remove_all_hooks(model)

    run_dir = _save_phase1_outputs(
        out_dir,
        model_short,
        dataset_name,
        num_layers,
        layer_num_heads,
        layer_head_dims,
        tot,
        ablation_result,
        per_head_token_deltas,
        title_suffix=" (CODE ONLY)",
        x_label="Per-token delta loss (CODE)",
    )
    return ablation_result, run_dir


def run_phase2_chain_code(model, tokenizer, data, num_layers, layer_num_heads, layer_head_dims, layers, ablation_result, top_k, model_short, dataset_name, run_dir):
    print(f"\n[phase2] Progressive top-{top_k} knockout")
    print("[phase2] Metrics calculated for CODE ONLY")

    sorted_heads = _rank_valid_heads(ablation_result, layer_num_heads)
    top_k = min(top_k, len(sorted_heads))

    knockout_avg_losses = []
    knockout_per_token_all = []

    for k in tqdm(range(0, top_k + 1), desc=f"Phase2 {model_short}/{dataset_name}"):
        current_heads = sorted_heads[:k]
        layer_to_heads = {}
        for li, hi, _ in current_heads:
            layer_to_heads.setdefault(li, []).append(hi)

        hooks = []
        for li, hlist in layer_to_heads.items():
            o_proj = layers[li].self_attn.o_proj
            hooks.append(o_proj.register_forward_pre_hook(make_ablate_heads_pre_hook(hlist, layer_head_dims[li])))

        total_loss = 0.0
        n_valid = 0
        all_per_token = []

        for sample in data:
            full_ids, code_start_idx, code_len = build_chain_code_tokens(sample["question"], sample["chain"], sample["code"], tokenizer, max_seq_len=MAX_SEQ_LEN)
            tokens = torch.tensor([full_ids], device=model.device)

            if code_start_idx <= 0 or code_start_idx >= tokens.shape[1]:
                continue
            code_tokens = tokens[:, code_start_idx : code_start_idx + code_len]
            code_len = code_tokens.shape[1]
            if code_len == 0:
                continue

            with torch.no_grad():
                logits = model(tokens).logits

            code_logits = logits[:, code_start_idx - 1 : code_start_idx - 1 + code_len, :]
            if code_logits.shape[1] != code_len:
                continue

            per_token = compute_per_token_loss(code_logits, code_tokens)
            all_per_token.extend(per_token.tolist())
            total_loss += per_token.mean()
            n_valid += 1

        avg_loss = total_loss / max(n_valid, 1)
        knockout_avg_losses.append(avg_loss)
        knockout_per_token_all.append(all_per_token)

        if k > 0:
            li, hi, imp = sorted_heads[k - 1]
            print(f"  k={k:3d} | +L{li}H{hi} (imp={imp:.4f}) | avg_loss={avg_loss:.4f}")
        else:
            print(f"  k=  0 | baseline | avg_loss={avg_loss:.4f}")

        for h in hooks:
            h.remove()

    remove_all_hooks(model)
    print("[phase2] Done.")

    _save_phase2_outputs(
        run_dir,
        model_short,
        dataset_name,
        top_k,
        sorted_heads,
        knockout_avg_losses,
        knockout_per_token_all,
        title_suffix=" (CODE ONLY)",
        x_label="Per-token CE loss (CODE)",
    )


# ===================== Main experiment runner =====================

def run_head_ablation(model_name, dataset_name, num_samples, top_k, out_dir, mode="regular", rating_range=None, cf_data_path="data/datasets/cf_dataset.parquet", hml_model_dir=None, seed=42):
    model_short = model_name.split("/")[-1]
    print(f"\n{'#'*70}")
    print(f"# Model: {model_name}  |  Dataset: {dataset_name}  |  Samples: {num_samples}")
    if rating_range:
        print(f"# Rating range: {rating_range}")
    if hml_model_dir:
        print(f"# HML model dir: {hml_model_dir}  |  Seed: {seed}")
    print(f"# Mode: {mode}")
    print(f"{'#'*70}")

    model, tokenizer = load_model_and_tokenizer(model_name)
    torch.set_grad_enabled(False)

    layers = get_transformer_layers(model)
    num_layers = get_num_hidden_layers(model, layers=layers)
    layer_head_dims, layer_num_heads, max_heads = get_layer_head_layout(model, layers=layers)

    if hml_model_dir:
        if mode == "chain_code":
            data = load_hml_dataset_chain_code(
                dataset_name,
                num_samples,
                hml_model_dir=hml_model_dir,
                seed=seed,
                tokenizer=tokenizer,
                max_seq_len=MAX_SEQ_LEN,
            )
        else:
            data = load_hml_dataset(
                dataset_name,
                num_samples,
                hml_model_dir=hml_model_dir,
                seed=seed,
                no_cot=(mode == "no_cot"),
                tokenizer=tokenizer,
                max_seq_len=MAX_SEQ_LEN,
            )
    elif mode == "chain_code":
        data = load_coding_dataset_chain_code(
            dataset_name,
            num_samples,
            rating_range=rating_range,
            cf_data_path=cf_data_path,
            tokenizer=tokenizer,
            max_seq_len=MAX_SEQ_LEN,
        )
    else:
        data = load_coding_dataset(
            dataset_name,
            num_samples,
            rating_range=rating_range,
            cf_data_path=cf_data_path,
            no_cot=(mode == "no_cot"),
            tokenizer=tokenizer,
            max_seq_len=MAX_SEQ_LEN,
        )

    unique_dims = sorted(set(layer_head_dims))
    if len(unique_dims) == 1:
        print(f"[model] {model_short}: {num_layers}L x <= {max_heads}H, head_dim={unique_dims[0]}")
    else:
        print(f"[model] {model_short}: {num_layers}L x <= {max_heads}H, mixed head_dim={unique_dims}")
    print(f"[model] Total heads: {sum(layer_num_heads)}")

    if mode == "chain_code":
        ablation_result, run_dir = run_phase1_chain_code(
            model,
            tokenizer,
            data,
            num_layers,
            layer_num_heads,
            layer_head_dims,
            layers,
            model_short,
            dataset_name,
            out_dir,
        )
        run_phase2_chain_code(
            model,
            tokenizer,
            data,
            num_layers,
            layer_num_heads,
            layer_head_dims,
            layers,
            ablation_result,
            top_k,
            model_short,
            dataset_name,
            run_dir,
        )
    else:
        ablation_result, run_dir = run_phase1_regular(
            model,
            tokenizer,
            data,
            num_layers,
            layer_num_heads,
            layer_head_dims,
            layers,
            model_short,
            dataset_name,
            out_dir,
        )
        run_phase2_regular(
            model,
            tokenizer,
            data,
            num_layers,
            layer_num_heads,
            layer_head_dims,
            layers,
            ablation_result,
            top_k,
            model_short,
            dataset_name,
            run_dir,
        )

    free_model(model, tokenizer)
    print(f"[done] Freed memory for {model_name}")


# ===================== CLI =====================

def _resolve_default_out_dir(mode: str) -> str:
    if mode == "chain_code":
        return "head_ablation_results_chain_code"
    if mode == "no_cot":
        return "head_ablation_results_no_cot"
    return "head_ablation_results"


def _resolve_default_cf_path(mode: str) -> str:
    if mode == "chain_code":
        return os.path.join("data", "datasets", "cf_dataset_chain_code.parquet")
    return os.path.join("data", "datasets", "cf_dataset.parquet")


def main():
    from coding_utils import hml_dataset_to_outdir

    parser = argparse.ArgumentParser(description="Head Ablation analysis on coding datasets")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help='HuggingFace model name, or "all"')
    parser.add_argument(
        "--dataset",
        type=str,
        default="livecodebench",
        help='Dataset name (livecodebench/mbpp/humaneval/codeforces), or "all"',
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="regular",
        choices=["regular", "no_cot", "chain_code"],
        help="Analysis mode",
    )
    parser.add_argument(
        "--rating_range",
        type=str,
        default=None,
        choices=RATING_RANGES,
        help="Codeforces rating range filter (required when --dataset codeforces)",
    )
    parser.add_argument("--cf_data_path", type=str, default=None, help="Path to Codeforces parquet file")
    parser.add_argument("--hml_model_dir", type=str, default=None, help="Path to HML model results dir (e.g. data/benchmarks/Qwen--Qwen3-8B)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for HML sampling")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    models = ALL_MODELS if args.model == "all" else [args.model]
    is_hml = args.hml_model_dir is not None

    if is_hml:
        datasets = HML_DATASETS if args.dataset == "all" else [args.dataset]
    elif args.mode == "chain_code":
        if args.dataset == "all":
            datasets = ["codeforces"]
        elif args.dataset != "codeforces":
            parser.error("--mode chain_code currently supports only --dataset codeforces (use --hml_model_dir for HML datasets)")
        else:
            datasets = [args.dataset]
    else:
        non_cf_datasets = [d for d in ALL_DATASETS if d != "codeforces"]
        datasets = non_cf_datasets if args.dataset == "all" else [args.dataset]

    out_dir_base = args.out_dir or _resolve_default_out_dir(args.mode)
    cf_data_path = args.cf_data_path or _resolve_default_cf_path(args.mode)

    for ds in datasets:
        if ds == "codeforces" and args.rating_range is None:
            parser.error("--rating_range is required when --dataset codeforces")

        if is_hml:
            # In HML mode, always nest outputs under results/hml/<dataset>/...
            # even when --out_dir is explicitly provided as an experiment name.
            ds_out_dir = hml_dataset_to_outdir(ds, out_dir_base)
        elif ds == "codeforces":
            ds_out_dir = rating_range_to_outdir(args.rating_range, os.path.join("results", out_dir_base))
        else:
            ds_out_dir = os.path.join("results", out_dir_base)

        for mdl in models:
            try:
                run_head_ablation(
                    model_name=mdl,
                    dataset_name=ds,
                    num_samples=args.num_samples,
                    top_k=args.top_k,
                    out_dir=ds_out_dir,
                    mode=args.mode,
                    rating_range=args.rating_range if ds == "codeforces" else None,
                    cf_data_path=cf_data_path,
                    hml_model_dir=args.hml_model_dir,
                    seed=args.seed,
                )
            except Exception as e:
                print(f"\n[ERROR] Failed for model={mdl}, dataset={ds}: {e}")
                import traceback
                traceback.print_exc()
                continue

    print("\n=== All done! ===")


if __name__ == "__main__":
    main()
