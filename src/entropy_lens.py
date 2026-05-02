"""
Entropy Lens Analysis on Coding Datasets
=======================================
Supports three modes:
  - regular: standard CoT answer-token metrics
  - no_cot: answer-token metrics on no-cot dataset variant
  - chain_code: code-token-only metrics on chain+code dataset
"""

import argparse
import gc
import os

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
    free_model,
    get_model_final_norm,
    get_num_hidden_layers,
    load_coding_dataset,
    load_coding_dataset_chain_code,
    load_hml_dataset,
    load_hml_dataset_chain_code,
    load_model_and_tokenizer,
    rating_range_to_outdir,
)


# ===================== Performance Tuning =====================

BATCH_SIZE = 32
MAX_SEQ_LEN = 3072


# ===================== Core per-token functions =====================

def compute_batched_layerwise_metrics(hidden_states, model, input_ids, final_norm=None):
    """Project each layer's hidden state through lm_head; compute entropy and loss."""
    entropies = []
    losses = []

    label_ids = input_ids[:, 1:].to(model.lm_head.weight.device).contiguous().view(-1)

    with torch.no_grad():
        for i, h in enumerate(hidden_states[1:]):
            if i < len(hidden_states[1:]) - 1:
                h_native = h.to(model.lm_head.weight.device)
                if final_norm is not None:
                    h_norm = final_norm(h_native)
                else:
                    h_norm = h_native
            else:
                h_norm = h.to(model.lm_head.weight.device)

            logits = model.lm_head(h_norm).to(torch.float32)
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            entropies.append(entropy.cpu().numpy())

            logits_shifted = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            loss_flat = F.cross_entropy(logits_shifted, label_ids, reduction="none")
            loss = loss_flat.view(logits.size(0), logits.size(1) - 1)
            losses.append(loss.cpu().numpy())

    return np.stack(entropies, axis=0), np.stack(losses, axis=0)


# ===================== Mode-specific runners =====================

def _save_and_plot(
    run_dir,
    model_short,
    dataset_name,
    num_layers,
    processed,
    all_token_entropies,
    all_token_losses,
    summary_header,
    title_suffix="",
):
    avg_entropy = np.array([np.mean(v) for v in all_token_entropies])
    std_entropy = np.array([np.std(v) for v in all_token_entropies])
    avg_loss = np.array([np.mean(v) for v in all_token_losses])
    std_loss = np.array([np.std(v) for v in all_token_losses])

    npz_path = os.path.join(run_dir, "per_token.npz")
    save_dict = {
        "avg_entropy": avg_entropy,
        "std_entropy": std_entropy,
        "avg_loss": avg_loss,
        "std_loss": std_loss,
        "num_layers": np.array([num_layers]),
        "num_samples": np.array([processed]),
        "total_tokens": np.array([len(all_token_entropies[0])]),
    }
    for i in range(num_layers):
        save_dict[f"entropy_layer_{i}"] = np.array(all_token_entropies[i], dtype=np.float32)
        save_dict[f"loss_layer_{i}"] = np.array(all_token_losses[i], dtype=np.float32)
    np.savez_compressed(npz_path, **save_dict)

    summary_lines = [summary_header]
    for i in range(num_layers):
        summary_lines.append(f"Layer {i:4d} | Loss: {avg_loss[i]:.4f} | Entropy: {avg_entropy[i]:.4f}")
    summary_lines.append("-" * 50 + "\n")
    summary_text = "\n".join(summary_lines)
    print(summary_text)

    summary_file = os.path.join(run_dir, "summary.txt")
    with open(summary_file, "w") as f:
        f.write(summary_text)

    return {
        "avg_entropy": avg_entropy,
        "std_entropy": std_entropy,
        "avg_loss": avg_loss,
        "std_loss": std_loss,
        "num_layers": num_layers,
    }


def run_entropy_lens_regular(
    model_name,
    dataset_name,
    num_samples,
    out_dir,
    rating_range=None,
    cf_data_path="data/datasets/cf_dataset.parquet",
    no_cot=False,
    hml_model_dir=None,
    seed=42,
):
    model_short = model_name.split("/")[-1]
    print(f"\n{'='*70}")
    print(f"  Model: {model_name}  |  Dataset: {dataset_name}  |  Samples: {num_samples}")
    if rating_range:
        print(f"  Rating range: {rating_range}")
    if hml_model_dir:
        print(f"  HML model dir: {hml_model_dir}  |  Seed: {seed}")
    print(f"  Mode: {'no_cot' if no_cot else 'regular'}")
    print(f"{'='*70}")

    model, tokenizer = load_model_and_tokenizer(model_name)
    tokenizer.padding_side = "left"

    num_layers = get_num_hidden_layers(model)
    hidden_size = getattr(getattr(model.config, "text_config", model.config), "hidden_size", getattr(model.config, "hidden_size", "?"))
    final_norm = get_model_final_norm(model)
    print(f"[model] {model_short}: {num_layers} layers, hidden={hidden_size}")

    if hml_model_dir:
        data = load_hml_dataset(
            dataset_name,
            num_samples,
            hml_model_dir=hml_model_dir,
            seed=seed,
            no_cot=no_cot,
            tokenizer=tokenizer,
            max_seq_len=MAX_SEQ_LEN,
        )
    else:
        data = load_coding_dataset(
            dataset_name,
            num_samples,
            rating_range=rating_range,
            cf_data_path=cf_data_path,
            no_cot=no_cot,
            tokenizer=tokenizer,
            max_seq_len=MAX_SEQ_LEN,
        )

    all_token_entropies = [[] for _ in range(num_layers)]
    all_token_losses = [[] for _ in range(num_layers)]

    global BATCH_SIZE
    processed = 0
    chunk_start = 0
    pbar = tqdm(total=len(data), desc=f"{model_short}/{dataset_name}")

    while chunk_start < len(data):
        chunk_end = min(chunk_start + BATCH_SIZE, len(data))
        chunk = data[chunk_start:chunk_end]

        try:
            texts = [build_cot_prompt(s["question"], s["answer"]) for s in chunk]
            answers = [s["answer"] for s in chunk]

            inputs = tokenizer(texts, return_tensors="pt", add_special_tokens=False, padding=True)
            labels = tokenizer(answers, return_tensors="pt", add_special_tokens=False, padding=True)

            seq_len = inputs["input_ids"].shape[1]
            ans_lens = [int(labels["attention_mask"][b].sum()) for b in range(len(chunk))]

            valid_b = [b for b in range(len(chunk)) if 0 < ans_lens[b] < seq_len]
            if not valid_b:
                chunk_start = chunk_end
                pbar.update(len(chunk))
                continue

            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states

            ent_matrix_batch, loss_matrix_batch = compute_batched_layerwise_metrics(
                hidden_states,
                model,
                inputs["input_ids"],
                final_norm=final_norm,
            )

            for b in valid_b:
                al = int(ans_lens[b])
                ent_sample = ent_matrix_batch[:, b, -al - 1 : -1]
                loss_sample = loss_matrix_batch[:, b, -al:]

                for layer_i in range(num_layers):
                    all_token_entropies[layer_i].extend(ent_sample[layer_i].tolist())
                    all_token_losses[layer_i].extend(loss_sample[layer_i].tolist())

            processed += len(valid_b)

            del outputs, hidden_states, ent_matrix_batch, loss_matrix_batch
            gc.collect()
            torch.cuda.empty_cache()

            chunk_start = chunk_end
            pbar.update(len(chunk))

        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "out of memory" in str(e).lower() and BATCH_SIZE > 1:
                BATCH_SIZE = max(1, BATCH_SIZE // 2)
                pbar.write(f"[OOM] Reducing GLOBAL batch size to {BATCH_SIZE} and retrying chunk {chunk_start}...")
                gc.collect()
                torch.cuda.empty_cache()
            else:
                pbar.close()
                raise

    pbar.close()

    run_dir = os.path.join(out_dir, model_short, dataset_name)
    os.makedirs(run_dir, exist_ok=True)

    title_suffix = " (NO COT)" if no_cot else ""
    summary_header = f"\n--- SUMMARY RESULTS FOR {model_short} ON {dataset_name}{title_suffix} ---"
    result = _save_and_plot(
        run_dir,
        model_short,
        dataset_name,
        num_layers,
        processed,
        all_token_entropies,
        all_token_losses,
        summary_header,
        title_suffix=title_suffix,
    )

    free_model(model, tokenizer)
    return result


def run_entropy_lens_chain_code(
    model_name,
    dataset_name,
    num_samples,
    out_dir,
    rating_range=None,
    cf_data_path="data/datasets/cf_dataset_chain_code.parquet",
    hml_model_dir=None,
    seed=42,
):
    global BATCH_SIZE

    model_short = model_name.split("/")[-1]
    print(f"\n{'='*70}")
    print(f"  Model: {model_name}  |  Dataset: {dataset_name}  |  Samples: {num_samples}")
    if rating_range:
        print(f"  Rating range: {rating_range}")
    if hml_model_dir:
        print(f"  HML model dir: {hml_model_dir}  |  Seed: {seed}")
    print("  Mode: chain_code (metrics on CODE ONLY)")
    print(f"{'='*70}")

    model, tokenizer = load_model_and_tokenizer(model_name)
    tokenizer.padding_side = "left"

    num_layers = get_num_hidden_layers(model)
    hidden_size = getattr(getattr(model.config, "text_config", model.config), "hidden_size", getattr(model.config, "hidden_size", "?"))
    final_norm = get_model_final_norm(model)
    print(f"[model] {model_short}: {num_layers} layers, hidden={hidden_size}")

    if hml_model_dir:
        data = load_hml_dataset_chain_code(
            dataset_name,
            num_samples,
            hml_model_dir=hml_model_dir,
            seed=seed,
            tokenizer=tokenizer,
            max_seq_len=MAX_SEQ_LEN,
        )
    else:
        data = load_coding_dataset_chain_code(
            dataset_name,
            num_samples,
            rating_range=rating_range,
            cf_data_path=cf_data_path,
            tokenizer=tokenizer,
            max_seq_len=MAX_SEQ_LEN,
        )

    all_token_entropies = [[] for _ in range(num_layers)]
    all_token_losses = [[] for _ in range(num_layers)]

    processed = 0
    chunk_start = 0
    pbar = tqdm(total=len(data), desc=f"{model_short}/{dataset_name}")

    while chunk_start < len(data):
        chunk_end = min(chunk_start + BATCH_SIZE, len(data))
        chunk = data[chunk_start:chunk_end]

        try:
            full_ids_batch = []
            code_start_indices = []
            code_lens = []
            for s in chunk:
                full_ids, code_start_idx, code_len = build_chain_code_tokens(
                    s["question"], s["chain"], s["code"], tokenizer, max_seq_len=MAX_SEQ_LEN
                )
                full_ids_batch.append(full_ids)
                code_start_indices.append(code_start_idx)
                code_lens.append(code_len)

            max_len = max(len(ids) for ids in full_ids_batch)
            pad_id = tokenizer.pad_token_id
            input_ids = []
            attention_mask = []
            for ids in full_ids_batch:
                pad_len = max_len - len(ids)
                input_ids.append(ids + [pad_id] * pad_len)
                attention_mask.append([1] * len(ids) + [0] * pad_len)

            inputs = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }

            valid_b = [b for b in range(len(chunk)) if code_lens[b] > 0]
            if not valid_b:
                chunk_start = chunk_end
                pbar.update(len(chunk))
                continue

            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states

            ent_matrix_batch, loss_matrix_batch = compute_batched_layerwise_metrics(
                hidden_states,
                model,
                inputs["input_ids"],
                final_norm=final_norm,
            )

            for b in valid_b:
                code_len = code_lens[b]
                code_start = code_start_indices[b]
                pred_start = code_start - 1
                pred_end = pred_start + code_len
                if pred_start < 0:
                    continue

                ent_sample = ent_matrix_batch[:, b, pred_start:pred_end]
                loss_sample = loss_matrix_batch[:, b, pred_start:pred_end]
                if ent_sample.shape[1] != code_len or loss_sample.shape[1] != code_len:
                    continue

                for layer_i in range(num_layers):
                    all_token_entropies[layer_i].extend(ent_sample[layer_i].tolist())
                    all_token_losses[layer_i].extend(loss_sample[layer_i].tolist())

            processed += len(valid_b)

            del outputs, hidden_states, ent_matrix_batch, loss_matrix_batch
            gc.collect()
            torch.cuda.empty_cache()

            chunk_start = chunk_end
            pbar.update(len(chunk))

        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "out of memory" in str(e).lower() and BATCH_SIZE > 1:
                BATCH_SIZE = max(1, BATCH_SIZE // 2)
                pbar.write(f"\n[OOM] Reducing GLOBAL batch size to {BATCH_SIZE} and retrying...")
                gc.collect()
                torch.cuda.empty_cache()
            else:
                pbar.write(f"\n[ERROR] Failed at chunk {chunk_start}-{chunk_end}: {e}")
                pbar.close()
                raise

    pbar.close()

    run_dir = os.path.join(out_dir, model_short, dataset_name)
    os.makedirs(run_dir, exist_ok=True)

    summary_header = f"\n--- SUMMARY RESULTS FOR {model_short} ON {dataset_name} (CHAIN_CODE) ---\nMetrics calculated for CODE ONLY"
    result = _save_and_plot(
        run_dir,
        model_short,
        dataset_name,
        num_layers,
        processed,
        all_token_entropies,
        all_token_losses,
        summary_header,
        title_suffix=" (CODE ONLY)",
    )

    free_model(model, tokenizer)
    return result


# ===================== CLI =====================

def _resolve_default_out_dir(mode: str) -> str:
    if mode == "chain_code":
        return "entropy_loss_results_chain_code"
    if mode == "no_cot":
        return "entropy_loss_results_no_cot"
    return "entropy_loss_results"


def _resolve_default_cf_path(mode: str) -> str:
    if mode == "chain_code":
        return os.path.join("data", "datasets", "cf_dataset_chain_code.parquet")
    return os.path.join("data", "datasets", "cf_dataset.parquet")


def main():
    from coding_utils import hml_dataset_to_outdir

    parser = argparse.ArgumentParser(description="Entropy Lens analysis on coding datasets")
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
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    models = ALL_MODELS if args.model == "all" else [args.model]
    is_hml = args.hml_model_dir is not None

    if is_hml:
        # HML mode: all 3 modes supported for all HML datasets
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

        comparison_results = {}
        for mdl in models:
            try:
                if args.mode == "chain_code":
                    result = run_entropy_lens_chain_code(
                        model_name=mdl,
                        dataset_name=ds,
                        num_samples=args.num_samples,
                        out_dir=ds_out_dir,
                        rating_range=args.rating_range if not is_hml else None,
                        cf_data_path=cf_data_path,
                        hml_model_dir=args.hml_model_dir,
                        seed=args.seed,
                    )
                else:
                    result = run_entropy_lens_regular(
                        model_name=mdl,
                        dataset_name=ds,
                        num_samples=args.num_samples,
                        out_dir=ds_out_dir,
                        rating_range=args.rating_range if ds == "codeforces" else None,
                        cf_data_path=cf_data_path,
                        no_cot=(args.mode == "no_cot"),
                        hml_model_dir=args.hml_model_dir,
                        seed=args.seed,
                    )
                comparison_results[mdl.split("/")[-1]] = result
            except Exception as e:
                print(f"\n[ERROR] Failed for model={mdl}, dataset={ds}: {e}")
                import traceback
                traceback.print_exc()
                continue

    print("\n=== All done! ===")


if __name__ == "__main__":
    main()
