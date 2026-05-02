# HML LLM Interpretability on Coding Benchmarks

This repository contains tools and experiments for analyzing and pruning attention heads in large language models (LLMs), evaluated on coding benchmarks. The primary focus is **HML: HumanEval, MBPP, and LiveCodeBench**. We also explore differences in attention circuits between HuggingFace base models and their distilled variants.

Code repository: https://github.com/SayandeepB9/BTP

Evaluated benchmark pools:
- HumanEval: 164 tasks
- MBPP: 500 tasks
- LiveCodeBench v6: 175 tasks

## Core Experiments (HML)

### 1. Entropy & Loss Analysis (`src/entropy_lens.py`)
Analyzes per-token entropy and cross-entropy loss across model layers. Supports multiple reasoning modes:
- **`regular`**: CoT mode where loss is computed on reasoning + code tokens.
- **`chain_code`**: Context includes CoT, but metrics are isolated to the code block.

### 2. Head Ablation (`src/head_ablation.py`)
Quantifies the importance of individual attention heads by zeroing them out one-by-one and measuring the resulting delta-loss.
- Supports single-head ablation matrices.
- Supports "knockout" experiments (iteratively removing the most important heads).

### 3. Iterative Taylor Pruning (`src/iterative_taylor_pruning.py`)
Prunes attention heads iteratively based on a first-order Taylor approximation of their contribution to the loss.
- Configurable `num_iters` and `prune_per_iter`.
- Results are stored as `.json` and `.npz` history files for visualization.

### 4. Pruned Model Inference (`src/run_pruned.py`)
Evaluating model performance under specific head pruning regimes by running models after masking pruned attention heads.

---

## Directory Layout

```text
├── src/                         # Python Source Code
│   ├── coding_utils.py          # Unified data loaders and model utilities
│   ├── entropy_lens.py          # Entropy & loss tracking
│   ├── head_ablation.py         # Importance & knockout experiments
│   ├── iterative_taylor_pruning.py # Taylor-based iterative pruning
│   ├── evaluate.py              # Correctness evaluation for benchmarks
│   ├── run_inference.py         # Model generation inference script (vLLM supported)
│   └── run_pruned.py            # Run models after masking pruned attention heads
│
├── scripts/run/                 # Shell runners
│   ├── run_hml.sh               # Run ablation/entropy sweep on HML
│   ├── run_inference.sh         # Run inference via vLLM
│   └── run_evaluate.sh          # Evaluate generated code correctness
│
├── notebooks/hml/               # Analysis & Visualization Notebooks
│   ├── entropy_analysis.ipynb
│   ├── head_ablation_analysis.ipynb
│   ├── visualize_optimization.ipynb
│   ├── pruned_run_analysis.ipynb
│   └── compare_cot_vs_chain_code.ipynb
```

## Workflows

### 1) Run Inference & Evaluation
First, generate solutions using a vLLM server and evaluate them to filter for correct solutions (`verdict=1`).

Typical inference serving uses `--max-model-len 32768`.
```bash
bash scripts/run/run_inference.sh
bash scripts/run/run_evaluate.sh
```

### 2) Core Experiments (Ablation / Entropy)
```bash
bash scripts/run/run_hml.sh
```
Direct python invocation examples:
```bash
python src/entropy_lens.py --model Qwen/Qwen3-8B --dataset mbpp --mode chain_code
python src/head_ablation.py --model Qwen/Qwen3-8B --dataset mbpp --mode chain_code
```

Notebooks in `notebooks/` are configured to read from experimental output paths. Uses symmetric `coolwarm` heatmaps for importance analysis.

1. Run experiments from `scripts/run/` or `src/`.
2. Open notebook and execute cells.
3. Inspect plots/tables for model and rating trends.

## Citation

If you use this codebase, please cite:

- Michel, Levy, and Neubig (2019), *Are Sixteen Heads Really Better than One?* (NeurIPS), https://arxiv.org/abs/1905.10650
