#!/usr/bin/env bash
# Run iterative Taylor pruning on HML datasets (HumanEval, MBPP, LiveCodeBench).
# Each dataset runs both regular and chain_code modes for each model.
#
# Usage:
#   bash scripts/run/run_taylor_hml.sh

set -euo pipefail

# Allow overriding which models to run.
# Single-model override: MODEL=Qwen/Qwen3-4B bash scripts/run/run_taylor_hml.sh
# Multi-model via array (edit this list or use MODELS env-var array):
if [[ -n "${MODEL:-}" ]]; then
  MODELS=("$MODEL")
else
  MODELS=("Qwen/Qwen3-4B" "meta-llama/Llama-3.2-3B-Instruct")
fi
SEED=42

NUM_SAMPLES="${NUM_SAMPLES:-100}"
NUM_ITERS="${NUM_ITERS:-100}"
PRUNE_PER_ITER="${PRUNE_PER_ITER:-5}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

DATASETS=("humaneval" "mbpp" "livecodebench")

for MODEL in "${MODELS[@]}"; do
  MODEL_SHORT="${MODEL##*/}"
  MODEL_DIR_NAME=$(echo "$MODEL" | sed 's|/|--|g')
  HML_MODEL_DIR="data/benchmarks/${MODEL_DIR_NAME}"

  echo "================================================================"
  echo " Iterative Taylor Pruning -- $(date)"
  echo " Model: ${MODEL}"
  echo " HML dir: ${HML_MODEL_DIR}"
  echo " Seed: ${SEED}"
  echo " Datasets: ${DATASETS[*]}"
  echo " Modes: regular, chain_code"
  echo " Samples: ${NUM_SAMPLES}, Iters: ${NUM_ITERS}, Prune/iter: ${PRUNE_PER_ITER}"
  echo "================================================================"

  if [[ ! -d "$HML_MODEL_DIR" ]]; then
    echo "ERROR: HML model dir not found: ${HML_MODEL_DIR}"
    echo "Run inference first with scripts/run/run_inference.sh"
    exit 1
  fi

  for DS in "${DATASETS[@]}"; do
    echo ""
    echo "----------------------------------------------------------------"
    echo " TAYLOR PRUNING | model=${MODEL_SHORT} | dataset=${DS} | modes=regular+chain_code"
    echo "----------------------------------------------------------------"
    TAYLOR_JSON="results/hml/${DS}/iterative_taylor/${MODEL_SHORT}/iterative_taylor_results.json"
    TAYLOR_CC_JSON="results/hml/${DS}/iterative_taylor_chain_code/${MODEL_SHORT}/iterative_taylor_results.json"
    if [[ -f "$TAYLOR_JSON" && -f "$TAYLOR_CC_JSON" ]]; then
      echo "  [skip] Taylor results already exist for both modes: ${DS}"
    else
      # Python handles both modes (regular, chain_code) in a single run
      "$PYTHON_BIN" src/iterative_taylor_pruning.py \
        --model "${MODEL}" \
        --dataset "${DS}" \
        --hml_model_dir "${HML_MODEL_DIR}" \
        --num_samples "${NUM_SAMPLES}" \
        --num_iters "${NUM_ITERS}" \
        --prune_per_iter "${PRUNE_PER_ITER}" \
        --seed "${SEED}"
    fi
  done

  echo ""
  echo "================================================================"
  echo " Taylor pruning complete for ${MODEL_SHORT} -- $(date)"
  echo " Results in: results/hml/*/iterative_taylor*/${MODEL_SHORT}/"
  echo "================================================================"
done
