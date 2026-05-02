#!/usr/bin/env bash
# Run head ablation + entropy lens on HumanEval, MBPP, and LiveCodeBench
# using model-generated chains from inference results.
#
# Runs 3 datasets x 2 modes (regular, chain_code) x 2 experiments per model.

set -euo pipefail

# Allow overriding which models to run.
# Single-model override: MODEL=Qwen/Qwen3-4B bash scripts/run/run_hml.sh
# Multi-model via array (edit this list or use MODELS env-var array):
if [[ -n "${MODEL:-}" ]]; then
  MODELS=("$MODEL")
else
  MODELS=("Qwen/Qwen3-4B" "meta-llama/Llama-3.2-3B-Instruct")
fi
SEED=42

NUM_SAMPLES_ENTROPY="${NUM_SAMPLES_ENTROPY:-100}"
NUM_SAMPLES_ABLATION="${NUM_SAMPLES_ABLATION:-100}"
TOP_K="${TOP_K:-20}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

DATASETS=("humaneval" "mbpp" "livecodebench")
MODES=("regular" "chain_code")

for MODEL in "${MODELS[@]}"; do
  MODEL_SHORT="${MODEL##*/}"
  MODEL_DIR_NAME=$(echo "$MODEL" | sed 's|/|--|g')
  HML_MODEL_DIR="data/benchmarks/${MODEL_DIR_NAME}"

  echo "================================================================"
  echo " HML sweep -- $(date)"
  echo " Model: ${MODEL}"
  echo " HML dir: ${HML_MODEL_DIR}"
  echo " Seed: ${SEED}"
  echo " Datasets: ${DATASETS[*]}"
  echo " Modes: ${MODES[*]}"
  echo "================================================================"

  if [[ ! -d "$HML_MODEL_DIR" ]]; then
    echo "ERROR: HML model dir not found: ${HML_MODEL_DIR}"
    echo "Run inference first with scripts/run/run_inference.sh"
    exit 1
  fi

  for MODE in "${MODES[@]}"; do
    echo ""
    echo "================================================================"
    echo " MODE: ${MODE}"
    echo "================================================================"

    if [[ "$MODE" == "chain_code" ]]; then
      ABLATION_VARIANT="head_ablation_results_chain_code"
      ENTROPY_VARIANT="entropy_loss_results_chain_code"
    else
      ABLATION_VARIANT="head_ablation_results"
      ENTROPY_VARIANT="entropy_loss_results"
    fi

    for DS in "${DATASETS[@]}"; do
      echo ""
      echo "----------------------------------------------------------------"
      echo " HEAD ABLATION | model=${MODEL_SHORT} | dataset=${DS} | mode=${MODE}"
      echo "----------------------------------------------------------------"
      ABLATION_NPZ="results/hml/${DS}/${ABLATION_VARIANT}/${MODEL_SHORT}/${DS}/ablation.npz"
      if [[ -f "$ABLATION_NPZ" ]]; then
        echo "  [skip] ablation.npz already exists: ${ABLATION_NPZ}"
      else
        "$PYTHON_BIN" src/head_ablation.py \
          --dataset "${DS}" \
          --model "${MODEL}" \
          --num_samples "${NUM_SAMPLES_ABLATION}" \
          --top_k "${TOP_K}" \
          --mode "${MODE}" \
          --hml_model_dir "${HML_MODEL_DIR}" \
          --seed "${SEED}"
      fi

      echo ""
      echo "----------------------------------------------------------------"
      echo " ENTROPY LENS  | model=${MODEL_SHORT} | dataset=${DS} | mode=${MODE}"
      echo "----------------------------------------------------------------"
      ENTROPY_NPZ="results/hml/${DS}/${ENTROPY_VARIANT}/${MODEL_SHORT}/${DS}/per_token.npz"
      if [[ -f "$ENTROPY_NPZ" ]]; then
        echo "  [skip] per_token.npz already exists: ${ENTROPY_NPZ}"
      else
        "$PYTHON_BIN" src/entropy_lens.py \
          --dataset "${DS}" \
          --model "${MODEL}" \
          --num_samples "${NUM_SAMPLES_ENTROPY}" \
          --mode "${MODE}" \
          --hml_model_dir "${HML_MODEL_DIR}" \
          --seed "${SEED}"
      fi
    done
  done

  echo ""
  echo "================================================================"
  echo " HML sweep complete for ${MODEL_SHORT} -- $(date)"
  echo " Results in: results/hml/"
  echo "================================================================"
done
