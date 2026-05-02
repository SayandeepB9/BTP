#!/usr/bin/env bash
# Run run_pruned.py sweep over all HML models (except Nemotron):
#   top    : 1, 2 heads
#   bottom : 1%, 2%, 5%, 10% heads
#
# Optional env knobs:
#   PYTHON_BIN, PORT, MAX_MODEL_LEN, MAX_TOKENS, WORKERS,
#   STARTUP_TIMEOUT, RESULTS_DIR, TEMP_ROOT,
#   RUN_EVALUATE=true|false, KEEP_TEMP_MODEL=true|false,
#   STOP_ON_ERROR=true|false, MODEL=<hf-id-or-model-dir-name>
#
# Usage:
#   bash scripts/run/run_pruned_sweep.sh
#   MODEL=Qwen/Qwen3-8B bash scripts/run/run_pruned_sweep.sh

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TOKENS="${MAX_TOKENS:-28000}"
WORKERS="${WORKERS:-32}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-9000}"
RESULTS_DIR="${RESULTS_DIR:-results/hml}"
TEMP_ROOT="${TEMP_ROOT:-data/tmp_pruned_models}"
RUN_EVALUATE="${RUN_EVALUATE:-true}"
KEEP_TEMP_MODEL="${KEEP_TEMP_MODEL:-false}"
STOP_ON_ERROR="${STOP_ON_ERROR:-false}"

# Same list as run_check.sh, excluding nvidia--NVIDIA-Nemotron-Nano-9B-v2.
MODEL_DIRS=(
  # "deepseek-ai--DeepSeek-R1-Distill-Qwen-7B"
  "deepseek-ai--DeepSeek-R1-Distill-Llama-8B"
  "deepseek-ai--DeepSeek-R1-0528-Qwen3-8B"
  "meta-llama--Llama-3.2-3B-Instruct"
  "Qwen--Qwen3-4B"
  "Qwen--Qwen3-8B"
  # "Qwen--Qwen3-0.6B"
  "Qwen--Qwen3-14B"
  # "deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B"
  # "deepseek-ai--DeepSeek-R1-Distill-Qwen-14B"
  # "microsoft--Phi-4-mini-reasoning"
)

if [[ -n "${MODEL:-}" ]]; then
  if [[ "$MODEL" == */* ]]; then
    MODEL_DIRS=("$(echo "$MODEL" | sed 's|/|--|g')")
  else
    MODEL_DIRS=("$MODEL")
  fi
fi

TOP_COUNTS=(1 2 5 10)
BOTTOM_PCTS=(1 2 5 10)
DATASETS=(humaneval mbpp livecodebench)

TOTAL_RUNS=0
FAILED_RUNS=0

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

# Print dataset names that still have pending eligible entries for a given
# pruning setting (model + top/bottom variant).
pending_datasets_for_case() {
  local model_dir="$1"
  local model_short="$2"
  local out_subdir="$3"
  shift 3

  "$PYTHON_BIN" - "$PROJECT_DIR" "$model_dir" "$model_short" "$out_subdir" "$@" <<'PYEOF'
import json
import os
import sys

project_dir, model_dir, model_short, out_subdir, *datasets = sys.argv[1:]

file_map = {
  "humaneval": "humaneval.json",
  "mbpp": "mbpp.json",
  "livecodebench": "livecodebench_v6.json",
}

def truthy_verdict(value):
  if isinstance(value, bool):
    return value
  if isinstance(value, (int, float)):
    return int(value) == 1
  if isinstance(value, str):
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "pass", "passed"}
  return False

for ds in datasets:
  fname = file_map.get(ds)
  if fname is None:
    continue

  base_path = os.path.join(project_dir, "data", "benchmarks", model_dir, fname)
  out_path = os.path.join(project_dir, "data", "benchmarks", "pruning", model_short, out_subdir, fname)

  try:
    with open(base_path) as f:
      base_rows = json.load(f)
  except Exception:
    # If base input cannot be read, keep this dataset pending so the caller
    # surfaces the issue in the main run.
    print(ds)
    continue

  if not isinstance(base_rows, list):
    print(ds)
    continue

  verdict_present = 0
  eligible = []
  for i, row in enumerate(base_rows):
    if not isinstance(row, dict):
      continue
    if "verdict" in row:
      verdict_present += 1
    if truthy_verdict(row.get("verdict")):
      eligible.append(i)

  # Match run_pruned.py behavior: if there are no verdict fields, this dataset
  # is not in a valid "complete" state for pruning.
  if verdict_present == 0:
    print(ds)
    continue

  # No base-true rows means nothing to run for this dataset.
  if not eligible:
    continue

  try:
    with open(out_path) as f:
      out_rows = json.load(f)
  except Exception:
    print(ds)
    continue

  if not isinstance(out_rows, list):
    print(ds)
    continue

  pending_count = 0
  for idx in eligible:
    if idx >= len(out_rows):
      pending_count += 1
      continue
    row = out_rows[idx] if isinstance(out_rows[idx], dict) else {}
    if not row.get("reasoning_chain") and not row.get("solution"):
      pending_count += 1

  if pending_count > 0:
    print(ds)
PYEOF
}

run_and_track() {
  local label="$1"
  shift

  TOTAL_RUNS=$((TOTAL_RUNS + 1))
  echo ""
  echo "[$(timestamp)] [run ${TOTAL_RUNS}] ${label}"
  echo "[$(timestamp)] command: $*"

  if "$@"; then
    echo "[$(timestamp)] [ok] ${label}"
    return 0
  fi

  FAILED_RUNS=$((FAILED_RUNS + 1))
  echo "[$(timestamp)] [fail] ${label}"
  if [[ "$STOP_ON_ERROR" == "true" ]]; then
    echo "[$(timestamp)] STOP_ON_ERROR=true -> exiting early"
    exit 1
  fi
  return 1
}

evaluate_folder() {
  local folder="$1"
  local label="$2"

  if [[ "$RUN_EVALUATE" != "true" ]]; then
    return 0
  fi

  if [[ ! -d "$folder" ]]; then
    echo "[$(timestamp)] [warn] missing folder for evaluation: $folder"
    return 0
  fi

  run_and_track "evaluate ${label}" \
    "$PYTHON_BIN" src/evaluate.py "$PROJECT_DIR/$folder"
}

echo "========================================================"
echo " run_pruned sweep started at $(timestamp)"
echo " Python:      $PYTHON_BIN"
echo " Models:      ${#MODEL_DIRS[@]}"
echo " Results dir: $RESULTS_DIR"
echo " Datasets:    ${DATASETS[*]}"
echo "========================================================"

for MODEL_DIR in "${MODEL_DIRS[@]}"; do
  BASE_MODEL="${MODEL_DIR/--//}"
  MODEL_SHORT="${BASE_MODEL##*/}"

  echo ""
  echo "========================================================"
  echo " Model: $BASE_MODEL"
  echo "========================================================"

  for K in "${TOP_COUNTS[@]}"; do
    OUT_SUBDIR="top_${K}"
    OUT_DIR="data/benchmarks/pruning/${MODEL_SHORT}/${OUT_SUBDIR}"

    mapfile -t PENDING_DATASETS < <(
      pending_datasets_for_case "$MODEL_DIR" "$MODEL_SHORT" "$OUT_SUBDIR" "${DATASETS[@]}"
    )
    if [[ ${#PENDING_DATASETS[@]} -eq 0 ]]; then
      echo "[$(timestamp)] [skip] ${BASE_MODEL} top-${K} already complete"
      continue
    fi
    echo "[$(timestamp)] [pending] ${BASE_MODEL} top-${K}: ${PENDING_DATASETS[*]}"

    CMD=(
      "$PYTHON_BIN" src/run_pruned.py
      --base-model "$BASE_MODEL"
      --prune-type top
      --prune-k "$K"
      --datasets "${PENDING_DATASETS[@]}"
      --results-dir "$RESULTS_DIR"
      --port "$PORT"
      --max-model-len "$MAX_MODEL_LEN"
      --max-tokens "$MAX_TOKENS"
      --workers "$WORKERS"
      --startup-timeout "$STARTUP_TIMEOUT"
      --temp-root "$TEMP_ROOT"
    )
    if [[ "$KEEP_TEMP_MODEL" == "true" ]]; then
      CMD+=(--keep-temp-model)
    fi

    run_and_track "${BASE_MODEL} top-${K}" "${CMD[@]}"
    evaluate_folder "$OUT_DIR" "${BASE_MODEL} ${OUT_SUBDIR}"
  done

  for PCT in "${BOTTOM_PCTS[@]}"; do
    OUT_SUBDIR="bottom_pct_${PCT}"
    OUT_DIR="data/benchmarks/pruning/${MODEL_SHORT}/${OUT_SUBDIR}"

    mapfile -t PENDING_DATASETS < <(
      pending_datasets_for_case "$MODEL_DIR" "$MODEL_SHORT" "$OUT_SUBDIR" "${DATASETS[@]}"
    )
    if [[ ${#PENDING_DATASETS[@]} -eq 0 ]]; then
      echo "[$(timestamp)] [skip] ${BASE_MODEL} bottom-${PCT}% already complete"
      continue
    fi
    echo "[$(timestamp)] [pending] ${BASE_MODEL} bottom-${PCT}%: ${PENDING_DATASETS[*]}"

    CMD=(
      "$PYTHON_BIN" src/run_pruned.py
      --base-model "$BASE_MODEL"
      --prune-type bottom
      --prune-percent "$PCT"
      --datasets "${PENDING_DATASETS[@]}"
      --results-dir "$RESULTS_DIR"
      --port "$PORT"
      --max-model-len "$MAX_MODEL_LEN"
      --max-tokens "$MAX_TOKENS"
      --workers "$WORKERS"
      --startup-timeout "$STARTUP_TIMEOUT"
      --temp-root "$TEMP_ROOT"
    )
    if [[ "$KEEP_TEMP_MODEL" == "true" ]]; then
      CMD+=(--keep-temp-model)
    fi

    run_and_track "${BASE_MODEL} bottom-${PCT}%" "${CMD[@]}"
    evaluate_folder "$OUT_DIR" "${BASE_MODEL} ${OUT_SUBDIR}"
  done
done

echo ""
echo "========================================================"
echo " run_pruned sweep finished at $(timestamp)"
echo " Total jobs attempted: $TOTAL_RUNS"
echo " Failed jobs:          $FAILED_RUNS"
echo "========================================================"

if [[ "$FAILED_RUNS" -gt 0 ]]; then
  exit 1
fi
