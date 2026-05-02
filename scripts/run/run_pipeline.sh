#!/usr/bin/env bash
# Run the full HML pipeline for a single model with smart step detection.
#
# SKIP_INFERENCE / SKIP_EVALUATE / SKIP_ANALYSIS / SKIP_TAYLOR:
#   unset/empty (default) – smart mode: auto-detect based on data quality checks
#   "true"                – always run this step
#   "false"               – skip this step entirely
#
# Smart mode runs only the missing/invalid targets for each step.
# Upstream reruns mark downstream datasets as "dirty" so dependent results are
# refreshed only where needed.
# Smart conditions:
#   Inference  – any dataset < 50 % done → run on only the incomplete datasets
#   Evaluate   – datasets touched by inference, plus datasets with missing verdicts
#   Analysis   – datasets touched by evaluate, plus missing/invalid entropy/ablation targets
#   Taylor     – datasets touched by evaluate, plus missing/invalid Taylor targets
#
# The pipeline aborts only if a step was run but its post-run check still fails.
#
# Usage:
#   bash scripts/run/run_pipeline.sh [<model-name>]
#   MODEL=Qwen/Qwen3-4B bash scripts/run/run_pipeline.sh
#   SKIP_INFERENCE=false SKIP_EVALUATE=false bash scripts/run/run_pipeline.sh
#
# Tunable knobs (all have defaults):
#   PORT, MAX_MODEL_LEN, MAX_TOKENS, WORKERS, USE_COMPLETIONS, SEED,
#   NUM_SAMPLES_ABLATION, NUM_SAMPLES_ENTROPY, NUM_SAMPLES_TAYLOR,
#   TOP_K, NUM_ITERS, PRUNE_PER_ITER

set -u   # catch unset variable references; -e / pipefail intentionally omitted

# ── model ─────────────────────────────────────────────────────────────────────
# Edit MODEL here to set a default without passing it on the command line:
# MODEL="nvidia/NVIDIA-Nemotron-Nano-9B-v2"
# MODEL="microsoft/Phi-4-mini-reasoning"
# MODEL="google/gemma-4-E2B-it"
# MODEL="google/gemma-4-E4B-it"
# MODEL="google/gemma-4-26B-A4B-it"
# MODEL="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
# MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# MODEL="openai/gpt-oss-20b"
# MODEL="Qwen/Qwen3-8B"
# MODEL="Qwen/Qwen3-0.6B"
MODEL="Qwen/Qwen3-14B"
# MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

if [[ -n "${1:-}" ]]; then
  MODEL="$1"
fi
if [[ -z "${MODEL:-}" ]]; then
  echo "Usage: bash scripts/run/run_pipeline.sh <model-name>"
  echo "  or:  MODEL=<model-name> bash scripts/run/run_pipeline.sh"
  exit 1
fi

# ── conda env python binaries ─────────────────────────────────────────────────
# Inference (step 1) runs in vllm_env; everything else runs in myenv.
if command -v conda &>/dev/null; then
  CONDA_BASE="$(conda info --base 2>/dev/null)"
  PYTHON_MYENV="${CONDA_BASE}/envs/myenv/bin/python3"
  PYTHON_VLLM="${CONDA_BASE}/envs/vllm_env/bin/python3"
  [[ ! -x "$PYTHON_MYENV" ]] && { echo "WARNING: $PYTHON_MYENV not found — falling back to python3"; PYTHON_MYENV="python3"; }
  [[ ! -x "$PYTHON_VLLM" ]] && { echo "WARNING: $PYTHON_VLLM not found — falling back to python3"; PYTHON_VLLM="python3"; }
else
  echo "WARNING: conda not found — using system python3 for all steps"
  PYTHON_MYENV="python3"
  PYTHON_VLLM="python3"
fi

# ── tunables ──────────────────────────────────────────────────────────────────
# Auto-select port: use 8000 if free, otherwise 8001
if [[ -z "${PORT:-}" ]]; then
  if ss -tlnH sport = :8000 2>/dev/null | grep -q ':8000'; then
    PORT=8001
    echo "[pipeline] Port 8000 is in use — using port 8001."
  else
    PORT=8000
  fi
fi

MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TOKENS="${MAX_TOKENS:-28000}"
WORKERS="${WORKERS:-32}"
USE_COMPLETIONS="${USE_COMPLETIONS:-false}"
SEED="${SEED:-42}"

NUM_SAMPLES_ABLATION="${NUM_SAMPLES_ABLATION:-100}"
NUM_SAMPLES_ENTROPY="${NUM_SAMPLES_ENTROPY:-100}"
NUM_SAMPLES_TAYLOR="${NUM_SAMPLES_TAYLOR:-100}"
TOP_K="${TOP_K:-20}"
NUM_ITERS="${NUM_ITERS:-100}"
PRUNE_PER_ITER="${PRUNE_PER_ITER:-5}"

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODEL_DIR_NAME="$(echo "$MODEL" | sed 's|/|--|g')"
HML_MODEL_DIR="data/benchmarks/${MODEL_DIR_NAME}"
MODEL_SHORT="${MODEL##*/}"
RESULTS_HML="results/hml"

HML_DATASETS=("humaneval" "mbpp" "livecodebench")

echo "========================================================"
echo " HML Full Pipeline  --  $(date)"
echo " Model:       $MODEL"
echo " HML dir:     $HML_MODEL_DIR"
echo " Python (myenv):    $PYTHON_MYENV"
echo " Python (vllm_env): $PYTHON_VLLM"
echo "========================================================"
echo ""

# ── Import shared check helpers from run_check.sh ─────────────────────────────
# Gives us: check_inference_json, check_results_json, check_npz
# shellcheck source=run_check.sh
source "$(dirname "${BASH_SOURCE[0]}")/run_check.sh"

# Override check_npz to use $PYTHON_MYENV (which has numpy) instead of python3
check_npz() {
    local fpath="$1" key="$2"
    "$PYTHON_MYENV" - "$fpath" "$key" <<'PYEOF'
import sys, numpy as np
fpath, key = sys.argv[1], sys.argv[2]
try:
    d = np.load(fpath, allow_pickle=True)
    if key not in d:
        print(f"MISSING_KEY:{key}")
        sys.exit(0)
    v = d[key].astype(float)
    issues = []
    if np.isnan(v).any():
        issues.append(f"NaN({int(np.isnan(v).sum())})")
    if (v == 0).all():
        issues.append("all-zero")
    if issues:
        print("ISSUES:" + ",".join(issues))
    else:
        print("OK")
except Exception as e:
    print(f"ERROR:{e}")
PYEOF
}

# ── Step state ────────────────────────────────────────────────────────────────
_inference_ran=false
_evaluate_ran=false
_analysis_ran=false
_evaluate_targets=()

# ── Condition helpers ─────────────────────────────────────────────────────────

# Prints space-separated dataset names that are < 50% complete; empty = all done.
_check_inference() {
  local needs=()
  local ds_names=("humaneval"      "mbpp"      "livecodebench")
  local ds_files=("humaneval.json" "mbpp.json" "livecodebench_v6.json")
  for i in "${!ds_names[@]}"; do
    local fpath="$HML_MODEL_DIR/${ds_files[$i]}"
    if [[ ! -f "$fpath" ]]; then
      needs+=("${ds_names[$i]}"); continue
    fi
    local result
    result=$(check_inference_json "$fpath")
    if [[ "$result" == ERROR:* ]]; then
      needs+=("${ds_names[$i]}"); continue
    fi
    local total _done empty errors loop
    IFS=',' read -r total _done empty errors loop <<< "$result"
    # _done * 2 < total  ≡  done/total < 0.5 (integer arithmetic, no floats)
    if [[ "$total" -eq 0 ]] || [[ $((_done * 2)) -lt "$total" ]]; then
      needs+=("${ds_names[$i]}")
    fi
  done
  echo "${needs[*]}"
}

# Prints space-separated dataset names with entries that have solution but no verdict.
_check_evaluate() {
  local needs=()
  local ds_names=("humaneval"      "mbpp"      "livecodebench")
  local ds_files=("humaneval.json" "mbpp.json" "livecodebench_v6.json")
  for i in "${!ds_names[@]}"; do
    local fpath="$HML_MODEL_DIR/${ds_files[$i]}"
    [[ ! -f "$fpath" ]] && continue
    local missing
    missing=$("$PYTHON_MYENV" - "$fpath" <<'PYEOF'
import sys, json
with open(sys.argv[1]) as f:
    data = json.load(f)
print(sum(1 for e in data if e.get("solution") and "verdict" not in e))
PYEOF
2>/dev/null || echo "?")
    if [[ "$missing" != "0" && "$missing" != "?" && -n "$missing" ]]; then
      needs+=("${ds_names[$i]}")
    fi
  done
  echo "${needs[*]}"
}

# Prints one target per line for missing/invalid analysis outputs:
#   entropy|<dataset>|<mode>
#   ablation|<dataset>|<mode>
_check_analysis_targets() {
  local ds mode variant npz res
  for ds in "${HML_DATASETS[@]}"; do
    for mode in regular chain_code; do
      if [[ "$mode" == "chain_code" ]]; then
        variant="entropy_loss_results_chain_code"
      else
        variant="entropy_loss_results"
      fi
      npz="$RESULTS_HML/$ds/$variant/$MODEL_SHORT/$ds/per_token.npz"
      if [[ ! -f "$npz" ]]; then
        echo "entropy|$ds|$mode"
      else
        res=$(check_npz "$npz" "avg_entropy")
        [[ "$res" != "OK" ]] && echo "entropy|$ds|$mode"
      fi

      if [[ "$mode" == "chain_code" ]]; then
        variant="head_ablation_results_chain_code"
      else
        variant="head_ablation_results"
      fi
      npz="$RESULTS_HML/$ds/$variant/$MODEL_SHORT/$ds/ablation.npz"
      if [[ ! -f "$npz" ]]; then
        echo "ablation|$ds|$mode"
      else
        res=$(check_npz "$npz" "ablation_mean")
        [[ "$res" != "OK" ]] && echo "ablation|$ds|$mode"
      fi
    done
  done
}

# Prints one target per line for missing/invalid Taylor outputs:
#   <dataset>|<mode>
_check_taylor_targets() {
  local ds mode variant jpath nan_count
  for ds in "${HML_DATASETS[@]}"; do
    for mode in regular chain_code; do
      if [[ "$mode" == "chain_code" ]]; then
        variant="iterative_taylor_chain_code"
      else
        variant="iterative_taylor"
      fi
      jpath="$RESULTS_HML/$ds/$variant/$MODEL_SHORT/iterative_taylor_results.json"
      if [[ ! -f "$jpath" ]]; then
        echo "$ds|$mode"
        continue
      fi

      nan_count=$("$PYTHON_MYENV" - "$jpath" <<'PYEOF'
import sys
try:
    with open(sys.argv[1]) as f:
        t = f.read()
    print(t.count("NaN") + t.count("nan") + t.count("Infinity"))
except Exception:
    print("?")
PYEOF
)
      [[ "$nan_count" != "0" ]] && echo "$ds|$mode"
    done
  done
}

_add_all_analysis_targets_for_dataset() {
  local ds="$1"
  _analysis_target_set["entropy|$ds|regular"]=1
  _analysis_target_set["entropy|$ds|chain_code"]=1
  _analysis_target_set["ablation|$ds|regular"]=1
  _analysis_target_set["ablation|$ds|chain_code"]=1
}

_add_all_taylor_targets_for_dataset() {
  local ds="$1"
  _taylor_target_set["$ds|regular"]=1
  _taylor_target_set["$ds|chain_code"]=1
}

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Inference
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Step 1/4: Inference  [vllm_env] ==="
_run_inference_datasets=()

case "${SKIP_INFERENCE:-}" in
  false)
    echo "[pipeline] Inference: disabled (SKIP_INFERENCE=false)."
    ;;
  true)
    _run_inference_datasets=("${HML_DATASETS[@]}")
    echo "[pipeline] Inference: forced on all datasets."
    ;;
  *)
    _needed=$(_check_inference)
    if [[ -n "$_needed" ]]; then
      read -ra _run_inference_datasets <<< "$_needed"
      echo "[pipeline] Inference needed for: ${_run_inference_datasets[*]}"
    else
      echo "[pipeline] Inference: all datasets ≥ 50% complete — skipping."
    fi
    ;;
esac

if [[ ${#_run_inference_datasets[@]} -gt 0 ]]; then
  _run_inference_cli_datasets=()
  for _ds in "${_run_inference_datasets[@]}"; do
    if [[ "$_ds" == "livecodebench" ]]; then
      _run_inference_cli_datasets+=("livecodebench_v6")
    else
      _run_inference_cli_datasets+=("$_ds")
    fi
  done

  EXTRA_FLAGS="--serve --port ${PORT} --max-model-len ${MAX_MODEL_LEN} --datasets ${_run_inference_cli_datasets[*]}"
  [[ "$USE_COMPLETIONS" == "true" ]] && EXTRA_FLAGS="$EXTRA_FLAGS --use-completions"

  unset CUDA_HOME CUDA_PATH CUDA_ROOT CONDA_DEFAULT_ENV CONDA_PREFIX

  "$PYTHON_VLLM" src/run_inference.py \
    --model-name "$MODEL" \
    --api-base "http://localhost:${PORT}/v1" \
    --max-tokens "$MAX_TOKENS" \
    --temperature 0.0 \
    --workers "$WORKERS" \
    $EXTRA_FLAGS || true

  # Post-check: verify every dataset we targeted is now ≥ 50%
  _still=$(_check_inference)
  _failed=()
  for _ds in "${_run_inference_datasets[@]}"; do
    echo "$_still" | grep -qw "$_ds" && _failed+=("$_ds")
  done
  if [[ ${#_failed[@]} -gt 0 ]]; then
    echo "[pipeline] ERROR: Inference still < 50% for: ${_failed[*]}"
    exit 1
  fi

  _inference_ran=true
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Evaluate
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Step 2/4: Evaluate  [myenv] ==="
_run_evaluate=false
_evaluate_recheck_all=false
declare -A _evaluate_target_set=()

case "${SKIP_EVALUATE:-}" in
  false)
    echo "[pipeline] Evaluate: disabled (SKIP_EVALUATE=false)."
    ;;
  true)
    _run_evaluate=true
    _evaluate_recheck_all=true
    for _ds in "${HML_DATASETS[@]}"; do
      _evaluate_target_set["$_ds"]=1
    done
    echo "[pipeline] Evaluate: forced on all datasets."
    ;;
  *)
    if [[ "$_inference_ran" == true ]]; then
      for _ds in "${_run_inference_datasets[@]}"; do
        _evaluate_target_set["$_ds"]=1
      done
    fi

    _needed=$(_check_evaluate)
    if [[ -n "$_needed" ]]; then
      read -ra _needed_ds <<< "$_needed"
      for _ds in "${_needed_ds[@]}"; do
        _evaluate_target_set["$_ds"]=1
      done
    fi

    if [[ ${#_evaluate_target_set[@]} -gt 0 ]]; then
      _run_evaluate=true
      echo "[pipeline] Evaluate needed for targeted datasets."
    else
      echo "[pipeline] Evaluate: all datasets have verdicts — skipping."
    fi
    ;;
esac

if [[ "$_run_evaluate" == true ]]; then
  if [[ ! -d "$HML_MODEL_DIR" ]]; then
    echo "[pipeline] ERROR: HML dir not found: $HML_MODEL_DIR"
    exit 1
  fi

  mapfile -t _evaluate_targets < <(printf '%s\n' "${!_evaluate_target_set[@]}" | sort)
  echo "[pipeline] Evaluate targets: ${_evaluate_targets[*]}"

  _evaluate_extra_flags=()
  [[ "$_evaluate_recheck_all" == true ]] && _evaluate_extra_flags+=("--recheck-all")

  PYTHONPATH="/tmp/LiveCodeBench:${PYTHONPATH:-}" \
    "$PYTHON_MYENV" src/evaluate.py "$HML_MODEL_DIR" --datasets "${_evaluate_targets[@]}" "${_evaluate_extra_flags[@]}" || true

  # Post-check targeted datasets only
  _still=$(_check_evaluate)
  _failed=()
  for _ds in "${_evaluate_targets[@]}"; do
    echo " $_still " | grep -q " $_ds " && _failed+=("$_ds")
  done
  if [[ ${#_failed[@]} -gt 0 ]]; then
    echo "[pipeline] ERROR: Evaluation still missing verdicts for: ${_failed[*]}"
    exit 1
  fi

  _evaluate_ran=true
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: HML Analysis (entropy lens + head ablation)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Step 3/4: HML Analysis  [myenv] ==="
_run_analysis=false
declare -A _analysis_target_set=()

case "${SKIP_ANALYSIS:-}" in
  false)
    echo "[pipeline] Analysis: disabled (SKIP_ANALYSIS=false)."
    ;;
  true)
    _run_analysis=true
    for _ds in "${HML_DATASETS[@]}"; do
      _add_all_analysis_targets_for_dataset "$_ds"
    done
    echo "[pipeline] Analysis: forced on all dataset/mode targets."
    ;;
  *)
    if [[ "$_evaluate_ran" == true ]]; then
      for _ds in "${_evaluate_targets[@]}"; do
        _add_all_analysis_targets_for_dataset "$_ds"
      done
    fi

    while IFS= read -r _target; do
      [[ -z "$_target" ]] && continue
      _analysis_target_set["$_target"]=1
    done < <(_check_analysis_targets)

    if [[ ${#_analysis_target_set[@]} -gt 0 ]]; then
      _run_analysis=true
      echo "[pipeline] Analysis: targeted rerun required."
    else
      echo "[pipeline] Analysis: all targets valid — skipping."
    fi
    ;;
esac

if [[ "$_run_analysis" == true ]]; then
  mapfile -t _analysis_targets < <(printf '%s\n' "${!_analysis_target_set[@]}" | sort)
  echo "[pipeline] Analysis targets (${#_analysis_targets[@]}):"
  printf '  - %s\n' "${_analysis_targets[@]}"

  for _target in "${_analysis_targets[@]}"; do
    IFS='|' read -r _kind _ds _mode <<< "$_target"

    if [[ "$_kind" == "entropy" ]]; then
      if [[ "$_mode" == "chain_code" ]]; then
        _variant="entropy_loss_results_chain_code"
      else
        _variant="entropy_loss_results"
      fi
      echo "[pipeline] Running entropy: dataset=$_ds mode=$_mode"
      "$PYTHON_MYENV" src/entropy_lens.py \
        --dataset "$_ds" \
        --model "$MODEL" \
        --num_samples "$NUM_SAMPLES_ENTROPY" \
        --mode "$_mode" \
        --hml_model_dir "$HML_MODEL_DIR" \
        --seed "$SEED" \
        --out_dir "$_variant" || true
    else
      if [[ "$_mode" == "chain_code" ]]; then
        _variant="head_ablation_results_chain_code"
      else
        _variant="head_ablation_results"
      fi
      echo "[pipeline] Running ablation: dataset=$_ds mode=$_mode"
      "$PYTHON_MYENV" src/head_ablation.py \
        --dataset "$_ds" \
        --model "$MODEL" \
        --num_samples "$NUM_SAMPLES_ABLATION" \
        --top_k "$TOP_K" \
        --mode "$_mode" \
        --hml_model_dir "$HML_MODEL_DIR" \
        --seed "$SEED" \
        --out_dir "$_variant" || true
    fi
  done

  # Post-check targeted analysis keys only
  declare -A _still_analysis_set=()
  while IFS= read -r _target; do
    [[ -z "$_target" ]] && continue
    _still_analysis_set["$_target"]=1
  done < <(_check_analysis_targets)

  _failed_analysis=()
  for _target in "${_analysis_targets[@]}"; do
    if [[ -n "${_still_analysis_set[$_target]:-}" ]]; then
      _failed_analysis+=("$_target")
    fi
  done
  if [[ ${#_failed_analysis[@]} -gt 0 ]]; then
    echo "[pipeline] ERROR: Analysis targets still missing/invalid after rerun: ${_failed_analysis[*]}"
    exit 1
  fi

  _analysis_ran=true
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Iterative Taylor Pruning
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Step 4/4: Iterative Taylor Pruning  [myenv] ==="
_run_taylor=false
declare -A _taylor_target_set=()

case "${SKIP_TAYLOR:-}" in
  false)
    echo "[pipeline] Taylor: disabled (SKIP_TAYLOR=false)."
    ;;
  true)
    _run_taylor=true
    for _ds in "${HML_DATASETS[@]}"; do
      _add_all_taylor_targets_for_dataset "$_ds"
    done
    echo "[pipeline] Taylor: forced on all dataset/mode targets."
    ;;
  *)
    if [[ "$_evaluate_ran" == true ]]; then
      for _ds in "${_evaluate_targets[@]}"; do
        _add_all_taylor_targets_for_dataset "$_ds"
      done
    fi

    while IFS= read -r _target; do
      [[ -z "$_target" ]] && continue
      _taylor_target_set["$_target"]=1
    done < <(_check_taylor_targets)

    if [[ ${#_taylor_target_set[@]} -gt 0 ]]; then
      _run_taylor=true
      echo "[pipeline] Taylor: targeted rerun required."
    else
      echo "[pipeline] Taylor: all targets valid — skipping."
    fi
    ;;
esac

if [[ "$_run_taylor" == true ]]; then
  mapfile -t _taylor_targets < <(printf '%s\n' "${!_taylor_target_set[@]}" | sort)
  echo "[pipeline] Taylor targets (${#_taylor_targets[@]}):"
  printf '  - %s\n' "${_taylor_targets[@]}"

  for _target in "${_taylor_targets[@]}"; do
    IFS='|' read -r _ds _mode <<< "$_target"
    echo "[pipeline] Running Taylor: dataset=$_ds mode=$_mode"
    "$PYTHON_MYENV" src/iterative_taylor_pruning.py \
      --model "$MODEL" \
      --dataset "$_ds" \
      --hml_model_dir "$HML_MODEL_DIR" \
      --mode "$_mode" \
      --num_samples "$NUM_SAMPLES_TAYLOR" \
      --num_iters "$NUM_ITERS" \
      --prune_per_iter "$PRUNE_PER_ITER" \
      --seed "$SEED" || true
  done

  # Post-check targeted Taylor keys only
  declare -A _still_taylor_set=()
  while IFS= read -r _target; do
    [[ -z "$_target" ]] && continue
    _still_taylor_set["$_target"]=1
  done < <(_check_taylor_targets)

  _failed_taylor=()
  for _target in "${_taylor_targets[@]}"; do
    if [[ -n "${_still_taylor_set[$_target]:-}" ]]; then
      _failed_taylor+=("$_target")
    fi
  done
  if [[ ${#_failed_taylor[@]} -gt 0 ]]; then
    echo "[pipeline] ERROR: Taylor targets still missing/invalid after rerun: ${_failed_taylor[*]}"
    exit 1
  fi
fi
echo ""

echo "========================================================"
echo " Pipeline complete for ${MODEL_SHORT}  --  $(date)"
echo " Inference : $HML_MODEL_DIR"
echo " Analysis  : results/hml/"
echo "========================================================"
