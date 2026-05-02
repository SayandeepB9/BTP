#!/usr/bin/env bash
# Run inference on all benchmarks using a vLLM-served model.
#
# Usage:
#   bash scripts/run/run_inference.sh <model-name>
#   MODEL=Qwen/Qwen3-4B bash scripts/run/run_inference.sh
#   MODEL=Qwen/Qwen3-4B SERVE=true bash scripts/run/run_inference.sh
#
# Set SERVE=true to have the script auto-start and stop the vLLM server.
# Otherwise, start the server manually first (e.g. scripts/infra/serve_vllm.sh).

set -euo pipefail

# Resolve model from arg or env
if [[ -n "${1:-}" ]]; then
  MODEL="$1"
elif [[ -z "${MODEL:-}" ]]; then
  echo "Usage: bash scripts/run/run_inference.sh <model-name>"
  echo "  or:  MODEL=<model-name> bash scripts/run/run_inference.sh"
  exit 1
fi

PORT="${PORT:-8000}"
WORKERS="${WORKERS:-32}"
MAX_TOKENS="${MAX_TOKENS:-28000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
USE_COMPLETIONS="${USE_COMPLETIONS:-false}"
SERVE="${SERVE:-false}"
API_BASE="http://localhost:${PORT}/v1"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "========================================"
echo "Running inference"
echo "========================================"
echo "Model:        $MODEL"
echo "API:          $API_BASE"
echo "Workers:      $WORKERS"
echo "Max tokens:   $MAX_TOKENS"
echo "Serve mode:   $SERVE"
echo "========================================"

EXTRA_FLAGS=""
if [[ "$USE_COMPLETIONS" == "true" ]]; then
  EXTRA_FLAGS="$EXTRA_FLAGS --use-completions"
fi
if [[ "$SERVE" == "true" ]]; then
  EXTRA_FLAGS="$EXTRA_FLAGS --serve --port $PORT --max-model-len $MAX_MODEL_LEN"
  # Unset conda env vars so vLLM's CUDA detection isn't misled by the active env
  unset CUDA_HOME CUDA_PATH CUDA_ROOT CONDA_DEFAULT_ENV CONDA_PREFIX
fi

python3 "$PROJECT_DIR/src/run_inference.py" \
    --model-name "$MODEL" \
    --api-base "$API_BASE" \
    --max-tokens "$MAX_TOKENS" \
    --temperature 0.0 \
    --workers "$WORKERS" \
    $EXTRA_FLAGS
