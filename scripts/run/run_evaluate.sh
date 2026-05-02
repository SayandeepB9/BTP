#!/usr/bin/env bash
# Evaluate solutions in a model results folder.
#
# Usage:
#   bash scripts/run/run_evaluate.sh <model-name>
#   bash scripts/run/run_evaluate.sh deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
#   MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B bash scripts/run/run_evaluate.sh
#   FOLDER=data/benchmarks/Qwen--Qwen3-8B bash scripts/run/run_evaluate.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Resolve the folder to evaluate.
# Priority: FOLDER env var > model name arg > MODEL env var
if [[ -n "${FOLDER:-}" ]]; then
  : # use as-is
elif [[ -n "${1:-}" ]]; then
  # Accept a full HF model name (e.g. "Qwen/Qwen3-4B") or a short slug
  MODEL_ARG="$1"
  FOLDER="data/benchmarks/$(echo "$MODEL_ARG" | sed 's|/|--|g')"
elif [[ -n "${MODEL:-}" ]]; then
  FOLDER="data/benchmarks/$(echo "$MODEL" | sed 's|/|--|g')"
else
  echo "Usage: bash scripts/run/run_evaluate.sh <model-name>"
  echo "  or:  MODEL=<model-name> bash scripts/run/run_evaluate.sh"
  echo "  or:  FOLDER=data/benchmarks/<dir> bash scripts/run/run_evaluate.sh"
  exit 1
fi

echo "========================================"
echo "Evaluating solutions"
echo "========================================"
echo "Folder: $FOLDER"
echo "========================================"

export PYTHONPATH="/tmp/LiveCodeBench:${PYTHONPATH:-}"

python3 "$PROJECT_DIR/src/evaluate.py" "$PROJECT_DIR/$FOLDER"
