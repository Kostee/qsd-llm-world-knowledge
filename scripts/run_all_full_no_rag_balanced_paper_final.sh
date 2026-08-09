#!/usr/bin/env bash

# Reproduce the final no-RAG experiment on the 440-item balanced QSD dataset.
# Intended for Git Bash / Linux / macOS.
#
# Usage:
#   bash scripts/run_all_full_no_rag_balanced_paper_final.sh

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-balanced}"
LIMIT_MODE="${LIMIT_MODE:-all}"
REPEATS="${REPEATS:-5}"
SEED="${SEED:-42}"

MODELS=(
  "gpt-5.1"
  "gpt-4o"
  "gpt-4o-mini"
  "qwen:qwen-max"
  "qwen:qwen3-8b"
  "openrouter:meta-llama/llama-3.1-70b-instruct"
  "openrouter:meta-llama/llama-3.1-8b-instruct"
)

SUCCESS_MODELS=()
FAILED_MODELS=()

for model in "${MODELS[@]}"; do
  echo "------------------------------------------------------------"
  echo "Running model: $model"
  echo "------------------------------------------------------------"

  "$PYTHON_BIN" src/llm_zero_shot.py \
    --dataset "$DATASET" \
    --limit "$LIMIT_MODE" \
    --model "$model" \
    --repeats "$REPEATS" \
    --seed "$SEED"

  rc=$?

  if [[ $rc -eq 0 ]]; then
    echo "[OK] $model"
    SUCCESS_MODELS+=("$model")
  else
    echo "[FAIL] $model (exit code: $rc)"
    FAILED_MODELS+=("$model")
  fi

  echo
  sleep 1
done

echo "============================================================"
echo "Experiment finished."
echo "Successful models: ${#SUCCESS_MODELS[@]}"
for model in "${SUCCESS_MODELS[@]}"; do
  echo "  - $model"
done

echo
echo "Failed models: ${#FAILED_MODELS[@]}"
for model in "${FAILED_MODELS[@]}"; do
  echo "  - $model"
done
echo "============================================================"

if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
  exit 1
fi

exit 0
