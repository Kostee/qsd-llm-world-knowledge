#!/usr/bin/env bash

# Run full no-RAG evaluation on the final paper pseudo dataset for the 7 paper models.
# Intended for Git Bash / Linux / macOS.
# Usage:
#   bash scripts/run_all_full_no_rag_pseudo_paper_final.sh
# Optional overrides:
#   DATASET_PATH=data/private/pseudo_paper_final_model_ready.csv REPEATS=5 SEED=42 bash scripts/run_all_full_no_rag_pseudo_paper_final.sh

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_PATH="${DATASET_PATH:-data/private/pseudo_paper_final_model_ready.csv}"
LIMIT_MODE="${LIMIT_MODE:-all}"
REPEATS="${REPEATS:-5}"
SEED="${SEED:-42}"

MODELS=(
  "gpt-4o"
  "gpt-5.1"
  "gpt-4o-mini"
  "qwen:qwen-max"
  "qwen:qwen3-8b"
  "openrouter:meta-llama/llama-3.1-70b-instruct"
  "openrouter:meta-llama/llama-3.1-8b-instruct"
)

if [[ ! -f "$DATASET_PATH" ]]; then
  echo "[ERROR] Dataset not found: $DATASET_PATH"
  echo "This script expects the final model-ready dataset at that path."
  exit 1
fi

echo "============================================================"
echo "Repo root   : $REPO_ROOT"
echo "Python      : $PYTHON_BIN"
echo "Dataset     : $DATASET_PATH"
echo "Limit       : $LIMIT_MODE"
echo "Repeats     : $REPEATS"
echo "Seed        : $SEED"
echo "Models      : ${#MODELS[@]}"
echo "============================================================"
echo

SUCCESS_MODELS=()
FAILED_MODELS=()

for model in "${MODELS[@]}"; do
  echo "------------------------------------------------------------"
  echo "Running model: $model"
  echo "Started at  : $(date '+%Y-%m-%d %H:%M:%S')"
  echo "------------------------------------------------------------"

  "$PYTHON_BIN" src/llm_zero_shot.py \
    --dataset "$DATASET_PATH" \
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
  echo "Finished at : $(date '+%Y-%m-%d %H:%M:%S')"
  echo
  sleep 1
done

echo "============================================================"
echo "Run finished."
echo "Successful: ${#SUCCESS_MODELS[@]}"
for model in "${SUCCESS_MODELS[@]}"; do
  echo "  - $model"
done

echo "Failed    : ${#FAILED_MODELS[@]}"
for model in "${FAILED_MODELS[@]}"; do
  echo "  - $model"
done
echo "============================================================"

if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
  exit 1
fi
