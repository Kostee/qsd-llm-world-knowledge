#!/usr/bin/env bash

# Reproduce the final no-RAG pseudosentence experiment reported in the paper.
#
# The script first derives the 320-row model-ready evaluation file from the
# canonical public 160-item benchmark and then evaluates the seven paper models.
#
# Intended for Git Bash / Linux / macOS.
#
# Usage:
#   bash scripts/run_all_full_no_rag_pseudo_paper_final.sh
#
# Optional overrides:
#   PYTHON_BIN=python
#   SOURCE_DATASET_PATH=data/public/pseudosentences_emnlp2026.csv
#   DATASET_PATH=data/generated/pseudo_paper_final_model_ready.csv
#   PROMPT_ONLY_PATH=data/generated/pseudo_paper_final_prompt_only.csv
#   PREPARE_DATASET=1
#   LIMIT_MODE=all
#   REPEATS=5
#   SEED=42

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"

SOURCE_DATASET_PATH="${SOURCE_DATASET_PATH:-data/public/pseudosentences_emnlp2026.csv}"
DATASET_PATH="${DATASET_PATH:-data/generated/pseudo_paper_final_model_ready.csv}"
PROMPT_ONLY_PATH="${PROMPT_ONLY_PATH:-data/generated/pseudo_paper_final_prompt_only.csv}"

PREPARE_DATASET="${PREPARE_DATASET:-1}"
LIMIT_MODE="${LIMIT_MODE:-all}"
REPEATS="${REPEATS:-5}"
SEED="${SEED:-42}"

# Exact seven-model set used in the paper.
MODELS=(
  "gpt-5.1"
  "gpt-4o"
  "gpt-4o-mini"
  "qwen:qwen-max"
  "qwen:qwen3-8b"
  "openrouter:meta-llama/llama-3.1-70b-instruct"
  "openrouter:meta-llama/llama-3.1-8b-instruct"
)

echo "============================================================"
echo "Final pseudosentence experiment"
echo "============================================================"
echo "Repo root       : $REPO_ROOT"
echo "Python          : $PYTHON_BIN"
echo "Source dataset  : $SOURCE_DATASET_PATH"
echo "Model-ready data: $DATASET_PATH"
echo "Prompt-only data: $PROMPT_ONLY_PATH"
echo "Prepare dataset : $PREPARE_DATASET"
echo "Limit           : $LIMIT_MODE"
echo "Repeats         : $REPEATS"
echo "Seed            : $SEED"
echo "Models          : ${#MODELS[@]}"
echo "============================================================"
echo

if [[ "$PREPARE_DATASET" == "1" ]]; then
  if [[ ! -f "$SOURCE_DATASET_PATH" ]]; then
    echo "[ERROR] Canonical source dataset not found:"
    echo "        $SOURCE_DATASET_PATH"
    exit 1
  fi

  echo "[1/2] Preparing final pseudosentence evaluation dataset..."
  echo

  "$PYTHON_BIN" scripts/prepare_pseudo_paper_final.py \
    --input "$SOURCE_DATASET_PATH" \
    --model-ready-output "$DATASET_PATH" \
    --prompt-only-output "$PROMPT_ONLY_PATH"

  rc=$?

  if [[ $rc -ne 0 ]]; then
    echo
    echo "[ERROR] Dataset preparation failed (exit code: $rc)."
    exit "$rc"
  fi

  echo
  echo "[OK] Dataset preparation completed."
  echo
else
  echo "[INFO] PREPARE_DATASET=0; using an existing model-ready dataset."
  echo
fi

if [[ ! -f "$DATASET_PATH" ]]; then
  echo "[ERROR] Model-ready dataset not found:"
  echo "        $DATASET_PATH"
  exit 1
fi

echo "[2/2] Running model evaluation..."
echo

SUCCESS_MODELS=()
FAILED_MODELS=()

for model in "${MODELS[@]}"; do
  echo "------------------------------------------------------------"
  echo "Running model: $model"
  echo "Started at   : $(date '+%Y-%m-%d %H:%M:%S')"
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
  echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
  echo

  # Small separator / cooling-off point between providers.
  sleep 1
done

echo "============================================================"
echo "Experiment finished."
echo
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
