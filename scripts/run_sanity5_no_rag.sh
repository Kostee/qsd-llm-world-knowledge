#!/usr/bin/env bash
set -euo pipefail

VENV_PY="$(pwd)/.venv/Scripts/python.exe"

MODELS=(
  "gpt-4o"
  "gpt-5"
  "gpt-4-0613"
  "qwen:qwen3-8b"
  "gemini:gemini-2.0-flash-exp"
  "openrouter:meta-llama/llama-3.1-8b-instruct"
)

DATASETS=("balanced" "invented")

for ds in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    echo "=== DATASET=$ds LIMIT=5 RAG=no MODEL=$model REPEATS=5 ==="
    "$VENV_PY" src/llm_zero_shot.py \
      --dataset "$ds" \
      --limit 5 \
      --model "$model" \
      --repeats 5 \
      --seed 42
  done
done