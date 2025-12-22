#!/usr/bin/env bash
set -euo pipefail

VENV_PY="$(pwd)/.venv/Scripts/python.exe"

MODELS=(
  # OpenAI
  "gpt-4o"
  "gpt-5.1"
  "gpt-4o-mini"
  "gpt-4-0613"

  # Qwen (DashScope)
  "qwen:qwen3-8b"
  "qwen:qwen-turbo"
  "qwen:qwen-plus"
  "qwen:qwen-max"

  # OpenRouter
  "openrouter:meta-llama/llama-3.1-8b-instruct"
  "openrouter:google/gemma-2-9b-it"
  "openrouter:mistralai/mistral-small-3.2-24b-instruct"
  "openrouter:meta-llama/llama-3.1-70b-instruct"
)

DATASETS=("balanced" "invented")

for ds in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    echo "=== DATASET=$ds LIMIT=all RAG=no MODEL=$model REPEATS=5 ==="
    "$VENV_PY" src/llm_zero_shot.py \
      --dataset "$ds" \
      --limit all \
      --model "$model" \
      --repeats 5 \
      --seed 42
  done
done
