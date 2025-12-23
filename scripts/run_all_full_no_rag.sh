#!/usr/bin/env bash
set -euo pipefail

VENV_PY="$(pwd)/.venv/Scripts/python.exe"

MODELS=(
  # OpenAI
  "gpt-4-0613"
  "gpt-4o"
  "gpt-5.1"
  "gpt-4o-mini"

  # Qwen (DashScope)
  "qwen:qwen3-8b"
  "qwen:qwen-turbo"
  "qwen:qwen-plus"
  "qwen:qwen-max"

  # OpenRouter
  "openrouter:meta-llama/llama-3.1-8b-instruct"
  "openrouter:meta-llama/llama-3.1-70b-instruct"
  "openrouter:mistralai/mistral-small-3.2-24b-instruct"
  "openrouter:mistralai/mixtral-8x22b-instruct"
  "openrouter:cohere/command-r-plus-08-2024"
  "openrouter:deepseek/deepseek-chat-v3.1"
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
