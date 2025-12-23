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
  "openrouter:mistralai/mistral-small-3.2-24b-instruct"
  "openrouter:meta-llama/llama-3.1-70b-instruct"
  "openrouter:cohere/command-r-plus-08-2024"
  "openrouter:mistralai/mixtral-8x22b-instruct"
  "openrouter:deepseek/deepseek-chat-v3.1"
)

# RAG runs: recommended only for balanced
DATASETS=("balanced")

for ds in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    echo "=== DATASET=$ds LIMIT=all RAG=yes MODEL=$model REPEATS=5 ==="
    "$VENV_PY" src/llm_zero_shot.py \
      --dataset "$ds" \
      --limit all \
      --rag \
      --model "$model" \
      --repeats 5 \
      --seed 42
  done
done

# If you REALLY want to run RAG also on invented, uncomment below:
#
# DATASETS=("balanced" "invented")
# for ds in "${DATASETS[@]}"; do
#   for model in "${MODELS[@]}"; do
#     echo "=== DATASET=$ds LIMIT=all RAG=yes MODEL=$model REPEATS=5 ==="
#     "$VENV_PY" src/llm_zero_shot.py \
#       --dataset "$ds" \
#       --limit all \
#       --rag \
#       --model "$model" \
#       --repeats 5 \
#       --seed 42
#   done
# done
