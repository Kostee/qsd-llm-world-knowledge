#!/usr/bin/env bash
set -euo pipefail

# Edit this list as needed (these are "Maciek-style" model IDs).
MODELS=(
  "gpt-4o"
  "qwen:qwen3-8b"
  "gemini:gemini-2.0-flash-exp"
  "openrouter:meta-llama/llama-3.1-8b-instruct"
)

DATASETS=("balanced" "invented")

for ds in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    echo "=== DATASET=$ds LIMIT=all RAG=no MODEL=$model REPEATS=5 ==="
    python3 ./src/llm_zero_shot.py \
      --dataset "$ds" \
      --limit all \
      --model "$model" \
      --repeats 5 \
      --seed 42
  done
done
