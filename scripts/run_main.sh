#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${CONFIG:-configs/main.yaml}"

.venv/bin/catch-uq validate-data --config "$CONFIG"
for model in qwen25_3b llama32_1b llama31_8b qwen3_8b; do
  for evaluation_run in 41 42 43; do
    for dataset in minerva math500 gsm8k; do
      .venv/bin/catch-uq run \
        --config "$CONFIG" \
        --model "$model" \
        --dataset "$dataset" \
        --evaluation-run "$evaluation_run"
    done
  done
  .venv/bin/catch-uq evaluate --config "$CONFIG" --results-root results --model "$model"
done
