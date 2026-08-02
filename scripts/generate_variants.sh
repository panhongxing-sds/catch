#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
: "${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY in the environment; do not put it in a file.}"

for dataset in minerva math500 gsm8k; do
  .venv/bin/catch-uq generate-variants \
    --config configs/main.yaml \
    --dataset "$dataset" \
    --input "data/benchmarks/${dataset}.jsonl" \
    --output "data/variants/${dataset}.jsonl"
done
