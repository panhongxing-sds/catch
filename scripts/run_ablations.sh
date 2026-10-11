#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
for model in qwen25_3b llama32_1b llama31_8b qwen3_8b; do
  for variant in full V1 V2 V3 V4 V5; do
    .venv/bin/catch-uq evaluate --model "$model" --mode both --variant "$variant"
  done
done
