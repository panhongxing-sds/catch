#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

for model in llama32_1b llama31_8b; do
  for sigma in 0.01 0.03 0.05 0.07 0.10 0.15; do
    for dataset in minerva math500 gsm8k; do
      .venv/bin/catch-uq run --dataset "$dataset" --model "$model" --evaluation-run 41 \
        --weight-rank 4 --weight-sigma "$sigma" --output-tag "sensitivity/sigma_${sigma}"
    done
    .venv/bin/catch-uq evaluate-run --config configs/main.yaml \
      --results-root "results/sensitivity/sigma_${sigma}" \
      --model "$model" --evaluation-run 41
  done
  for rank in 1 2 4 8 16; do
    for dataset in minerva math500 gsm8k; do
      .venv/bin/catch-uq run --dataset "$dataset" --model "$model" --evaluation-run 41 \
        --weight-rank "$rank" --weight-sigma 0.03 --output-tag "sensitivity/rank_${rank}"
    done
    .venv/bin/catch-uq evaluate-run --config configs/main.yaml \
      --results-root "results/sensitivity/rank_${rank}" \
      --model "$model" --evaluation-run 41
  done
done
