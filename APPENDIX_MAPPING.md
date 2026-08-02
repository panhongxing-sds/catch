# CATCH appendix mapping

| Paper location | CATCH setting | Configuration or implementation |
|---|---|---|
| A.2 / Table 3 | R=4, W=4, base+8=9 | `configs/main.yaml`, `runner.py` |
| A.2 / Table 3 | rank=4, sigma=0.03, Q/K in all layers | `perturbations.py` |
| A.2 / Table 3 | M=10, maximum tokens=2048, bfloat16 | `generation.py`, `scoring.py` |
| A.2 / Table 3 | evaluation runs 41/42/43 | `configs/main.yaml` |
| A.2 / Table 3 | weight groups 42-45 / 46-49 / 50-53 | `configs/main.yaml`, `runner.py` |
| A.2.1 | four controlled text transformations | `prompts.py`, `variants.py` |
| A.2.2 | SVD low-rank Gaussian update with sigma/sqrt(d) | `perturbations.py` |
| A.2.3 / Equations 6-7 | pre-answer top-M entropy with proximity weighting | `scoring.py` |
| A.2.4 / Algorithm 1 | training-fold median, standardization, and LODO logistic fusion | `scoring.py`, `evaluate.py` |
| A.3.1 / Table 4 | Qwen boxed prompt and Llama step-by-step prompt | `prompts.py` |
| A.3.2 | DeepSeek-V4-Pro generation and independent verification | `variants.py` |
| A.4 | boxed-first extraction and answer normalization | `answers.py` |
| C.2 | sigma and rank sensitivity grids | `scripts/run_sensitivity.sh` |
| D.5.1 | 272/500/1319 x 4 = 8364 | `validate-data`, `configs/main.yaml` |
| D.5.2 | nine greedy CATCH decodes and held-out calibration | `runner.py`, `evaluate.py` |

This project contains no baseline implementation or experiment entry point for
another uncertainty method. Before every run, `method_config.py` locks the main
experimental settings listed above. Engineering details such as retry logic,
timeouts, resumable execution, and output recording improve reproducibility
without changing the mathematical definition or experimental budget of CATCH.
