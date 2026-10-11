# Paper-to-code mapping

| Paper item | Implementation |
|---|---|
| R=W=4 complementary perturbations | `runner.py`, `configs/main.yaml` |
| All-layer Q/K SVD update, rank=4, sigma=0.03 | `perturbations.py` |
| Independent instances restored between seeds | `LowRankGaussianPerturber.sampled` |
| Eq. 5: Answer Shift | `scoring.answer_shift` |
| Eqs. 6–7: normalized top-M proximity-weighted entropy | `local_topm_entropy`, `run_hesitation`, `contextual_hesitation` |
| Eq. 8 / Algorithm 2: hesitation-gated score | `training_free_score`, `trajectory_commitment` |
| Algorithm 1: training-fold median, normalization, LODO logistic fusion | `fit_leave_one_dataset_out`, `fit_calibrator` |
| Invalid TF trajectories supply zero support | `trajectory_commitment`; no imputation |
| A.3: generation/verifier prompts | `prompts.py`, `variants.py` |
| Separate DeepSeek-V4-Pro generator and GPT-5.6 verifier | `api.py`, `configs/main.yaml` |
| Answer-span extraction | `answers.py`, `generation.py` |
| Main cohorts, models, seed groups | `method_config.py`, `configs/main.yaml` |
| V1–V5 supervised and TF ablations | `component_features`, `training_free_variant`, `scripts/run_ablations.sh` |
| B.1: sigma/rank/top-M sensitivity | `scripts/run_sensitivity.sh` |
| C.4: larger models and extra datasets | `run --hf-model/--input/--variants` |
| C.4: open-source generator comparison | `generate-variants --generator-model` |
| C.4: Q/K, V/O, MLP, Q/K/V/O targets | `run --target` |
| Explicit extra-study fusion training source | `evaluate-external`, split overlap checks, calibration audits |

Main settings are checked before execution; extra studies require separate output
tags. This repository releases CATCH method and evaluation code. It does not
include original dataset snapshots, baseline implementations, SC sampling pools,
historical analysis outputs, or published-result dumps. See README for artifact scope.
