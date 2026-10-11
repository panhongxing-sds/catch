# CATCH

Reference implementation of **Complementary Perturbation-attended Consistency
with Contextual Hesitation**, updated for the paper's supervised and training-free
algorithms. The training-free implementation uses **Hesitation-Gated Consistency
Estimation**, rather than adding two question-level signals.

## Method

Each question receives one greedy base response, four verified text perturbation
responses, and four weight perturbation responses. Top-M renormalized token
entropy is measured before the final-answer span, with linear proximity weights:

```text
H_j = sum_t ((t+1)/alpha_j) h_jt / sum_t ((t+1)/alpha_j)
c_j = 1 - H_j / log(M)
S_TF = 1 - (1/K) sum_j 1[a_j equivalent to a_0] c_j
```

K=8 and M=10 in the main experiment. The base trace anchors a_0 and is excluded
from support aggregation. Invalid answers, absent pre-answer tokens, and missing
required probabilities supply zero support and remain in K. Commitment is a
proxy for low hesitation, not a correctness probability. S_TF is used for
unreliability ranking, not as a calibrated error probability.

The supervised algorithm retains Answer Shift and mean Contextual Hesitation.
Missing-value medians, standardization, and logistic fusion are fitted on the
non-held-out datasets only. Both algorithms reuse the same cached responses.

Weight updates use `U_r diag(epsilon) V_r^T`, with
`epsilon ~ Normal(0, sigma^2/d)`, rank=4, sigma=0.03, and all Q/K projections.
Each model instance is fixed during one generation. Original weights are
restored exactly before the next seed; updates never accumulate across runs.
Entropy comes from raw HF prediction logits, including for sampled decoding.

## Installation and checks

```bash
bash scripts/setup.sh
.venv/bin/catch-uq validate-config
.venv/bin/pytest -q
```

Python 3.10+ is required. Full inference needs enough accelerator memory for the
selected model. Unit tests and the tiny randomly initialized HF integration test
run on CPU, without pretrained model downloads or API calls.

## Main experiment data

Provide the exact evaluated snapshots as JSONL:

```text
data/benchmarks/minerva.jsonl  # 272 questions
data/benchmarks/math500.jsonl  # 500 questions
data/benchmarks/gsm8k.jsonl    # 1319 questions
```

Each row contains a stable unique ID, the complete question presented to the
model, and its reference answer or solution:

```json
{"id":"math500_0","question":"...","answer":"..."}
```

```bash
.venv/bin/catch-uq validate-data
```

The main models are Qwen2.5-3B, Llama-3.2-1B, Llama-3.1-8B, and Qwen3-8B,
covering 8,364 model-question pairs. Runs 41/42/43 use weight seed groups
42–45 / 46–49 / 50–53. Generation is greedy, bfloat16, with at most 2,048
new tokens. Configuration is in `configs/main.yaml`.

**Artifact scope:** this release provides executable method/evaluation code,
not the original benchmark snapshots, verified variant caches, model outputs,
baseline implementations, or published-result dumps. Numerical results are not
hardcoded. Exact table reproduction requires the original snapshots, variants,
checkpoints, grading rules, and environment. Code validation does not imply
that the full paper experiments have been rerun.

## Text perturbations

```bash
export DEEPSEEK_API_KEY='...'
export OPENAI_API_KEY='...'
bash scripts/generate_variants.sh
```

Four families are generated: rewording, role-preserving substitution, condition
reordering, and irrelevant distractor insertion. DeepSeek-V4-Pro uses temperature
0.8 and top-p 0.95. A separate GPT-5.6 Responses API call verifies semantic
preservation; only an explicit `EQUIVALENT` verdict is accepted. Numerical/unit/
operator checks and duplicate rejection precede verification. API keys stay in
environment variables. Endpoint overrides are `DEEPSEEK_BASE_URL` and
`OPENAI_BASE_URL`. Candidate checks do not guarantee semantic equivalence.

For the open-source generator ablation, keep the verifier and change the generator:

```bash
.venv/bin/catch-uq generate-variants --dataset math500 \
  --input data/benchmarks/math500.jsonl \
  --output data/variants/qwen3_8b_math500.jsonl \
  --generator-model Qwen/Qwen3-8B --generator-seed 42
```

Use `meta-llama/Meta-Llama-3.1-8B-Instruct` for the other generator. These
commands require API/model access and can incur service charges when run.

## Generation and scoring

```bash
# One-question smoke run stored separately from the main experiment.
.venv/bin/catch-uq run --model qwen25_3b --dataset math500 \
  --evaluation-run 41 --limit 1 --output-tag smoke

# Main model/dataset/run shards, followed by both scoring algorithms.
bash scripts/run_main.sh

# Label-free scoring of cached perturbation trajectories.
.venv/bin/catch-uq score --input results/qwen25_3b/run_41/math500.jsonl \
  --output results/qwen25_3b/run_41/training_free.jsonl

# Evaluate either or both algorithms.
.venv/bin/catch-uq evaluate --model qwen25_3b --mode both
.venv/bin/catch-uq evaluate --model qwen25_3b --mode training_free
```

Generation records include responses, answers, raw top-M log probabilities,
answer-span offsets, seeds, and component scores. Run manifests record input,
configuration, implementation hashes, and package versions. Resume rejects
incompatible caches. Variant files record their generator/verifier configuration.
Old caches without manifests can be scored, but generation uses a fresh output tag.

Evaluation writes per-question scores, metrics, and supervised calibration audits.
AUROC/AUPRC treat incorrect answers as positive. Acc* retains the lowest-scoring
50%, resolving ties in stable input order. Metrics are stored as fractions;
multiply by 100 for percentages. Run summaries use sample standard deviation.

## Sensitivity and ablation

```bash
bash scripts/run_sensitivity.sh
bash scripts/run_ablations.sh
```

Sensitivity covers sigma={0.01,0.03,0.05,0.07,0.10,0.15}, rank={1,2,4,8,16},
and M={2,4,8,10}. Top-M changes are rescored from stored top-10 distributions.
Supervised fusion is refitted using the same LODO rule for each setting.

V1 removes agreement/Answer Shift; V2 removes hesitation; V3 retains weight
perturbations; V4 retains text perturbations; V5 uses full-trace mean top-M
entropy. Supervised and training-free ablations are reported separately. V3/V4
adjust K to the retained trajectories. Unsupported trajectories supply zero
training-free support in all variants, including V1/V2.

Perturbation targets support `QK`, `VO`, `MLP`, and `QKVO`:

```bash
.venv/bin/catch-uq run --model llama31_8b --dataset math500 \
  --evaluation-run 41 --target VO --output-tag additional/target_VO
```

Generator-specific caches use `--variants` and a distinct output tag such as
`additional/generator_qwen3`. Additional studies remain separate from main runs.

## Larger models and additional datasets

Pass an explicit target model ID and additional data/variant paths:

```bash
.venv/bin/catch-uq run --model qwen3_32b --hf-model Qwen/Qwen3-32B \
  --dataset gsm8k --evaluation-run 41 --output-tag additional/qwen3_32b

.venv/bin/catch-uq run --model qwen3_8b --dataset hotpotqa \
  --input data/benchmarks/hotpotqa.jsonl --variants data/variants/hotpotqa.jsonl \
  --evaluation-run 41 --output-tag additional/hotpotqa
```

The `question` field must include the complete actual model input, including
context when the protocol uses context. For datasets requiring a different answer
judge, attach the protocol's binary `incorrect` labels to generated records
before evaluation. Bundled symbolic/numeric normalization targets mathematical
experiments; it is not an official HotpotQA grader.

Additional supervised evaluation requires an explicit train/test manifest.
The implementation does not guess the source of logistic fusion parameters:

```json
{"train":["calibration_minerva.jsonl","calibration_math500.jsonl"],
 "test":["gsm8k.jsonl"]}
```

Paths resolve relative to the manifest; files use the runner's trajectory schema.
The evaluator rejects overlapping IDs and identical questions across train/test,
fits all statistics on training data only, and records input hashes:

```bash
.venv/bin/catch-uq evaluate-external --manifest study.json \
  --output results/study_metrics.json --mode both
```

With `--mode training_free`, only `test` is needed. Labels affect metrics, not
training-free scoring. See `APPENDIX_MAPPING.md` for the paper/code mapping.
