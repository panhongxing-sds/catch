# CATCH

CATCH (Complementary Perturbation-attended Consistency with Contextual
Hesitation) estimates uncertainty in mathematical reasoning by large language
models. This repository contains only the CATCH method: text perturbations,
weight perturbations, Answer Shift, Contextual Hesitation, fusion calibration,
the main experiment, and CATCH hyperparameter sensitivity experiments. It does
not contain baseline implementations, sampling pools, historical analyses, or
data records.

## Paper configuration

- Datasets: Minerva 272, MATH-500 500, and GSM8K 1,319; 8,364 instances across
  four models.
- Models: Qwen2.5-3B, Llama-3.2-1B, Llama-3.1-8B, and Qwen3-8B.
- Nine greedy generations per question: one base generation, four controlled
  text perturbations, and four low-rank weight perturbations.
- Weight perturbations: every `q_proj` and `k_proj`, `rank=4`, `sigma=0.03`.
- Evaluation runs: 41, 42, and 43; weight seeds 42-45, 46-49, and 50-53.
- Decoding: `T=0`, at most 2,048 new tokens, bfloat16, and Hugging Face
  Transformers.
- Contextual Hesitation: renormalized top-10 probability entropy before the
  answer span, linearly weighted by proximity to the answer.
- Answer Shift and Contextual Hesitation are standardized within each training
  fold and fused using leave-one-dataset-out calibration over three datasets.
- Metrics: AUROC, AUPRC, and selective accuracy at 50% coverage.

See [APPENDIX_MAPPING.md](APPENDIX_MAPPING.md) for the itemized correspondence.
The unified configuration is in [`configs/main.yaml`](configs/main.yaml).
Code-level validation runs before each experiment and rejects changes that
conflict with the paper-locked configuration. Validate it without data:

```bash
.venv/bin/catch-uq validate-config --config configs/main.yaml
```

## 1. Installation

```bash
bash scripts/setup.sh
```

Python 3.10+, CUDA, and sufficient GPU memory for the selected model are
required. All paths are resolved from the project root.

## 2. Data

This repository contains no data records. Before running an experiment, supply
the three fixed snapshots used by the paper:

```text
data/benchmarks/minerva.jsonl   # 272 rows
data/benchmarks/math500.jsonl   # 500 rows
data/benchmarks/gsm8k.jsonl     # 1319 rows
```

Each line must use this schema:

```json
{"id":"math500_0","question":"...","answer":"..."}
```

Validate the exact row counts:

```bash
.venv/bin/catch-uq validate-data --config configs/main.yaml
```

## 3. Generate CATCH text perturbations

```bash
export DEEPSEEK_API_KEY='...'
bash scripts/generate_variants.sh
```

The pipeline creates one candidate from each of the four transformation
families and verifies each candidate in a separate thinking-mode call. It
rejects and resamples candidates that change a numeric, unit, or operator
anchor, duplicate another candidate, or do not receive `EQUIVALENT` from the
verifier. Family instructions, bounded retries, and network timeouts are
implementation details that do not alter the method. The base generation
prompt, verification prompt, model, temperature, and top-p remain identical to
the appendix. The API key is read only from the environment and is never
written to a file.

## 4. Run CATCH

Run a one-record smoke test:

```bash
.venv/bin/catch-uq run --model qwen25_3b --dataset math500 \
  --evaluation-run 41 --limit 1
```

Run the complete main experiment:

```bash
bash scripts/run_main.sh
```

Results are appended record by record, and reruns skip completed IDs. After all
three runs finish, the script performs leave-one-dataset-out calibration and
reports CATCH scores and mean plus or minus sample-standard-deviation metrics.

## 5. Run CATCH sensitivity experiments

Following Appendix C.2, the sensitivity experiment uses four weight seeds and
the two Llama models. It generates features for all three main datasets under
each setting so that the same leave-one-dataset-out calibration protocol can be
applied, then reads the corresponding metrics for the 272 Minerva records:

```bash
bash scripts/run_sensitivity.sh
```

The script covers `sigma={0.01,0.03,0.05,0.07,0.10,0.15}` and
`rank={1,2,4,8,16}`.

## 6. Tests

```bash
.venv/bin/pytest -q
```

The generic copyright notice required by the MIT License is retained in
`LICENSE`.
