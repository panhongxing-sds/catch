"""CATCH leave-one-dataset-out calibration and paper metrics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .io import read_jsonl
from .scoring import (
    component_features,
    fit_calibrator,
    fit_leave_one_dataset_out,
    training_free_variant,
)


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.shape != scores.shape or labels.ndim != 1 or not len(labels):
        raise ValueError(
            "Metrics require equally sized nonempty label and score vectors"
        )
    if not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Scores must be finite and incorrect labels must be binary")
    if len(np.unique(labels)) != 2:
        raise ValueError(
            "AUROC requires both correct and incorrect answers in the evaluated cohort"
        )
    order = np.argsort(scores, kind="stable")
    retained = order[: max(1, len(order) // 2)]
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "selective_accuracy_50": float(np.mean(labels[retained] == 0)),
    }


def evaluate_model_run(
    results_root: str | Path,
    model_key: str,
    evaluation_run: int,
    datasets: tuple[str, ...] = ("minerva", "math500", "gsm8k"),
    *,
    mode: str = "both",
    variant: str = "full",
    top_m: int = 10,
) -> dict:
    run_root = Path(results_root) / model_key / f"run_{evaluation_run}"
    rows: list[dict] = []
    for dataset in datasets:
        shard = read_jsonl(run_root / f"{dataset}.jsonl")
        if not shard or any(row["dataset"] != dataset for row in shard):
            raise ValueError(f"Empty or mislabeled dataset shard: {dataset}")
        rows.extend(shard)
    if mode not in {"both", "supervised", "training_free"}:
        raise ValueError(f"Unknown scoring mode: {mode}")
    if len({(row["dataset"], row["id"]) for row in rows}) != len(rows):
        raise ValueError(
            "Duplicate model-question records would change the evaluation cohort"
        )
    if mode != "training_free":
        for row in rows:
            row["answer_shift"], row["contextual_hesitation"] = component_features(
                row, variant, top_m
            )
    calibrators = fit_leave_one_dataset_out(rows) if mode != "training_free" else {}
    output: dict[str, dict] = {}
    scored_rows: list[dict] = []
    for dataset in datasets:
        fold = [row for row in rows if row["dataset"] == dataset]
        labels = np.asarray([row["incorrect"] for row in fold])
        output[dataset] = {"n": len(fold), "error_rate": float(labels.mean())}
        fold_scores = {}
        if mode != "training_free":
            features = np.asarray(
                [[row["answer_shift"], row["contextual_hesitation"]] for row in fold]
            )
            fold_scores["supervised"] = calibrators[dataset].predict_unreliability(
                features
            )
        if mode != "supervised":
            fold_scores["training_free"] = np.asarray(
                [training_free_variant(row, variant, top_m) for row in fold]
            )
        for name, scores in fold_scores.items():
            output[dataset][name] = metrics(labels, scores)
        for index, row in enumerate(fold):
            scored_rows.append(
                {
                    "id": row["id"],
                    "dataset": dataset,
                    "incorrect": int(labels[index]),
                    **{
                        name: float(scores[index])
                        for name, scores in fold_scores.items()
                    },
                }
            )
    stem = f"{mode}_{variant}_M{top_m}"
    (run_root / f"{stem}_scores.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            for row in scored_rows
        ),
        encoding="utf-8",
    )
    (run_root / f"{stem}_metrics.json").write_text(
        json.dumps(output, indent=2, allow_nan=False), encoding="utf-8"
    )
    calibration_audit = {
        name: {
            "training_datasets": [d for d in datasets if d != name],
            "feature_medians": c.feature_medians.tolist(),
            "feature_means": c.feature_means.tolist(),
            "feature_scales": c.feature_scales.tolist(),
            "coefficients": c.classifier.coef_.tolist(),
            "intercept": c.classifier.intercept_.tolist(),
        }
        for name, c in calibrators.items()
    }
    if calibration_audit:
        (run_root / f"{stem}_calibration.json").write_text(
            json.dumps(calibration_audit, indent=2), encoding="utf-8"
        )
    return output


def aggregate_runs(
    results_root: str | Path,
    model_key: str,
    evaluation_runs: tuple[int, ...] = (41, 42, 43),
    datasets: tuple[str, ...] = ("minerva", "math500", "gsm8k"),
    *,
    mode: str = "both",
    variant: str = "full",
    top_m: int = 10,
) -> dict:
    if len(evaluation_runs) < 2:
        raise ValueError("Mean/std reporting requires at least two evaluation runs")
    per_run = [
        evaluate_model_run(
            results_root,
            model_key,
            run,
            datasets,
            mode=mode,
            variant=variant,
            top_m=top_m,
        )
        for run in evaluation_runs
    ]
    summary: dict[str, dict] = {}
    for dataset in datasets:
        summary[dataset] = {}
        for score_mode in ("supervised", "training_free"):
            if score_mode not in per_run[0][dataset]:
                continue
            summary[dataset][score_mode] = {}
            for metric_name in ("auroc", "auprc", "selective_accuracy_50"):
                values = np.asarray(
                    [run[dataset][score_mode][metric_name] for run in per_run]
                )
                summary[dataset][score_mode][metric_name] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                }
    destination = (
        Path(results_root) / model_key / f"{mode}_{variant}_M{top_m}_mean_std.json"
    )
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def evaluate_external(
    manifest_path: str,
    output_path: str,
    variant: str = "full",
    top_m: int = 10,
    mode: str = "both",
) -> dict:
    """Explicit calibration source for additional datasets; never guess a split."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if mode not in {"both", "supervised", "training_free"}:
        raise ValueError(f"Unknown scoring mode: {mode}")
    root = Path(manifest_path).resolve().parent

    def load_group(name):
        rows, hashes = [], {}
        for filename in manifest.get(name, []):
            path = root / filename
            hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.extend(read_jsonl(path))
        keys = [(str(row["dataset"]), str(row["id"])) for row in rows]
        if len(set(keys)) != len(keys):
            raise ValueError(f"Duplicate IDs in {name} records")
        return rows, hashes, set(keys)

    test, test_hashes, test_keys = load_group("test")
    if not test:
        raise ValueError("Manifest must specify nonempty test files")
    labels = np.asarray([row["incorrect"] for row in test])
    scores, audit = (
        {},
        {
            "mode": mode,
            "variant": variant,
            "top_m": top_m,
            "test_files_sha256": test_hashes,
            "test_n": len(test),
        },
    )
    if mode != "training_free":
        train, train_hashes, train_keys = load_group("train")
        if train_keys & test_keys:
            raise ValueError("Calibration/test question IDs overlap")
        train_questions = {
            row["question"].strip() for row in train if row.get("question")
        }
        if any(row.get("question", "").strip() in train_questions for row in test):
            raise ValueError("Calibration/test questions overlap under different IDs")
        for row in train:
            row["answer_shift"], row["contextual_hesitation"] = component_features(
                row, variant, top_m
            )
        calibrator = fit_calibrator(train)
        scores["supervised"] = calibrator.predict_unreliability(
            np.asarray([component_features(row, variant, top_m) for row in test])
        )
        audit.update(
            train_files_sha256=train_hashes,
            train_n=len(train),
            feature_medians=calibrator.feature_medians.tolist(),
            feature_means=calibrator.feature_means.tolist(),
            feature_scales=calibrator.feature_scales.tolist(),
            coefficients=calibrator.classifier.coef_.tolist(),
            intercept=calibrator.classifier.intercept_.tolist(),
        )
    if mode != "supervised":
        scores["training_free"] = np.asarray(
            [training_free_variant(row, variant, top_m) for row in test]
        )
    output = {name: metrics(labels, values) for name, values in scores.items()}
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"metrics": output, "audit": audit}, indent=2), encoding="utf-8"
    )
    destination.with_suffix(".scores.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "id": row["id"],
                    "dataset": row["dataset"],
                    "incorrect": int(labels[i]),
                    **{name: float(values[i]) for name, values in scores.items()},
                }
            )
            + "\n"
            for i, row in enumerate(test)
        ),
        encoding="utf-8",
    )
    return output
