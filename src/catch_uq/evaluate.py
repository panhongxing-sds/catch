"""CATCH leave-one-dataset-out calibration and paper metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .io import read_jsonl
from .scoring import fit_leave_one_dataset_out


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    order = np.argsort(scores)
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
) -> dict:
    run_root = Path(results_root) / model_key / f"run_{evaluation_run}"
    rows: list[dict] = []
    for dataset in datasets:
        rows.extend(read_jsonl(run_root / f"{dataset}.jsonl"))
    calibrators = fit_leave_one_dataset_out(rows)
    output: dict[str, dict] = {}
    scored_rows: list[dict] = []
    for dataset, calibrator in calibrators.items():
        fold = [row for row in rows if row["dataset"] == dataset]
        features = np.asarray(
            [[row.get("answer_shift", np.nan), row.get("contextual_hesitation", np.nan)] for row in fold]
        )
        scores = calibrator.predict_unreliability(features)
        labels = np.asarray([row["incorrect"] for row in fold], dtype=int)
        output[dataset] = metrics(labels, scores)
        scored_rows.extend(
            {"id": row["id"], "dataset": dataset, "catch_score": float(score)}
            for row, score in zip(fold, scores)
        )
    (run_root / "catch_scores.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scored_rows), encoding="utf-8"
    )
    (run_root / "metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def aggregate_runs(
    results_root: str | Path,
    model_key: str,
    evaluation_runs: tuple[int, ...] = (41, 42, 43),
    datasets: tuple[str, ...] = ("minerva", "math500", "gsm8k"),
) -> dict:
    per_run = [evaluate_model_run(results_root, model_key, run, datasets) for run in evaluation_runs]
    summary: dict[str, dict] = {}
    for dataset in datasets:
        summary[dataset] = {}
        for metric_name in ("auroc", "auprc", "selective_accuracy_50"):
            values = np.asarray([run[dataset][metric_name] for run in per_run])
            summary[dataset][metric_name] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
            }
    destination = Path(results_root) / model_key / "metrics_mean_std.json"
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
