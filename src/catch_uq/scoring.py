"""CATCH component scores and leave-one-dataset-out calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .answers import answers_equivalent


def answer_shift(base_answer: str, perturbation_answers: list[str]) -> float:
    """Equation 5: fraction of perturbation answers different from the base."""
    if not perturbation_answers:
        return float("nan")
    return float(
        np.mean(
            [
                not answers_equivalent(answer, base_answer)
                for answer in perturbation_answers
            ]
        )
    )


def local_topm_entropy(top_logprobs: list[float]) -> float:
    if not top_logprobs:
        return float("nan")
    values = np.asarray(top_logprobs, dtype=float)
    if (
        np.isnan(values).any()
        or np.isposinf(values).any()
        or not np.isfinite(values).any()
    ):
        return float("nan")
    values = values - np.max(values)
    probabilities = np.exp(values)
    probabilities /= probabilities.sum()
    return float(-np.sum(probabilities * np.log(np.clip(probabilities, 1e-30, None))))


def run_hesitation(
    token_trace: list[dict], answer_start_token: int | None, top_m: int = 10
) -> float:
    """Equation 7 for one response, restricted to tokens before the answer span."""
    if top_m < 2:
        raise ValueError("The fixed top-M size must be at least 2")
    if (
        answer_start_token is None
        or answer_start_token <= 0
        or answer_start_token > len(token_trace)
    ):
        return float("nan")
    alpha = answer_start_token
    entropies: list[float] = []
    for token in token_trace[:alpha]:
        values = token.get("top_logprobs") or token.get("topk_logprobs") or []
        if len(values) < top_m:
            return float("nan")
        entropies.append(local_topm_entropy(list(values)[:top_m]))
    if not entropies or any(not math.isfinite(x) for x in entropies):
        return float("nan")
    weights = np.arange(1, alpha + 1, dtype=float) / alpha
    return float(np.sum(weights * np.asarray(entropies)) / np.sum(weights))


def contextual_hesitation(runs: list[dict], top_m: int = 10) -> float:
    """Equation 7 over all perturbation runs; missing spans remain missing for fold imputation."""
    if not runs:
        return float("nan")
    values = [
        run_hesitation(run.get("token_trace", []), run.get("answer_start_token"), top_m)
        for run in runs
    ]
    if any(not math.isfinite(value) for value in values):
        return float("nan")
    return float(np.mean(values))


def full_trace_hesitation(token_trace: list[dict], top_m: int = 10) -> float:
    """V5: mean of the same renormalized top-M entropy over the full trace."""
    if top_m < 2:
        raise ValueError("The fixed top-M size must be at least 2")
    values = []
    for token in token_trace:
        logprobs = token.get("top_logprobs") or token.get("topk_logprobs") or []
        if len(logprobs) < top_m:
            return float("nan")
        values.append(local_topm_entropy(logprobs[:top_m]))
    return (
        float(np.mean(values))
        if values and all(map(math.isfinite, values))
        else float("nan")
    )


def trajectory_commitment(
    run: dict, top_m: int = 10, *, full_trace: bool = False
) -> float:
    """Algorithm 2: unsupported trajectories have c=0, without imputation."""
    if top_m < 2:
        raise ValueError("The fixed top-M size must be at least 2")
    alpha = run.get("answer_start_token")
    if not run.get("answer") or alpha is None or alpha < 1:
        return 0.0
    trace = run.get("token_trace", [])
    if alpha > len(trace):
        return 0.0
    h = (
        full_trace_hesitation(trace, top_m)
        if full_trace
        else run_hesitation(trace, alpha, top_m)
    )
    return float(np.clip(1 - h / math.log(top_m), 0, 1)) if math.isfinite(h) else 0.0


def training_free_score(
    base_answer: str,
    runs: list[dict],
    top_m: int = 10,
    *,
    remove_agreement: bool = False,
    remove_hesitation: bool = False,
    full_trace: bool = False,
) -> float:
    """Equation 8: one minus agreement-weighted commitment, including invalid runs in K.

    V1 removes agreement. V2 sets c=1 for valid trajectories. V5 changes
    the entropy span. Invalid trajectories still supply zero support.
    """
    if top_m < 2 or not runs:
        raise ValueError(
            "Training-free scoring requires M>=2 and at least one perturbation run"
        )
    support = 0.0
    for run in runs:
        c = trajectory_commitment(run, top_m, full_trace=full_trace)
        if remove_hesitation:
            h = run_hesitation(
                run.get("token_trace", []), run.get("answer_start_token"), top_m
            )
            c = float(bool(run.get("answer")) and math.isfinite(h))
        if remove_agreement or answers_equivalent(run.get("answer", ""), base_answer):
            support += c
    return float(np.clip(1 - support / len(runs), 0, 1))


def component_features(
    row: dict, variant: str = "full", top_m: int = 10
) -> list[float]:
    """Recompute supervised V1--V5 components from a shared trajectory record."""
    if variant not in {"full", "V1", "V2", "V3", "V4", "V5"}:
        raise ValueError(f"Unknown ablation: {variant}")
    runs = row.get("text_runs", []) + row.get("weight_runs", [])
    if variant == "V3":
        runs = row.get("weight_runs", [])
    elif variant == "V4":
        runs = row.get("text_runs", [])
    shift = answer_shift(row["base"]["answer"], [run.get("answer", "") for run in runs])
    if variant == "V5":
        h = [full_trace_hesitation(run.get("token_trace", []), top_m) for run in runs]
        hesitation = (
            float(np.mean(h)) if h and all(map(math.isfinite, h)) else float("nan")
        )
    else:
        hesitation = contextual_hesitation(runs, top_m)
    return [0.0 if variant == "V1" else shift, 0.0 if variant == "V2" else hesitation]


def training_free_variant(row: dict, variant: str = "full", top_m: int = 10) -> float:
    if variant not in {"full", "V1", "V2", "V3", "V4", "V5"}:
        raise ValueError(f"Unknown ablation: {variant}")
    runs = row.get("text_runs", []) + row.get("weight_runs", [])
    if variant == "V3":
        runs = row.get("weight_runs", [])
    elif variant == "V4":
        runs = row.get("text_runs", [])
    return training_free_score(
        row["base"]["answer"],
        runs,
        top_m,
        remove_agreement=variant == "V1",
        remove_hesitation=variant == "V2",
        full_trace=variant == "V5",
    )


@dataclass
class FoldCalibrator:
    held_out_dataset: str
    feature_medians: np.ndarray
    feature_means: np.ndarray
    feature_scales: np.ndarray
    classifier: object

    def transform(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=float).copy()
        missing = ~np.isfinite(x)
        x[missing] = np.take(self.feature_medians, np.where(missing)[1])
        return (x - self.feature_means) / self.feature_scales

    def predict_unreliability(self, features: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(self.transform(features))[:, 1]


def fit_leave_one_dataset_out(rows: list[dict]) -> dict[str, FoldCalibrator]:
    """Fit Appendix A.2.4 calibrators using labels from non-held-out datasets only."""
    datasets = sorted({str(row["dataset"]) for row in rows})
    if len(datasets) < 2:
        raise ValueError("LODO requires at least two datasets")
    calibrators: dict[str, FoldCalibrator] = {}
    for held_out in datasets:
        train = [row for row in rows if row["dataset"] != held_out]
        calibrators[held_out] = fit_calibrator(train, held_out)
    return calibrators


def fit_calibrator(train: list[dict], held_out: str = "external") -> FoldCalibrator:
    """Fit imputation, normalization, and logistic fusion on explicit training rows."""
    from sklearn.linear_model import LogisticRegression

    if not train:
        raise ValueError("Fusion requires nonempty training records")
    x = np.asarray(
        [
            [row.get("answer_shift", np.nan), row.get("contextual_hesitation", np.nan)]
            for row in train
        ],
        dtype=float,
    )
    y = np.asarray([row["incorrect"] for row in train])
    x[~np.isfinite(x)] = np.nan
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError(
            f"Training fold for {held_out} must contain correct and incorrect answers"
        )
    if np.isnan(x).all(axis=0).any():
        raise ValueError(
            f"Cannot impute {held_out}: a CATCH feature is missing in every training row"
        )
    medians = np.nanmedian(x, axis=0)
    missing = ~np.isfinite(x)
    x[missing] = np.take(medians, np.where(missing)[1])
    means, scales = x.mean(axis=0), x.std(axis=0)
    scales[scales == 0] = 1.0
    classifier = LogisticRegression(solver="lbfgs", random_state=0, max_iter=1000)
    classifier.fit((x - means) / scales, y)
    return FoldCalibrator(held_out, medians, means, scales, classifier)
