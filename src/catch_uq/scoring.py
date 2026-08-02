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
        np.mean([not answers_equivalent(answer, base_answer) for answer in perturbation_answers])
    )


def local_topm_entropy(top_logprobs: list[float]) -> float:
    if not top_logprobs:
        return float("nan")
    values = np.asarray(top_logprobs, dtype=float)
    values = values - np.max(values)
    probabilities = np.exp(values)
    probabilities /= probabilities.sum()
    return float(-np.sum(probabilities * np.log(np.clip(probabilities, 1e-30, None))))


def run_hesitation(token_trace: list[dict], answer_start_token: int | None, top_m: int = 10) -> float:
    """Equation 7 for one response, restricted to tokens before the answer span."""
    if answer_start_token is None or answer_start_token <= 0:
        return float("nan")
    alpha = min(answer_start_token, len(token_trace))
    entropies: list[float] = []
    for token in token_trace[:alpha]:
        values = token.get("top_logprobs") or token.get("topk_logprobs") or []
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
    from sklearn.linear_model import LogisticRegression

    datasets = sorted({str(row["dataset"]) for row in rows})
    calibrators: dict[str, FoldCalibrator] = {}
    for held_out in datasets:
        train = [row for row in rows if row["dataset"] != held_out]
        x = np.asarray(
            [[row.get("answer_shift", np.nan), row.get("contextual_hesitation", np.nan)] for row in train],
            dtype=float,
        )
        y = np.asarray([int(row["incorrect"]) for row in train], dtype=int)
        medians = np.nanmedian(x, axis=0)
        if np.any(~np.isfinite(medians)):
            raise ValueError(f"Cannot impute {held_out}: a CATCH feature is missing in every training row")
        missing = ~np.isfinite(x)
        x[missing] = np.take(medians, np.where(missing)[1])
        means = x.mean(axis=0)
        scales = x.std(axis=0)
        scales[scales == 0] = 1.0
        clf = LogisticRegression(solver="lbfgs", random_state=0, max_iter=1000)
        clf.fit((x - means) / scales, y)
        calibrators[held_out] = FoldCalibrator(held_out, medians, means, scales, clf)
    return calibrators
