import math

import numpy as np

from catch_uq.scoring import (
    answer_shift,
    contextual_hesitation,
    fit_leave_one_dataset_out,
    local_topm_entropy,
    run_hesitation,
)


def test_answer_shift_equation_5():
    assert answer_shift("25", ["25", "23", "25", "23"]) == 0.5


def test_topm_entropy_is_renormalized():
    assert math.isclose(local_topm_entropy([0.0, 0.0]), math.log(2), rel_tol=1e-7)


def test_contextual_hesitation_marks_missing_run_for_fold_imputation():
    trace = [
        {"top_logprobs": [0.0, -1.0]},
        {"top_logprobs": [0.0, -2.0]},
        {"top_logprobs": [0.0, -3.0]},
    ]
    actual = contextual_hesitation(
        [{"token_trace": trace, "answer_start_token": 3}, {"token_trace": [], "answer_start_token": None}]
    )
    assert math.isnan(actual)


def test_contextual_hesitation_averages_all_valid_perturbation_runs():
    trace = [
        {"top_logprobs": [0.0, -1.0]},
        {"top_logprobs": [0.0, -2.0]},
    ]
    expected = run_hesitation(trace, 2, top_m=10)
    assert contextual_hesitation(
        [
            {"token_trace": trace, "answer_start_token": 2},
            {"token_trace": trace, "answer_start_token": 2},
        ]
    ) == expected


def test_leave_one_dataset_out_never_uses_held_out_labels():
    rows = []
    for dataset, offset in (("a", 0.0), ("b", 0.1), ("c", 0.2)):
        for index in range(8):
            rows.append(
                {
                    "dataset": dataset,
                    "answer_shift": index / 7 + offset,
                    "contextual_hesitation": index / 10,
                    "incorrect": int(index >= 4),
                }
            )
    calibrators = fit_leave_one_dataset_out(rows)
    scores = calibrators["a"].predict_unreliability(np.array([[0.0, 0.0], [1.0, 1.0]]))
    assert scores[1] > scores[0]
    flipped = [dict(row) for row in rows]
    for row in flipped:
        if row["dataset"] == "a":
            row["incorrect"] = 1 - row["incorrect"]
    flipped_calibrator = fit_leave_one_dataset_out(flipped)["a"]
    assert np.allclose(
        calibrators["a"].classifier.coef_, flipped_calibrator.classifier.coef_
    )
    assert np.allclose(
        calibrators["a"].classifier.intercept_, flipped_calibrator.classifier.intercept_
    )
