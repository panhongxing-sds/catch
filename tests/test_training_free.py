import copy
import math

import numpy as np
import pytest

from catch_uq.scoring import (
    component_features,
    fit_leave_one_dataset_out,
    run_hesitation,
    training_free_score,
    training_free_variant,
    trajectory_commitment,
)


def trajectory(answer="2", entropy="zero", alpha=2):
    logprobs = [0.0, -math.inf] if entropy == "zero" else [0.0, 0.0]
    return {
        "answer": answer,
        "answer_start_token": alpha,
        "token_trace": [{"top_logprobs": logprobs} for _ in range(3)],
    }


def test_gate_uses_joint_trajectory_information_not_separate_means():
    a = [trajectory("2"), trajectory("3", "max")]
    b = [trajectory("2", "max"), trajectory("3")]
    assert training_free_score("2", a, 2) == 0.5
    assert training_free_score("2", b, 2) == 1.0
    assert np.mean(
        [run_hesitation(r["token_trace"], 2, 2) for r in a]
    ) == pytest.approx(np.mean([run_hesitation(r["token_trace"], 2, 2) for r in b]))


@pytest.mark.parametrize(
    "bad",
    [
        {"answer": "", "answer_start_token": 2},
        {"answer": "2", "answer_start_token": None},
        {"answer": "2", "answer_start_token": 0},
        {"answer": "2", "answer_start_token": 100},
        {"answer": "2", "answer_start_token": 2, "token_trace": []},
        {
            "answer": "2",
            "answer_start_token": 2,
            "token_trace": [{"top_logprobs": [0.0]}] * 3,
        },
    ],
)
def test_invalid_runs_have_zero_support_but_stay_in_denominator(bad):
    invalid = trajectory() | bad
    assert trajectory_commitment(invalid, 2) == 0
    assert training_free_score("2", [trajectory(), invalid], 2) == 0.5


def test_near_answer_linear_weights_and_normalized_bounds():
    trace = [{"top_logprobs": [0.0, -math.inf]}, {"top_logprobs": [0.0, 0.0]}]
    assert run_hesitation(trace, 2, 2) == pytest.approx(2 * math.log(2) / 3)
    assert trajectory_commitment(
        {"answer": "2", "answer_start_token": 2, "token_trace": trace}, 2
    ) == pytest.approx(1 / 3)


def test_ablation_gates_and_budgets():
    row = {
        "base": {"answer": "2"},
        "text_runs": [trajectory()],
        "weight_runs": [trajectory("3", "max")],
    }
    assert training_free_variant(row, "V1", 2) == 0.5
    assert training_free_variant(row, "V2", 2) == 0.5
    assert training_free_variant(row, "V3", 2) == 1
    assert training_free_variant(row, "V4", 2) == 0
    assert component_features(row, "V1", 2)[0] == 0
    assert component_features(row, "V2", 2)[1] == 0
    changed = copy.deepcopy(row)
    changed["text_runs"][0]["token_trace"][2] = {"top_logprobs": [0.0, 0.0]}
    assert training_free_variant(changed, "full", 2) == 0.5
    assert training_free_variant(changed, "V5", 2) == pytest.approx(2 / 3)


def test_held_out_features_do_not_change_imputation_or_scaling():
    rows = [
        {
            "dataset": d,
            "answer_shift": i / 6,
            "contextual_hesitation": None if i == 0 else i / 4,
            "incorrect": int(i >= 3),
        }
        for d in ["a", "b", "c"]
        for i in range(6)
    ]
    original = fit_leave_one_dataset_out(rows)["a"]
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["dataset"] == "a":
            row.update(answer_shift=1e9, contextual_hesitation=-1e9, incorrect=0)
    held_out = fit_leave_one_dataset_out(changed)["a"]
    for name in ("feature_medians", "feature_means", "feature_scales"):
        assert np.array_equal(getattr(original, name), getattr(held_out, name))
    assert np.array_equal(original.classifier.coef_, held_out.classifier.coef_)


def test_missing_labels_do_not_affect_training_free_score():
    row = {"base": {"answer": "2"}, "text_runs": [trajectory()], "weight_runs": []}
    assert training_free_variant(row, top_m=2) == 0
    with pytest.raises(ValueError):
        training_free_score("2", [], 2)
    with pytest.raises(ValueError):
        training_free_score("2", [trajectory()], 1)
