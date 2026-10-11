import copy
import json

import pytest

from catch_uq.cli import build_parser
from catch_uq.evaluate import evaluate_external, evaluate_model_run, metrics
from catch_uq.io import append_jsonl


def records(dataset):
    rows = []
    for i in range(6):
        answer = "2" if i < 3 else "3"
        run = {
            "answer": answer,
            "answer_start_token": 1,
            "token_trace": [{"top_logprobs": [0.0, -float(i + 1)]}],
        }
        rows.append(
            {
                "id": str(i),
                "dataset": dataset,
                "incorrect": int(i >= 3),
                "question": f"{dataset} question {i}",
                "base": {"answer": "2"},
                "text_runs": [run],
                "weight_runs": [run],
            }
        )
    return rows


def test_both_algorithms_are_scored_from_same_cached_trajectories(tmp_path):
    for dataset in ("a", "b", "c"):
        for row in records(dataset):
            append_jsonl(tmp_path / "tiny" / "run_41" / f"{dataset}.jsonl", row)
    result = evaluate_model_run(tmp_path, "tiny", 41, ("a", "b", "c"), top_m=2)
    assert all(result[d]["n"] == 6 for d in result)
    assert all(
        set(result[d]) == {"n", "error_rate", "supervised", "training_free"}
        for d in result
    )
    audit = json.loads(
        (tmp_path / "tiny/run_41/both_full_M2_calibration.json").read_text()
    )
    assert audit["a"]["training_datasets"] == ["b", "c"]
    assert metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == {
        "auroc": 1.0,
        "auprc": 1.0,
        "selective_accuracy_50": 1.0,
    }


def test_training_free_evaluation_needs_no_training_dataset(tmp_path):
    for row in records("a"):
        append_jsonl(tmp_path / "tiny/run_41/a.jsonl", row)
    result = evaluate_model_run(
        tmp_path, "tiny", 41, ("a",), mode="training_free", top_m=2
    )
    assert "supervised" not in result["a"]
    assert not list(tmp_path.rglob("*calibration.json"))


def test_external_calibration_manifest_rejects_overlapping_questions(tmp_path):
    train, test = records("train"), records("test")
    for row in train:
        append_jsonl(tmp_path / "train.jsonl", row)
    for row in test:
        append_jsonl(tmp_path / "test.jsonl", row)
    manifest = tmp_path / "study.json"
    manifest.write_text(json.dumps({"train": ["train.jsonl"], "test": ["test.jsonl"]}))
    result = evaluate_external(str(manifest), str(tmp_path / "metrics.json"), top_m=2)
    assert "supervised" in result and "training_free" in result
    overlap = copy.deepcopy(test[0])
    overlap["question"] = train[0]["question"]
    (tmp_path / "test.jsonl").write_text(json.dumps(overlap) + "\n")
    with pytest.raises(ValueError, match="questions overlap"):
        evaluate_external(str(manifest), str(tmp_path / "metrics.json"), top_m=2)
    manifest.write_text(json.dumps({"train": ["test.jsonl"], "test": ["test.jsonl"]}))
    with pytest.raises(ValueError, match="IDs overlap"):
        evaluate_external(str(manifest), str(tmp_path / "metrics.json"), top_m=2)


def test_cli_exposes_training_free_scoring_and_ablation():
    args = build_parser().parse_args(
        [
            "score",
            "--input",
            "runs.jsonl",
            "--output",
            "scores.jsonl",
            "--variant",
            "V5",
            "--top-m",
            "4",
        ]
    )
    assert args.variant == "V5" and args.top_m == 4
    assert not hasattr(args, "mode")
