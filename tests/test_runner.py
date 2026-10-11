import json
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")

from catch_uq import runner
from catch_uq.io import append_jsonl, load_config, read_jsonl


def test_nine_generation_budget_metadata_resume_and_training_free_score(
    tmp_path, monkeypatch
):
    cfg = load_config(Path(__file__).parents[1] / "configs/main.yaml")
    cfg["experiment"]["output_root"] = str(tmp_path / "results")
    input_path, variants_path = tmp_path / "input.jsonl", tmp_path / "variants.jsonl"
    cfg["datasets"]["math500"].update(file=str(input_path), variants=str(variants_path))
    cfg_path = tmp_path / "main.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    append_jsonl(input_path, {"id": "one", "question": "Compute 1+1.", "answer": "2"})
    append_jsonl(
        variants_path,
        {"id": "one", "variants": [f"Compute 1+1. Variant {i}" for i in range(4)]},
    )
    model = torch.nn.Module()
    model.q_proj = torch.nn.Linear(5, 4, bias=False)
    model.k_proj = torch.nn.Linear(5, 4, bias=False)
    original = model.q_proj.weight.detach().clone()

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return messages[-1]["content"]

    monkeypatch.setattr(
        runner, "load_model_and_tokenizer", lambda *a, **kw: (model, Tokenizer())
    )
    snapshots = []

    def generate(*args):
        snapshots.append(model.q_proj.weight.detach().clone())
        return {
            "answer": "2",
            "answer_start_token": 1,
            "token_trace": [{"top_logprobs": [float(-i) for i in range(10)]}],
        }

    monkeypatch.setattr(runner, "generate_greedy", generate)
    output = runner.run_experiment(
        str(cfg_path), "math500", "qwen25_3b", 41, limit=1, output_tag="smoke"
    )
    assert len(snapshots) == 9
    assert all(torch.equal(w, original) for w in snapshots[:5])
    assert all(not torch.equal(w, original) for w in snapshots[5:])
    assert torch.equal(model.q_proj.weight, original)
    row = read_jsonl(output)[0]
    assert len(row["text_runs"]) == len(row["weight_runs"]) == 4
    assert row["weight_seeds"] == [42, 43, 44, 45]
    assert row["training_free_score"] == pytest.approx(
        row["contextual_hesitation"] / __import__("math").log(10)
    )
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["top_m"] == 10 and len(manifest["code_sha256"]) == 64
    runner.run_experiment(
        str(cfg_path), "math500", "qwen25_3b", 41, limit=1, output_tag="smoke"
    )
    assert len(snapshots) == 9
    with pytest.raises(ValueError, match="mismatch"):
        runner.run_experiment(
            str(cfg_path), "math500", "qwen25_3b", 41, limit=2, output_tag="smoke"
        )
