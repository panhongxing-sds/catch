"""End-to-end nine-decode CATCH experiment runner."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path

from tqdm import tqdm

from .answers import answers_equivalent
from .data import load_benchmark, load_variants, validate_scope
from .generation import (
    GenerationSettings,
    build_prompt,
    generate_greedy,
    load_model_and_tokenizer,
)
from .io import append_jsonl, load_config, read_jsonl
from .method_config import validate_main_config, validate_sensitivity_override
from .perturbations import LowRankGaussianPerturber, WeightPerturbationConfig
from .scoring import answer_shift, contextual_hesitation, training_free_score

TARGETS = {
    "QK": ("q_proj", "k_proj"),
    "VO": ("v_proj", "o_proj"),
    "MLP": ("gate_proj", "up_proj", "down_proj"),
    "QKVO": ("q_proj", "k_proj", "v_proj", "o_proj"),
}


def _completed_ids(path: Path) -> set[str]:
    return {str(row["id"]) for row in read_jsonl(path)} if path.exists() else set()


def run_experiment(
    config_path: str,
    dataset: str,
    model_key: str,
    evaluation_run: int,
    *,
    limit: int | None = None,
    device_map: str = "auto",
    weight_rank: int | None = None,
    weight_sigma: float | None = None,
    output_tag: str | None = None,
    top_m: int | None = None,
    target: str = "QK",
    hf_model: str | None = None,
    input_path: str | None = None,
    variants_path: str | None = None,
) -> Path:
    config = load_config(config_path)
    validate_main_config(config)
    if evaluation_run not in config["experiment"]["evaluation_runs"]:
        raise ValueError("Evaluation run must be 41, 42, or 43")
    validate_sensitivity_override(weight_rank, weight_sigma, output_tag, top_m)
    if target not in TARGETS:
        raise ValueError(f"Unknown perturbation target: {target}")
    if (target != "QK" or hf_model or input_path or variants_path) and not (
        output_tag and output_tag.startswith("additional/")
    ):
        raise ValueError(
            "Additional studies require a distinct additional/* output tag"
        )
    dataset_config = config["datasets"].get(dataset, {})
    model_name = hf_model or config["models"][model_key]
    benchmark_path = input_path or dataset_config.get("file")
    variants_path = variants_path or dataset_config.get("variants")
    if not benchmark_path or not variants_path:
        raise ValueError(
            "Additional datasets require explicit --input and --variants files"
        )
    rows = load_benchmark(benchmark_path, dataset)
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    if limit is None and not input_path:
        validate_scope(rows, int(dataset_config["expected_examples"]), dataset)
    else:
        rows = rows[:limit]
    variants = load_variants(variants_path)
    missing = [row["id"] for row in rows if row["id"] not in variants]
    if missing:
        raise ValueError(
            f"Missing four verified variants for {len(missing)} records; first: {missing[0]}"
        )

    output_root = Path(config["experiment"]["output_root"])
    if output_tag:
        output_root = output_root / output_tag
    output_path = output_root / model_key / f"run_{evaluation_run}" / f"{dataset}.jsonl"
    manifest = {
        "dataset": dataset,
        "model": model_name,
        "evaluation_run": evaluation_run,
        "limit": limit,
        "weight_rank": weight_rank or config["weight_perturbation"]["rank"],
        "weight_sigma": weight_sigma
        if weight_sigma is not None
        else config["weight_perturbation"]["sigma"],
        "target": target,
        "top_m": top_m or config["hesitation"]["top_m"],
        "input_sha256": hashlib.sha256(Path(benchmark_path).read_bytes()).hexdigest(),
        "variants_sha256": hashlib.sha256(Path(variants_path).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(
            b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))
        ).hexdigest(),
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "numpy")
        },
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(
                "Resume configuration/input/version mismatch; choose a new output tag"
            )
    elif output_path.exists():
        raise ValueError(
            "Cannot resume an old shard without a run manifest; choose a new output tag"
        )
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    completed = _completed_ids(output_path)
    if all(row["id"] in completed for row in rows):
        return output_path
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        device_map=device_map,
        precision=str(config["experiment"]["precision"]),
    )
    decoding = config["decoding"]["base_and_perturbations"]
    settings = GenerationSettings(
        max_new_tokens=int(config["experiment"]["max_new_tokens"]),
        top_m=int(top_m or config["hesitation"]["top_m"]),
        do_sample=bool(decoding["do_sample"]),
        temperature=float(decoding["temperature"]),
        top_p=float(decoding["top_p"]),
    )
    weight_cfg = config["weight_perturbation"]
    perturber = LowRankGaussianPerturber(
        model,
        WeightPerturbationConfig(
            rank=int(weight_rank if weight_rank is not None else weight_cfg["rank"]),
            sigma=float(
                weight_sigma if weight_sigma is not None else weight_cfg["sigma"]
            ),
            target_suffixes=TARGETS[target],
        ),
    )
    weight_seeds = list(weight_cfg["seed_groups"][evaluation_run])

    for row in tqdm(rows, desc=f"{model_key}/{dataset}/run{evaluation_run}"):
        if row["id"] in completed:
            continue
        base_prompt = build_prompt(tokenizer, row["question"], model_name)
        base = generate_greedy(model, tokenizer, base_prompt, settings)
        text_runs = [
            generate_greedy(
                model, tokenizer, build_prompt(tokenizer, variant, model_name), settings
            )
            for variant in variants[row["id"]]
        ]
        weight_runs: list[dict] = []
        for seed in weight_seeds:
            with perturber.sampled(int(seed)):
                generated = generate_greedy(model, tokenizer, base_prompt, settings)
            generated["weight_seed"] = int(seed)
            weight_runs.append(generated)
        perturbation_runs = text_runs + weight_runs
        hesitation = contextual_hesitation(perturbation_runs, settings.top_m)
        record = {
            **row,
            "model": model_name,
            "evaluation_run": evaluation_run,
            "weight_seeds": weight_seeds,
            "weight_rank": int(
                weight_rank if weight_rank is not None else weight_cfg["rank"]
            ),
            "weight_sigma": float(
                weight_sigma if weight_sigma is not None else weight_cfg["sigma"]
            ),
            "weight_target": target,
            "top_m": settings.top_m,
            "training_free_score": training_free_score(
                base["answer"], perturbation_runs, settings.top_m
            ),
            "base": base,
            "text_runs": text_runs,
            "weight_runs": weight_runs,
            "answer_shift": answer_shift(
                base["answer"], [run["answer"] for run in perturbation_runs]
            ),
            "contextual_hesitation": hesitation if math.isfinite(hesitation) else None,
            "incorrect": int(not answers_equivalent(base["answer"], row["reference"])),
        }
        append_jsonl(output_path, record)
    return output_path
