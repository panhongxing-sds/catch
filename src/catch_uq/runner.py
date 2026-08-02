"""End-to-end nine-decode CATCH experiment runner."""

from __future__ import annotations

import math
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
from .scoring import answer_shift, contextual_hesitation


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
) -> Path:
    config = load_config(config_path)
    validate_main_config(config)
    validate_sensitivity_override(weight_rank, weight_sigma, output_tag)
    dataset_config = config["datasets"][dataset]
    model_name = config["models"][model_key]
    benchmark_path = dataset_config["file"]
    variants_path = dataset_config["variants"]
    rows = load_benchmark(benchmark_path, dataset)
    if limit is None:
        validate_scope(rows, int(dataset_config["expected_examples"]), dataset)
    else:
        rows = rows[:limit]
    variants = load_variants(variants_path)
    missing = [row["id"] for row in rows if row["id"] not in variants]
    if missing:
        raise ValueError(f"Missing four verified variants for {len(missing)} records; first: {missing[0]}")

    output_root = Path(config["experiment"]["output_root"])
    if output_tag:
        output_root = output_root / output_tag
    output_path = output_root / model_key / f"run_{evaluation_run}" / f"{dataset}.jsonl"
    completed = _completed_ids(output_path)
    model, tokenizer = load_model_and_tokenizer(
        model_name,
        device_map=device_map,
        precision=str(config["experiment"]["precision"]),
    )
    decoding = config["decoding"]["base_and_perturbations"]
    settings = GenerationSettings(
        max_new_tokens=int(config["experiment"]["max_new_tokens"]),
        top_m=int(config["hesitation"]["top_m"]),
        do_sample=bool(decoding["do_sample"]),
        temperature=float(decoding["temperature"]),
        top_p=float(decoding["top_p"]),
    )
    weight_cfg = config["weight_perturbation"]
    perturber = LowRankGaussianPerturber(
        model,
        WeightPerturbationConfig(
            rank=int(weight_rank if weight_rank is not None else weight_cfg["rank"]),
            sigma=float(weight_sigma if weight_sigma is not None else weight_cfg["sigma"]),
            target_suffixes=tuple(weight_cfg["target_suffixes"]),
        ),
    )
    weight_seeds = list(weight_cfg["seed_groups"][evaluation_run])

    for row in tqdm(rows, desc=f"{model_key}/{dataset}/run{evaluation_run}"):
        if row["id"] in completed:
            continue
        base_prompt = build_prompt(tokenizer, row["question"], model_name)
        base = generate_greedy(model, tokenizer, base_prompt, settings)
        text_runs = [
            generate_greedy(model, tokenizer, build_prompt(tokenizer, variant, model_name), settings)
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
            "weight_rank": int(weight_rank if weight_rank is not None else weight_cfg["rank"]),
            "weight_sigma": float(weight_sigma if weight_sigma is not None else weight_cfg["sigma"]),
            "base": base,
            "text_runs": text_runs,
            "weight_runs": weight_runs,
            "answer_shift": answer_shift(base["answer"], [run["answer"] for run in perturbation_runs]),
            "contextual_hesitation": hesitation if math.isfinite(hesitation) else None,
            "incorrect": int(not answers_equivalent(base["answer"], row["reference"])),
        }
        append_jsonl(output_path, record)
    return output_path
