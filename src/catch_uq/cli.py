"""Command-line entry points for the CATCH-only workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .data import load_benchmark, validate_scope
from .evaluate import aggregate_runs, evaluate_model_run
from .io import append_jsonl, load_config, read_jsonl
from .method_config import validate_main_config
from .runner import run_experiment
from .scoring import training_free_variant
from .variants import generate_verified_variants


def _validate(args) -> None:
    config = load_config(args.config)
    validate_main_config(config)
    for dataset, spec in config["datasets"].items():
        rows = load_benchmark(spec["file"], dataset)
        validate_scope(rows, int(spec["expected_examples"]), dataset)
        print(f"{dataset}: {len(rows)} rows OK")


def _validate_config(args) -> None:
    validate_main_config(load_config(args.config))
    print("CATCH configuration matches the paper appendix")


def _variants(args) -> None:
    config = load_config(args.config)
    validate_main_config(config)
    rephrase = config["text_perturbation"]
    generator = dict(rephrase["generator"])
    if args.generator_model:
        generator.update(
            model=args.generator_model,
            api_style="huggingface",
            seed=args.generator_seed,
        )
    manifest = {
        "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        "generator": generator,
        "verifier": rephrase["verifier"],
    }
    manifest_path = Path(args.output).with_suffix(".manifest.json")
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(
            "Variant cache configuration changed; choose a new output file"
        )
    if Path(args.output).exists() and not manifest_path.exists():
        raise ValueError(
            "Variant cache has no configuration manifest; choose a new output file"
        )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    completed = (
        {row["id"] for row in read_jsonl(args.output)}
        if Path(args.output).exists()
        else set()
    )
    for row in load_benchmark(args.input, args.dataset):
        if row["id"] in completed:
            continue
        variants = generate_verified_variants(
            row["question"],
            model=rephrase["generator"]["model"],
            temperature=float(rephrase["generator"]["temperature"]),
            top_p=float(rephrase["generator"]["top_p"]),
            generator=generator,
            verifier=rephrase["verifier"],
        )
        append_jsonl(args.output, {"id": row["id"], "variants": variants})
        print(f"{row['id']}: four variants generated and verified", flush=True)


def _run(args) -> None:
    path = run_experiment(
        args.config,
        args.dataset,
        args.model,
        args.evaluation_run,
        limit=args.limit,
        device_map=args.device_map,
        weight_rank=args.weight_rank,
        weight_sigma=args.weight_sigma,
        output_tag=args.output_tag,
        top_m=args.top_m,
        target=args.target,
        hf_model=args.hf_model,
        input_path=args.input,
        variants_path=args.variants,
    )
    print(path)


def _evaluate(args) -> None:
    config = load_config(args.config)
    validate_main_config(config)
    print(
        json.dumps(
            aggregate_runs(
                args.results_root,
                args.model,
                tuple(config["experiment"]["evaluation_runs"]),
                tuple(config["calibration"]["datasets"]),
                mode=args.mode,
                variant=args.variant,
                top_m=args.top_m,
            ),
            indent=2,
        )
    )


def _evaluate_run(args) -> None:
    config = load_config(args.config)
    validate_main_config(config)
    print(
        json.dumps(
            evaluate_model_run(
                args.results_root,
                args.model,
                args.evaluation_run,
                tuple(args.datasets.split(","))
                if args.datasets
                else tuple(config["calibration"]["datasets"]),
                mode=args.mode,
                variant=args.variant,
                top_m=args.top_m,
            ),
            indent=2,
        )
    )


def _score(args) -> None:
    rows = read_jsonl(args.input)
    scores = [
        {
            "id": row["id"],
            "training_free": training_free_variant(row, args.variant, args.top_m),
        }
        for row in rows
    ]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        "".join(json.dumps(row) + "\n" for row in scores), encoding="utf-8"
    )
    print(f"Scored {len(scores)} questions without labels or calibration")


def _external(args) -> None:
    from .evaluate import evaluate_external

    print(
        json.dumps(
            evaluate_external(
                args.manifest, args.output, args.variant, args.top_m, args.mode
            ),
            indent=2,
        )
    )


def _scoring_arguments(parser, *, label_free=False):
    if not label_free:
        parser.add_argument(
            "--mode", choices=["both", "supervised", "training_free"], default="both"
        )
    parser.add_argument(
        "--variant", choices=["full", "V1", "V2", "V3", "V4", "V5"], default="full"
    )
    parser.add_argument("--top-m", type=int, default=10)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="catch-uq")
    sub = parser.add_subparsers(required=True)

    validate = sub.add_parser(
        "validate-data", help="require the exact Appendix D.5.1 cohort sizes"
    )
    validate.add_argument("--config", default="configs/main.yaml")
    validate.set_defaults(func=_validate)

    validate_config = sub.add_parser(
        "validate-config",
        help="check every paper-locked CATCH method setting without loading data",
    )
    validate_config.add_argument("--config", default="configs/main.yaml")
    validate_config.set_defaults(func=_validate_config)

    variants = sub.add_parser(
        "generate-variants", help="generate and verify four CATCH text perturbations"
    )
    variants.add_argument("--config", default="configs/main.yaml")
    variants.add_argument("--input", required=True)
    variants.add_argument("--output", required=True)
    variants.add_argument("--dataset", required=True)
    variants.add_argument(
        "--generator-model", help="HF model ID for the open-source generator ablation"
    )
    variants.add_argument("--generator-seed", type=int, default=42)
    variants.set_defaults(func=_variants)

    run = sub.add_parser("run", help="run one CATCH model/dataset/evaluation-run shard")
    run.add_argument("--config", default="configs/main.yaml")
    run.add_argument("--dataset", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--evaluation-run", required=True, type=int, choices=[41, 42, 43])
    run.add_argument("--device-map", default="auto")
    run.add_argument("--limit", type=int)
    run.add_argument("--weight-rank", type=int)
    run.add_argument("--weight-sigma", type=float)
    run.add_argument("--output-tag")
    run.add_argument("--top-m", type=int)
    run.add_argument("--target", choices=["QK", "VO", "MLP", "QKVO"], default="QK")
    run.add_argument("--hf-model", help="HF target model ID for an additional study")
    run.add_argument("--input", help="Explicit additional benchmark JSONL")
    run.add_argument("--variants", help="Explicit additional verified variants JSONL")
    run.set_defaults(func=_run)

    evaluate = sub.add_parser(
        "evaluate", help="calibrate and aggregate CATCH over three runs"
    )
    evaluate.add_argument("--config", default="configs/main.yaml")
    evaluate.add_argument("--results-root", default="results")
    evaluate.add_argument("--model", required=True)
    _scoring_arguments(evaluate)
    evaluate.set_defaults(func=_evaluate)

    evaluate_run = sub.add_parser(
        "evaluate-run", help="calibrate one CATCH run, including a sensitivity setting"
    )
    evaluate_run.add_argument("--config", default="configs/main.yaml")
    evaluate_run.add_argument("--results-root", required=True)
    evaluate_run.add_argument("--model", required=True)
    evaluate_run.add_argument("--evaluation-run", type=int, default=41)
    evaluate_run.add_argument(
        "--datasets", help="Comma-separated shard names; TF supports a single dataset"
    )
    _scoring_arguments(evaluate_run)
    evaluate_run.set_defaults(func=_evaluate_run)

    score = sub.add_parser(
        "score",
        help="Label-free Equation 8 scoring of cached perturbation trajectories",
    )
    score.add_argument("--input", required=True)
    score.add_argument("--output", required=True)
    _scoring_arguments(score, label_free=True)
    score.set_defaults(func=_score)

    external = sub.add_parser(
        "evaluate-external",
        help="Evaluate an explicit disjoint train/test study manifest",
    )
    external.add_argument("--manifest", required=True)
    external.add_argument("--output", required=True)
    _scoring_arguments(external)
    external.set_defaults(func=_external)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
