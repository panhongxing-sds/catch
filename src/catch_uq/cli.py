"""Command-line entry points for the CATCH-only workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_benchmark, validate_scope
from .evaluate import aggregate_runs, evaluate_model_run
from .io import append_jsonl, load_config, read_jsonl
from .method_config import validate_main_config
from .runner import run_experiment
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
    completed = {row["id"] for row in read_jsonl(args.output)} if Path(args.output).exists() else set()
    for row in load_benchmark(args.input, args.dataset):
        if row["id"] in completed:
            continue
        variants = generate_verified_variants(
            row["question"],
            model=rephrase["generator"]["model"],
            temperature=float(rephrase["generator"]["temperature"]),
            top_p=float(rephrase["generator"]["top_p"]),
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
                tuple(config["calibration"]["datasets"]),
            ),
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="catch-uq")
    sub = parser.add_subparsers(required=True)

    validate = sub.add_parser("validate-data", help="require the exact Appendix D.5.1 cohort sizes")
    validate.add_argument("--config", default="configs/main.yaml")
    validate.set_defaults(func=_validate)

    validate_config = sub.add_parser(
        "validate-config", help="check every paper-locked CATCH method setting without loading data"
    )
    validate_config.add_argument("--config", default="configs/main.yaml")
    validate_config.set_defaults(func=_validate_config)

    variants = sub.add_parser("generate-variants", help="generate and verify four CATCH text perturbations")
    variants.add_argument("--config", default="configs/main.yaml")
    variants.add_argument("--input", required=True)
    variants.add_argument("--output", required=True)
    variants.add_argument("--dataset", required=True)
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
    run.set_defaults(func=_run)

    evaluate = sub.add_parser("evaluate", help="calibrate and aggregate CATCH over three runs")
    evaluate.add_argument("--config", default="configs/main.yaml")
    evaluate.add_argument("--results-root", default="results")
    evaluate.add_argument("--model", required=True)
    evaluate.set_defaults(func=_evaluate)

    evaluate_run = sub.add_parser(
        "evaluate-run", help="calibrate one CATCH run, including a sensitivity setting"
    )
    evaluate_run.add_argument("--config", default="configs/main.yaml")
    evaluate_run.add_argument("--results-root", required=True)
    evaluate_run.add_argument("--model", required=True)
    evaluate_run.add_argument("--evaluation-run", type=int, default=41)
    evaluate_run.set_defaults(func=_evaluate_run)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
