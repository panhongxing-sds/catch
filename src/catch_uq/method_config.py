"""Fail-fast validation of the anonymous-paper CATCH main configuration."""

from __future__ import annotations

EXPECTED_DATASETS = {"minerva": 272, "math500": 500, "gsm8k": 1319}
EXPECTED_MODELS = {
    "qwen25_3b": "Qwen/Qwen2.5-3B-Instruct",
    "llama32_1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama31_8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "qwen3_8b": "Qwen/Qwen3-8B",
}
EXPECTED_FAMILIES = [
    "surface_rewording",
    "role_preserving_entity_or_scenario_substitution",
    "condition_reordering",
    "irrelevant_distractor_insertion",
]
EXPECTED_SEED_GROUPS = {
    41: [42, 43, 44, 45],
    42: [46, 47, 48, 49],
    43: [50, 51, 52, 53],
}
SENSITIVITY_SIGMAS = {0.01, 0.03, 0.05, 0.07, 0.10, 0.15}
SENSITIVITY_RANKS = {1, 2, 4, 8, 16}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"CATCH appendix configuration mismatch: {message}")


def validate_main_config(config: dict) -> None:
    experiment = config["experiment"]
    _require(experiment["evaluation_runs"] == [41, 42, 43], "evaluation runs must be 41/42/43")
    _require(experiment["max_new_tokens"] == 2048, "maximum new tokens must be 2048")
    _require(experiment["precision"] == "bfloat16", "precision must be bfloat16")
    _require(
        experiment["inference_framework"] == "huggingface_transformers",
        "inference framework must be Hugging Face Transformers",
    )

    datasets = config["datasets"]
    _require(set(datasets) == set(EXPECTED_DATASETS), "dataset set differs from the main experiment")
    for name, expected_size in EXPECTED_DATASETS.items():
        _require(
            datasets[name]["expected_examples"] == expected_size,
            f"{name} must contain {expected_size} examples",
        )
    _require(config["models"] == EXPECTED_MODELS, "backbone model identifiers differ from Table 1")

    decoding = config["decoding"]["base_and_perturbations"]
    _require(decoding["do_sample"] is False, "base and perturbation decodes must be greedy")
    _require(decoding["temperature"] == 0.0, "greedy temperature must be 0")
    _require(decoding["top_p"] == 1.0, "greedy top_p must be 1")

    text = config["text_perturbation"]
    _require(text["count"] == 4, "R must be 4")
    _require(text["families"] == EXPECTED_FAMILIES, "text transformation families or order differ")
    _require(text["generator"]["model"] == "deepseek-v4-pro", "text model must be deepseek-v4-pro")
    _require(text["generator"]["temperature"] == 0.8, "text temperature must be 0.8")
    _require(text["generator"]["top_p"] == 0.95, "text top_p must be 0.95")
    _require(text["verifier"]["model"] == "deepseek-v4-pro", "verifier model must match Appendix A.3.2")
    _require(text["verifier"]["thinking_mode"] is True, "verifier thinking mode must be enabled")

    weight = config["weight_perturbation"]
    _require(weight["count"] == 4, "W must be 4")
    _require(weight["rank"] == 4, "main-experiment weight rank must be 4")
    _require(weight["sigma"] == 0.03, "main-experiment weight sigma must be 0.03")
    _require(weight["target_suffixes"] == ["q_proj", "k_proj"], "only Q/K projections are allowed")
    _require(weight["layers"] == "all", "Q/K perturbations must cover all layers")
    _require(weight["seed_groups"] == EXPECTED_SEED_GROUPS, "weight seed groups differ from Table 3")

    hesitation = config["hesitation"]
    _require(hesitation["top_m"] == 10, "M must be 10")
    _require(hesitation["include_base_trace"] is False, "the base trace must not enter hesitation")
    _require(hesitation["pre_answer_only"] is True, "hesitation must stop at the answer span")
    _require(hesitation["proximity_weight"] == "linear", "proximity weight must be linear")
    _require(
        hesitation["missing_answer_span"] == "training_fold_median",
        "missing hesitation must use the training-fold median",
    )

    calibration = config["calibration"]
    _require(calibration["protocol"] == "leave_one_dataset_out", "calibration must be LODO")
    _require(calibration["datasets"] == ["minerva", "math500", "gsm8k"], "LODO dataset order differs")
    _require(calibration["standardize_on_training_fold"] is True, "features must use training-fold statistics")
    _require(calibration["model"] == "logistic_regression", "fusion must use logistic regression")
    _require(
        calibration["features"] == ["answer_shift", "contextual_hesitation"],
        "fusion features must be Answer Shift and Contextual Hesitation",
    )
    _require(calibration["positive_label"] == "incorrect", "incorrect answers must be positive")
    _require(
        config["metrics"]
        == ["auroc", "auprc", "selective_accuracy_at_50_percent_coverage"],
        "reported main metrics differ from the paper",
    )


def validate_sensitivity_override(
    rank: int | None, sigma: float | None, output_tag: str | None
) -> None:
    if rank is None and sigma is None:
        return
    _require(
        bool(output_tag and output_tag.startswith("sensitivity/")),
        "rank/sigma overrides are allowed only under a sensitivity/* output tag",
    )
    effective_rank = 4 if rank is None else rank
    effective_sigma = 0.03 if sigma is None else sigma
    valid_sigma_sweep = effective_rank == 4 and effective_sigma in SENSITIVITY_SIGMAS
    valid_rank_sweep = effective_sigma == 0.03 and effective_rank in SENSITIVITY_RANKS
    _require(valid_sigma_sweep or valid_rank_sweep, "override is outside the Appendix C.2 grid")
