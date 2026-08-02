import copy
from pathlib import Path

import pytest

from catch_uq.io import load_config
from catch_uq.method_config import validate_main_config, validate_sensitivity_override


def test_appendix_main_configuration():
    root = Path(__file__).parents[1]
    cfg = load_config(root / "configs/main.yaml")
    assert cfg["experiment"]["evaluation_runs"] == [41, 42, 43]
    assert cfg["experiment"]["max_new_tokens"] == 2048
    assert sum(item["expected_examples"] for item in cfg["datasets"].values()) * 4 == 8364
    assert cfg["text_perturbation"]["count"] == 4
    assert cfg["weight_perturbation"]["count"] == 4
    assert cfg["weight_perturbation"]["rank"] == 4
    assert cfg["weight_perturbation"]["sigma"] == 0.03
    assert cfg["weight_perturbation"]["seed_groups"] == {
        41: [42, 43, 44, 45],
        42: [46, 47, 48, 49],
        43: [50, 51, 52, 53],
    }
    assert cfg["hesitation"]["top_m"] == 10
    validate_main_config(cfg)


def test_appendix_configuration_rejects_conflicting_changes():
    root = Path(__file__).parents[1]
    cfg = load_config(root / "configs/main.yaml")
    conflicting = copy.deepcopy(cfg)
    conflicting["weight_perturbation"]["sigma"] = 0.1
    with pytest.raises(ValueError, match="sigma"):
        validate_main_config(conflicting)


def test_only_appendix_sensitivity_overrides_are_allowed():
    validate_sensitivity_override(4, 0.15, "sensitivity/sigma_0.15")
    validate_sensitivity_override(16, 0.03, "sensitivity/rank_16")
    with pytest.raises(ValueError, match="sensitivity"):
        validate_sensitivity_override(4, 0.1, None)
    with pytest.raises(ValueError, match="grid"):
        validate_sensitivity_override(3, 0.03, "sensitivity/rank_3")
