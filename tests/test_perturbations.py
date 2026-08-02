import hashlib

import pytest

torch = pytest.importorskip("torch")

from catch_uq.perturbations import LowRankGaussianPerturber, WeightPerturbationConfig


class TinyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(5, 4, bias=False)
        self.k_proj = torch.nn.Linear(5, 4, bias=False)
        self.v_proj = torch.nn.Linear(5, 4, bias=False)


def test_only_qk_change_and_weights_restore():
    model = TinyAttention()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    perturber = LowRankGaussianPerturber(model, WeightPerturbationConfig(rank=2, sigma=0.3))
    with perturber.sampled(42):
        assert not torch.equal(model.q_proj.weight, before["q_proj.weight"])
        assert not torch.equal(model.k_proj.weight, before["k_proj.weight"])
        assert torch.equal(model.v_proj.weight, before["v_proj.weight"])
    assert torch.equal(model.q_proj.weight, before["q_proj.weight"])
    assert torch.equal(model.k_proj.weight, before["k_proj.weight"])


def test_seed_is_reproducible():
    model = TinyAttention()
    perturber = LowRankGaussianPerturber(model, WeightPerturbationConfig(rank=2, sigma=0.3))
    snapshots = []
    for _ in range(2):
        with perturber.sampled(7):
            snapshots.append(model.q_proj.weight.detach().clone())
    assert torch.equal(snapshots[0], snapshots[1])


def test_update_matches_appendix_low_rank_gaussian_formula():
    model = TinyAttention()
    config = WeightPerturbationConfig(rank=2, sigma=0.3)
    perturber = LowRankGaussianPerturber(model, config)
    before = model.q_proj.weight.detach().clone()
    u, vh = perturber._bases["q_proj"]
    matrix_seed = int.from_bytes(hashlib.sha256(b"19:q_proj").digest()[:8], "little")
    generator = torch.Generator(device="cpu").manual_seed(matrix_seed)
    epsilon = torch.randn(2, generator=generator) * (config.sigma / (model.q_proj.in_features**0.5))
    expected_delta = (u * epsilon.unsqueeze(0)) @ vh
    with perturber.sampled(19):
        actual_delta = model.q_proj.weight - before
        assert torch.allclose(actual_delta, expected_delta, atol=1e-7)
        assert torch.linalg.matrix_rank(actual_delta).item() <= config.rank
