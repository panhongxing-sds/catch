"""Appendix A.2.2 low-rank Gaussian weight perturbation."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class WeightPerturbationConfig:
    rank: int = 4
    sigma: float = 0.03
    target_suffixes: tuple[str, ...] = ("q_proj", "k_proj")


class LowRankGaussianPerturber:
    """Apply W' = W + U_r diag(epsilon) V_r^T and restore exactly afterward."""

    def __init__(self, model, config: WeightPerturbationConfig | None = None):
        import torch

        self.model = model
        self.config = config or WeightPerturbationConfig()
        self._modules: dict[str, object] = {}
        self._bases: dict[str, tuple[object, object]] = {}
        self._originals: dict[str, object] = {}
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            if not any(name.endswith(s) for s in self.config.target_suffixes):
                continue
            weight = module.weight.detach().float()
            u, _, vh = torch.linalg.svd(weight, full_matrices=False)
            rank = min(self.config.rank, u.shape[1], vh.shape[0])
            self._modules[name] = module
            self._originals[name] = module.weight.detach().cpu().clone()
            self._bases[name] = (
                u[:, :rank].cpu(),
                vh[:rank].cpu(),
            )
        if not self._modules:
            raise ValueError("No q_proj/k_proj linear layers were found in the model")

    @property
    def target_names(self) -> tuple[str, ...]:
        return tuple(self._modules)

    @contextmanager
    def sampled(self, seed: int) -> Iterator[None]:
        import torch

        def deltas():
            for name, module in self._modules.items():
                device = module.weight.device
                matrix_seed = int.from_bytes(
                    hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "little"
                )
                generator = torch.Generator(device=device)
                generator.manual_seed(matrix_seed)
                u_cpu, vh_cpu = self._bases[name]
                u = u_cpu.to(device=device, dtype=torch.float32)
                vh = vh_cpu.to(device=device, dtype=torch.float32)
                scale = self.config.sigma / (module.in_features**0.5)
                epsilon = torch.randn(
                    u.shape[1], generator=generator, device=device, dtype=torch.float32
                ) * scale
                delta = ((u * epsilon.unsqueeze(0)) @ vh).to(dtype=module.weight.dtype)
                yield module, delta

        try:
            with torch.no_grad():
                for module, delta in deltas():
                    module.weight.add_(delta)
            yield
        finally:
            with torch.no_grad():
                for name, module in self._modules.items():
                    module.weight.copy_(
                        self._originals[name].to(
                            device=module.weight.device, dtype=module.weight.dtype
                        )
                    )
