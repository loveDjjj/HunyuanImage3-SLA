"""Explicit student-only LoRA modules for Hunyuan attention and routed experts."""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import nn


class LowRankDelta(nn.Module):
    """A random / B zero LoRA branch with no base-weight ownership."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        alpha: float,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.a = nn.Linear(self.in_features, self.rank, bias=False, device=device, dtype=dtype)
        self.b = nn.Linear(self.rank, self.out_features, bias=False, device=device, dtype=dtype)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.b(self.a(value)) * self.scale


class LoRAInjectedLinear(nn.Module):
    """Preserve a frozen Linear and add a switchable low-rank output delta."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRAInjectedLinear requires nn.Linear, got {type(base)!r}.")
        self.base = base
        self.lora = LowRankDelta(
            base.in_features,
            base.out_features,
            rank=rank,
            alpha=alpha,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        self.lora_enabled = True

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = self.base(value)
        return output + self.lora(value) if self.lora_enabled else output


@dataclass(frozen=True)
class MoELoRAGeometry:
    layers: int
    experts_per_layer: int
    in_features: int
    out_features: int
    rank: int
    alpha: float

    @property
    def parameter_count(self) -> int:
        return (
            self.layers
            * self.experts_per_layer
            * self.rank
            * (self.in_features + self.out_features)
        )


class MoEDownLoRAManager:
    """Install LoRA only on routed Hunyuan MoE expert down projections."""

    def __init__(
        self,
        model: nn.Module,
        *,
        rank: int,
        alpha: float,
        expected_layers: int = 32,
        expected_experts_per_layer: int = 64,
        expected_in_features: int = 3072,
        expected_out_features: int = 4096,
    ) -> None:
        targets: list[tuple[int, int, nn.Module, nn.Linear]] = []
        seen_layers: set[int] = set()
        for module in model.modules():
            if module.__class__.__name__ != "HunyuanMoE":
                continue
            layer = int(getattr(module, "layer_idx", -1))
            experts = getattr(module, "experts", None)
            if layer < 0 or not isinstance(experts, nn.ModuleList):
                raise RuntimeError("HunyuanMoE is missing layer_idx or routed expert ModuleList.")
            if hasattr(module, "shared_mlp") and module.shared_mlp in experts:
                raise RuntimeError("Shared MLP must not be included in routed experts.")
            seen_layers.add(layer)
            for expert_index, expert in enumerate(experts):
                down_proj = getattr(expert, "down_proj", None)
                if not isinstance(down_proj, nn.Linear):
                    raise RuntimeError(
                        f"Layer {layer} expert {expert_index} down_proj must be nn.Linear, "
                        f"got {type(down_proj)!r}."
                    )
                targets.append((layer, expert_index, expert, down_proj))

        expected_layer_ids = set(range(expected_layers))
        if seen_layers != expected_layer_ids:
            raise RuntimeError(
                f"MoE layer audit failed: expected={sorted(expected_layer_ids)}, got={sorted(seen_layers)}."
            )
        per_layer = {layer: 0 for layer in seen_layers}
        shapes = set()
        for layer, _, _, projection in targets:
            per_layer[layer] += 1
            shapes.add((projection.in_features, projection.out_features))
        if set(per_layer.values()) != {expected_experts_per_layer}:
            raise RuntimeError(
                f"MoE expert audit failed: expected {expected_experts_per_layer}/layer, got {per_layer}."
            )
        expected_shape = (expected_in_features, expected_out_features)
        if shapes != {expected_shape}:
            raise RuntimeError(f"MoE down_proj audit failed: expected={expected_shape}, got={sorted(shapes)}.")

        self.wrappers: list[tuple[int, int, LoRAInjectedLinear]] = []
        for layer, expert_index, expert, projection in targets:
            wrapper = LoRAInjectedLinear(projection, rank=rank, alpha=alpha)
            expert.down_proj = wrapper
            self.wrappers.append((layer, expert_index, wrapper))
        self.geometry = MoELoRAGeometry(
            layers=expected_layers,
            experts_per_layer=expected_experts_per_layer,
            in_features=expected_in_features,
            out_features=expected_out_features,
            rank=rank,
            alpha=alpha,
        )

    @contextlib.contextmanager
    def disabled(self) -> Iterator[None]:
        previous = [wrapper.lora_enabled for _, _, wrapper in self.wrappers]
        for _, _, wrapper in self.wrappers:
            wrapper.lora_enabled = False
        try:
            yield
        finally:
            for (_, _, wrapper), enabled in zip(self.wrappers, previous):
                wrapper.lora_enabled = enabled

    def parameters(self) -> Iterator[nn.Parameter]:
        for _, _, wrapper in self.wrappers:
            yield from wrapper.lora.parameters()

    def canonical_parameter_names(self) -> Iterator[str]:
        for layer, expert, wrapper in self.wrappers:
            for factor, parameter in wrapper.lora.named_parameters():
                del parameter
                yield f"layers.{layer}.moe.experts.{expert}.down_lora.{factor}"
