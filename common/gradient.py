"""Gradient inspection that works with regular optimizers and DeepSpeed ZeRO-3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch


@dataclass(frozen=True)
class LocalGradientStats:
    element_count: int
    nonfinite_count: int
    squared_norm: float


def deepspeed_local_gradient(parameter: torch.nn.Parameter) -> torch.Tensor | None:
    # ZeRO clears parameter.grad after reduction. DeepSpeed exposes the local
    # partition through this API between backward() and optimizer.step().
    from deepspeed.utils import safe_get_local_grad

    return safe_get_local_grad(parameter)


def inspect_local_gradients(
    parameters: Iterable[torch.nn.Parameter],
    gradient_getter: Callable[[torch.nn.Parameter], torch.Tensor | None] | None = None,
) -> LocalGradientStats:
    getter = gradient_getter or (lambda parameter: parameter.grad)
    element_count = 0
    nonfinite_count = 0
    squared_norm = 0.0

    for parameter in parameters:
        gradient = getter(parameter)
        if gradient is None:
            continue
        gradient = gradient.detach()
        element_count += gradient.numel()
        finite = torch.isfinite(gradient)
        nonfinite_count += int((~finite).sum().item())
        if bool(finite.all().item()):
            squared_norm += float(gradient.float().square().sum().item())

    return LocalGradientStats(element_count, nonfinite_count, squared_norm)
