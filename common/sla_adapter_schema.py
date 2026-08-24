"""Stable interchange format for HunyuanImage3 SLA recovery weights."""

from __future__ import annotations

import re
from collections.abc import Mapping

import torch


FORMAT_VERSION = 1
ARCHITECTURE = "HunyuanImage3SparseLinearAttentionAdapter"
DEFAULT_NUM_LAYERS = 32
DEFAULT_HEAD_DIM = 128

_CANONICAL_RE = re.compile(r"^layers\.(\d+)\.sla\.proj_l\.(weight|bias)$")
_TRAINING_RE = re.compile(
    r"(?:^|\.)layers\.(\d+)\.(?:module\.)?self_attn\.sla\.proj_l\.(weight|bias)$"
)


def canonical_key(layer: int, parameter: str) -> str:
    if parameter not in {"weight", "bias"}:
        raise ValueError(f"Unsupported SLA parameter: {parameter}")
    return f"layers.{layer}.sla.proj_l.{parameter}"


def canonicalize_training_key(name: str) -> str | None:
    canonical = _CANONICAL_RE.fullmatch(name)
    if canonical:
        return canonical_key(int(canonical.group(1)), canonical.group(2))
    training = _TRAINING_RE.search(name)
    if training:
        return canonical_key(int(training.group(1)), training.group(2))
    return None


def expected_keys(num_layers: int = DEFAULT_NUM_LAYERS) -> set[str]:
    return {
        canonical_key(layer, parameter)
        for layer in range(num_layers)
        for parameter in ("weight", "bias")
    }


def extract_adapter_tensors(
    state_dict: Mapping[str, object],
    *,
    num_layers: int = DEFAULT_NUM_LAYERS,
    head_dim: int = DEFAULT_HEAD_DIM,
) -> dict[str, torch.Tensor]:
    """Select, materialize and validate only SLA proj_l tensors."""
    selected: dict[str, torch.Tensor] = {}
    sources: dict[str, str] = {}
    for source_name, value in state_dict.items():
        target_name = canonicalize_training_key(source_name)
        if target_name is None:
            continue
        if target_name in selected:
            raise ValueError(
                f"Multiple checkpoint parameters map to {target_name}: "
                f"{sources[target_name]} and {source_name}"
            )
        if not isinstance(value, torch.Tensor):
            contiguous = getattr(value, "contiguous", None)
            if contiguous is None:
                raise TypeError(f"Checkpoint value is not tensor-like: {source_name} ({type(value)!r})")
            value = contiguous()
        selected[target_name] = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
        sources[target_name] = source_name

    validate_adapter_tensors(selected, num_layers=num_layers, head_dim=head_dim)
    return dict(sorted(selected.items()))


def validate_adapter_tensors(
    tensors: Mapping[str, torch.Tensor],
    *,
    num_layers: int = DEFAULT_NUM_LAYERS,
    head_dim: int = DEFAULT_HEAD_DIM,
) -> None:
    actual, expected = set(tensors), expected_keys(num_layers)
    missing, unexpected = sorted(expected - actual), sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            f"Invalid SLA adapter keys: missing={missing or 'none'}, "
            f"unexpected={unexpected or 'none'}"
        )

    for name, tensor in tensors.items():
        match = _CANONICAL_RE.fullmatch(name)
        if match is None:
            raise ValueError(f"Invalid canonical SLA adapter key: {name}")
        parameter = match.group(2)
        expected_shape = (head_dim, head_dim) if parameter == "weight" else (head_dim,)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"Invalid shape for {name}: expected {expected_shape}, got {tuple(tensor.shape)}")
        if not tensor.is_floating_point():
            raise ValueError(f"SLA adapter tensor must be floating point: {name} ({tensor.dtype})")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"SLA adapter tensor contains NaN or Inf: {name}")


def parameter_count(tensors: Mapping[str, torch.Tensor]) -> int:
    return sum(tensor.numel() for tensor in tensors.values())
