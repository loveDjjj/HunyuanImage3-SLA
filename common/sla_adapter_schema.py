"""Stable interchange format for HunyuanImage3 SLA recovery weights."""

from __future__ import annotations

import re
from collections.abc import Mapping

import torch


FORMAT_VERSION = 2
ARCHITECTURE = "HunyuanImage3SparseLinearAttentionAdapter"
DEFAULT_NUM_LAYERS = 32
DEFAULT_HEAD_DIM = 128
DEFAULT_HIDDEN_SIZE = 4096
DEFAULT_Q_HEADS = 32
DEFAULT_KV_HEADS = 8
SUPPORTED_COMPONENTS = ("proj_l", "qkv_delta", "o_delta")

_CANONICAL_RE = re.compile(
    r"^layers\.(\d+)\.(sla\.proj_l|qkv_delta|o_delta)\.(weight|bias)$"
)
_TRAINING_RE = re.compile(
    r"(?:^|\.)layers\.(\d+)\.(?:module\.)?self_attn\."
    r"(sla\.proj_l|qkv_delta|o_delta)\.(weight|bias)$"
)


def canonical_key(layer: int, component: str, parameter: str) -> str:
    if component not in SUPPORTED_COMPONENTS:
        raise ValueError(f"Unsupported SLA adapter component: {component}")
    if parameter not in {"weight", "bias"}:
        raise ValueError(f"Unsupported SLA parameter: {parameter}")
    path = "sla.proj_l" if component == "proj_l" else component
    return f"layers.{layer}.{path}.{parameter}"


def canonicalize_training_key(name: str) -> str | None:
    canonical = _CANONICAL_RE.fullmatch(name)
    if canonical:
        component = "proj_l" if canonical.group(2) == "sla.proj_l" else canonical.group(2)
        return canonical_key(int(canonical.group(1)), component, canonical.group(3))
    training = _TRAINING_RE.search(name)
    if training:
        component = "proj_l" if training.group(2) == "sla.proj_l" else training.group(2)
        return canonical_key(int(training.group(1)), component, training.group(3))
    return None


def expected_keys(
    num_layers: int = DEFAULT_NUM_LAYERS,
    components: tuple[str, ...] = ("proj_l",),
) -> set[str]:
    parameters = {
        "proj_l": ("weight", "bias"),
        "qkv_delta": ("weight",),
        "o_delta": ("weight",),
    }
    return {
        canonical_key(layer, component, parameter)
        for layer in range(num_layers)
        for component in components
        for parameter in parameters[component]
    }


def infer_components(tensors: Mapping[str, torch.Tensor]) -> tuple[str, ...]:
    components = []
    names = set(tensors)
    for component in SUPPORTED_COMPONENTS:
        marker = ".sla.proj_l." if component == "proj_l" else f".{component}."
        if any(marker in name for name in names):
            components.append(component)
    return tuple(components)


def extract_adapter_tensors(
    state_dict: Mapping[str, object],
    *,
    num_layers: int = DEFAULT_NUM_LAYERS,
    head_dim: int = DEFAULT_HEAD_DIM,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    q_heads: int = DEFAULT_Q_HEADS,
    kv_heads: int = DEFAULT_KV_HEADS,
) -> dict[str, torch.Tensor]:
    """Select, materialize and validate SLA projection and attention deltas."""
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
        export_dtype = torch.float32 if ".sla.proj_l." in target_name else value.dtype
        selected[target_name] = value.detach().to(device="cpu", dtype=export_dtype).contiguous()
        sources[target_name] = source_name

    validate_adapter_tensors(
        selected,
        num_layers=num_layers,
        head_dim=head_dim,
        hidden_size=hidden_size,
        q_heads=q_heads,
        kv_heads=kv_heads,
    )
    return dict(sorted(selected.items()))


def validate_adapter_tensors(
    tensors: Mapping[str, torch.Tensor],
    *,
    num_layers: int = DEFAULT_NUM_LAYERS,
    head_dim: int = DEFAULT_HEAD_DIM,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    q_heads: int = DEFAULT_Q_HEADS,
    kv_heads: int = DEFAULT_KV_HEADS,
    components: tuple[str, ...] | None = None,
) -> None:
    components = infer_components(tensors) if components is None else tuple(components)
    if not components or components[0] != "proj_l":
        raise ValueError(f"SLA adapter must include proj_l; got components={components}")
    unknown = set(components) - set(SUPPORTED_COMPONENTS)
    if unknown:
        raise ValueError(f"Unsupported SLA adapter components: {sorted(unknown)}")
    actual, expected = set(tensors), expected_keys(num_layers, components)
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
        path, parameter = match.group(2), match.group(3)
        component = "proj_l" if path == "sla.proj_l" else path
        qkv_size = head_dim * (q_heads + 2 * kv_heads)
        shapes = {
            ("proj_l", "weight"): (head_dim, head_dim),
            ("proj_l", "bias"): (head_dim,),
            ("qkv_delta", "weight"): (qkv_size, hidden_size),
            ("o_delta", "weight"): (hidden_size, q_heads * head_dim),
        }
        expected_shape = shapes[(component, parameter)]
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"Invalid shape for {name}: expected {expected_shape}, got {tuple(tensor.shape)}")
        if not tensor.is_floating_point():
            raise ValueError(f"SLA adapter tensor must be floating point: {name} ({tensor.dtype})")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"SLA adapter tensor contains NaN or Inf: {name}")


def parameter_count(tensors: Mapping[str, torch.Tensor]) -> int:
    return sum(tensor.numel() for tensor in tensors.values())
