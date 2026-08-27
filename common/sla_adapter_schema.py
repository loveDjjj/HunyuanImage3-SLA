"""Stable interchange format for HunyuanImage3 SLA recovery weights."""

from __future__ import annotations

import re
from collections.abc import Mapping

import torch


FORMAT_VERSION = 3
ARCHITECTURE = "HunyuanImage3SparseLinearAttentionAdapter"
DEFAULT_NUM_LAYERS = 32
DEFAULT_HEAD_DIM = 128
DEFAULT_HIDDEN_SIZE = 4096
DEFAULT_Q_HEADS = 32
DEFAULT_KV_HEADS = 8
DEFAULT_NUM_EXPERTS = 64
DEFAULT_MOE_INTERMEDIATE_SIZE = 3072
SUPPORTED_COMPONENTS = (
    "proj_l",
    "qkv_delta",
    "o_delta",
    "qkv_lora",
    "o_lora",
    "moe_down_lora",
)

_CANONICAL_RE = re.compile(
    r"^layers\.(\d+)\.(sla\.proj_l|qkv_delta|o_delta)\.(weight|bias)$"
)
_TRAINING_RE = re.compile(
    r"(?:^|\.)layers\.(\d+)\.(?:module\.)?self_attn\."
    r"(sla\.proj_l|qkv_delta|o_delta)\.(weight|bias)$"
)
_LORA_CANONICAL_RE = re.compile(r"^layers\.(\d+)\.(qkv_lora|o_lora)\.(a|b)\.weight$")
_LORA_TRAINING_RE = re.compile(
    r"(?:^|\.)layers\.(\d+)\.(?:module\.)?self_attn\.(qkv_lora|o_lora)\.(a|b)\.weight$"
)
_MOE_CANONICAL_RE = re.compile(
    r"^layers\.(\d+)\.moe\.experts\.(\d+)\.down_lora\.(a|b)\.weight$"
)
_MOE_TRAINING_RE = re.compile(
    r"(?:^|\.)layers\.(\d+)\.(?:module\.)?mlp\.experts\.(\d+)\."
    r"down_proj\.lora\.(a|b)\.weight$"
)


def canonical_key(layer: int, component: str, parameter: str) -> str:
    if component not in SUPPORTED_COMPONENTS:
        raise ValueError(f"Unsupported SLA adapter component: {component}")
    if parameter not in {"weight", "bias"}:
        raise ValueError(f"Unsupported SLA parameter: {parameter}")
    path = "sla.proj_l" if component == "proj_l" else component
    return f"layers.{layer}.{path}.{parameter}"


def canonicalize_training_key(name: str) -> str | None:
    lora = _LORA_CANONICAL_RE.fullmatch(name) or _LORA_TRAINING_RE.search(name)
    if lora:
        return f"layers.{int(lora.group(1))}.{lora.group(2)}.{lora.group(3)}.weight"
    moe = _MOE_CANONICAL_RE.fullmatch(name) or _MOE_TRAINING_RE.search(name)
    if moe:
        return (
            f"layers.{int(moe.group(1))}.moe.experts.{int(moe.group(2))}."
            f"down_lora.{moe.group(3)}.weight"
        )
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
    num_experts: int = DEFAULT_NUM_EXPERTS,
) -> set[str]:
    parameters = {
        "proj_l": ("weight", "bias"),
        "qkv_delta": ("weight",),
        "o_delta": ("weight",),
        "qkv_lora": ("a.weight", "b.weight"),
        "o_lora": ("a.weight", "b.weight"),
    }
    result = {
        (
            f"layers.{layer}.{component}.{parameter}"
            if component in {"qkv_lora", "o_lora"}
            else canonical_key(layer, component, parameter)
        )
        for layer in range(num_layers)
        for component in components
        if component != "moe_down_lora"
        for parameter in parameters[component]
    }
    if "moe_down_lora" in components:
        result.update(
            f"layers.{layer}.moe.experts.{expert}.down_lora.{factor}.weight"
            for layer in range(num_layers)
            for expert in range(num_experts)
            for factor in ("a", "b")
        )
    return result


def infer_components(tensors: Mapping[str, torch.Tensor]) -> tuple[str, ...]:
    components = []
    names = set(tensors)
    for component in SUPPORTED_COMPONENTS:
        marker = (
            ".sla.proj_l."
            if component == "proj_l"
            else ".moe.experts."
            if component == "moe_down_lora"
            else f".{component}."
        )
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
    num_experts: int = DEFAULT_NUM_EXPERTS,
    moe_intermediate_size: int = DEFAULT_MOE_INTERMEDIATE_SIZE,
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
        num_experts=num_experts,
        moe_intermediate_size=moe_intermediate_size,
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
    num_experts: int = DEFAULT_NUM_EXPERTS,
    moe_intermediate_size: int = DEFAULT_MOE_INTERMEDIATE_SIZE,
    components: tuple[str, ...] | None = None,
) -> None:
    components = infer_components(tensors) if components is None else tuple(components)
    if not components or components[0] != "proj_l":
        raise ValueError(f"SLA adapter must include proj_l; got components={components}")
    unknown = set(components) - set(SUPPORTED_COMPONENTS)
    if unknown:
        raise ValueError(f"Unsupported SLA adapter components: {sorted(unknown)}")
    actual, expected = set(tensors), expected_keys(num_layers, components, num_experts)
    missing, unexpected = sorted(expected - actual), sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            f"Invalid SLA adapter keys: missing={missing or 'none'}, "
            f"unexpected={unexpected or 'none'}"
        )

    for name, tensor in tensors.items():
        lora_match = _LORA_CANONICAL_RE.fullmatch(name)
        moe_match = _MOE_CANONICAL_RE.fullmatch(name)
        if lora_match:
            component, factor = lora_match.group(2), lora_match.group(3)
            rank = tensor.shape[0] if factor == "a" else tensor.shape[1]
            qkv_size = head_dim * (q_heads + 2 * kv_heads)
            expected_shape = {
                ("qkv_lora", "a"): (rank, hidden_size),
                ("qkv_lora", "b"): (qkv_size, rank),
                ("o_lora", "a"): (rank, q_heads * head_dim),
                ("o_lora", "b"): (hidden_size, rank),
            }[(component, factor)]
        elif moe_match:
            factor = moe_match.group(3)
            rank = tensor.shape[0] if factor == "a" else tensor.shape[1]
            expected_shape = (
                (rank, moe_intermediate_size)
                if factor == "a"
                else (hidden_size, rank)
            )
        else:
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

    for component in ("qkv_lora", "o_lora"):
        if component in components:
            ranks = {
                tensor.shape[0] if name.endswith("a.weight") else tensor.shape[1]
                for name, tensor in tensors.items()
                if f".{component}." in name
            }
            if len(ranks) != 1:
                raise ValueError(f"{component} rank must be uniform, got {sorted(ranks)}.")
    if "moe_down_lora" in components:
        ranks = {
            tensor.shape[0] if name.endswith("a.weight") else tensor.shape[1]
            for name, tensor in tensors.items()
            if ".moe.experts." in name
        }
        if len(ranks) != 1:
            raise ValueError(f"moe_down_lora rank must be uniform, got {sorted(ranks)}.")


def parameter_count(tensors: Mapping[str, torch.Tensor]) -> int:
    return sum(tensor.numel() for tensor in tensors.values())
