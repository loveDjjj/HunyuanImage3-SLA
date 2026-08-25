"""Runtime replacement of HunyuanImage-3.0 Dense GQA attention with MindIE-SD SLA."""

from __future__ import annotations

import contextlib
import importlib
from dataclasses import dataclass
from typing import Iterator, Optional

import torch
from torch import nn


def _configure_training_backend(backend: str, *, head_dim: int, blkq: int, blkk: int) -> None:
    """Force MindIE's differentiable Triton path for attention adaptation."""
    if backend == "auto":
        return
    if backend != "triton":
        raise ValueError(f"training_backend must be 'auto' or 'triton', got {backend!r}")
    module = importlib.import_module("mindiesd.layers.flash_attn.sparse_linear_attn")
    supported = getattr(module, "_triton_shape_supported")
    if not supported(head_dim, blkq, blkk):
        raise ValueError(
            f"Triton SLA does not support head_dim={head_dim}, BLKQ={blkq}, BLKK={blkk}."
        )
    original = getattr(module, "_sla_original_backend_resolver", module._resolve_sparse_attn_backend)
    module._sla_original_backend_resolver = original

    def resolve(candidate_head_dim, candidate_blkq, candidate_blkk, *, where=""):
        if supported(candidate_head_dim, candidate_blkq, candidate_blkk):
            return "triton"
        return original(candidate_head_dim, candidate_blkq, candidate_blkk, where=where)

    module._resolve_sparse_attn_backend = resolve


def _is_hunyuan_dense_attention(module: nn.Module) -> bool:
    """Match the upstream attention by its public structure, not by import identity."""
    return (
        module.__class__.__name__ == "HunyuanImage3SDPAAttention"
        and all(hasattr(module, name) for name in ("qkv_proj", "o_proj", "num_heads", "num_key_value_heads"))
    )


class HunyuanImage3SLAAttention(nn.Module):
    """Adapter preserving upstream projections, RoPE, QK norm and GQA expansion.

    MindIE-SD accepts B,H,S,D tensors. Hunyuan's eight KV heads are expanded with
    the original upstream ``repeat_kv`` before the SLA call, yielding 32 heads.
    """

    def __init__(
        self,
        dense_attention: nn.Module,
        *,
        topk: float,
        blkq: int,
        blkk: int,
        use_bf16: bool,
        training_backend: str = "auto",
        trainable_components: tuple[str, ...] = ("proj_l",),
    ):
        super().__init__()
        valid_components = {"proj_l", "qkv_delta", "o_delta"}
        unknown = set(trainable_components) - valid_components
        if unknown:
            raise ValueError(f"Unknown SLA trainable components: {sorted(unknown)}")
        if "proj_l" not in trainable_components:
            raise ValueError("SLA training requires the proj_l component.")
        self.dense_attention = dense_attention
        self.trainable_components = tuple(trainable_components)
        self._attention_mode = "sla"
        self._upstream_module = importlib.import_module(dense_attention.__class__.__module__)
        from mindiesd.layers import SparseLinearAttention

        _configure_training_backend(
            training_backend,
            head_dim=dense_attention.head_dim,
            blkq=blkq,
            blkk=blkk,
        )

        self.sla = SparseLinearAttention(
            head_dim=dense_attention.head_dim,
            topk=topk,
            BLKQ=blkq,
            BLKK=blkk,
            use_bf16=use_bf16,
        )
        self.qkv_delta = None
        self.o_delta = None
        if "qkv_delta" in trainable_components:
            self.qkv_delta = nn.Linear(
                dense_attention.hidden_size,
                dense_attention.hidden_size_q + 2 * dense_attention.hidden_size_kv,
                bias=False,
                device=dense_attention.qkv_proj.weight.device,
                dtype=dense_attention.qkv_proj.weight.dtype,
            )
            nn.init.zeros_(self.qkv_delta.weight)
        if "o_delta" in trainable_components:
            self.o_delta = nn.Linear(
                dense_attention.hidden_size_q,
                dense_attention.hidden_size,
                bias=False,
                device=dense_attention.o_proj.weight.device,
                dtype=dense_attention.o_proj.weight.dtype,
            )
            nn.init.zeros_(self.o_delta.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: Optional[bool] = False,
        custom_pos_emb=None,
        **kwargs,
    ):
        if self._attention_mode == "dense":
            return self.dense_attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                custom_pos_emb=custom_pos_emb,
                **kwargs,
            )
        if output_attentions:
            raise NotImplementedError("SLA recovery training does not return attention weights.")
        if attention_mask is not None:
            raise ValueError(
                "MindIE-SD SparseLinearAttention has no arbitrary attention-mask input. "
                "Provide diffusion training batches with attention_mask=None."
            )

        dense = self.dense_attention
        bsz, q_len, _ = hidden_states.size()
        qkv_states = dense.qkv_proj(hidden_states)
        if self.qkv_delta is not None:
            qkv_states = qkv_states + self.qkv_delta(hidden_states)
        qkv_states = qkv_states.reshape(
            bsz, q_len, dense.num_key_value_heads, dense.num_key_value_groups + 2, dense.head_dim
        )
        query_states, key_states, value_states = torch.split(
            qkv_states, [dense.num_key_value_groups, 1, 1], dim=3
        )
        query_states = query_states.reshape(bsz, q_len, dense.num_heads, dense.head_dim).transpose(1, 2)
        key_states = key_states.reshape(bsz, q_len, dense.num_key_value_heads, dense.head_dim).transpose(1, 2)
        value_states = value_states.reshape(bsz, q_len, dense.num_key_value_heads, dense.head_dim).transpose(1, 2)

        if dense.use_rotary_pos_emb:
            cos, sin = custom_pos_emb
            query_states, key_states = self._upstream_module.apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )
        if dense.use_qk_norm:
            query_states = dense.query_layernorm(query_states)
            key_states = dense.key_layernorm(key_states)

        query_states = query_states.to(value_states.dtype)
        key_states = key_states.to(value_states.dtype)
        if past_key_value is not None:
            cache_kwargs = {"cache_position": position_ids}
            key_states, value_states = past_key_value.update(key_states, value_states, dense.layer_idx, cache_kwargs)
            query_states = query_states.to(key_states.dtype)

        # This is intentionally the upstream GQA expansion: 8 KV heads -> 32 heads.
        key_states = self._upstream_module.repeat_kv(key_states, dense.num_key_value_groups)
        value_states = self._upstream_module.repeat_kv(value_states, dense.num_key_value_groups)
        attn_output = self.sla(query_states, key_states, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        projected = dense.o_proj(attn_output)
        if self.o_delta is not None:
            projected = projected + self.o_delta(attn_output)
        return projected, None, past_key_value


@dataclass
class _Replacement:
    parent: nn.Module
    attribute: str
    dense: nn.Module
    sla: HunyuanImage3SLAAttention


class SLAReplacementManager:
    """Installs SLA modules and can temporarily restore Dense attention for the teacher."""

    def __init__(
        self,
        model: nn.Module,
        *,
        topk: float,
        blkq: int,
        blkk: int,
        use_bf16: bool,
        training_backend: str = "auto",
        trainable_components: tuple[str, ...] = ("proj_l",),
    ):
        self.model = model
        self.replacements: list[_Replacement] = []
        self._install(
            topk=topk,
            blkq=blkq,
            blkk=blkk,
            use_bf16=use_bf16,
            training_backend=training_backend,
            trainable_components=tuple(trainable_components),
        )
        if not self.replacements:
            raise RuntimeError("No HunyuanImage3SDPAAttention modules were found in the loaded model.")

    def _install(self, **sla_kwargs) -> None:
        # Take a snapshot before mutation. The wrapper owns the Dense module, so
        # mutating while recursively walking ``modules()`` would replace it again.
        targets = [
            (parent, attribute, child)
            for parent in list(self.model.modules())
            for attribute, child in list(parent.named_children())
            if _is_hunyuan_dense_attention(child)
        ]
        for parent, attribute, child in targets:
            sla = HunyuanImage3SLAAttention(child, **sla_kwargs)
            setattr(parent, attribute, sla)
            self.replacements.append(_Replacement(parent, attribute, child, sla))

    @contextlib.contextmanager
    def dense_teacher(self) -> Iterator[None]:
        previous_modes = [item.sla._attention_mode for item in self.replacements]
        for item in self.replacements:
            item.sla._attention_mode = "dense"
        try:
            yield
        finally:
            for item, mode in zip(self.replacements, previous_modes):
                item.sla._attention_mode = mode

    def trainable_parameters(self):
        for parameters in self.trainable_parameter_groups().values():
            yield from parameters

    def trainable_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        groups: dict[str, list[nn.Parameter]] = {"proj_l": []}
        for item in self.replacements:
            groups["proj_l"].extend(item.sla.sla.proj_l.parameters())
            if item.sla.qkv_delta is not None:
                groups.setdefault("qkv_delta", []).extend(item.sla.qkv_delta.parameters())
            if item.sla.o_delta is not None:
                groups.setdefault("o_delta", []).extend(item.sla.o_delta.parameters())
        return groups

    def trainable_parameter_names(self):
        for module_name, module in self.model.named_modules():
            if isinstance(module, HunyuanImage3SLAAttention):
                for name, _ in module.sla.proj_l.named_parameters():
                    yield f"{module_name}.sla.proj_l.{name}"
                if module.qkv_delta is not None:
                    yield f"{module_name}.qkv_delta.weight"
                if module.o_delta is not None:
                    yield f"{module_name}.o_delta.weight"
