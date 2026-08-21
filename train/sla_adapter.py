"""Runtime replacement of HunyuanImage-3.0 Dense GQA attention with MindIE-SD SLA."""

from __future__ import annotations

import contextlib
import importlib
from dataclasses import dataclass
from typing import Iterator, Optional

import torch
from torch import nn


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

    def __init__(self, dense_attention: nn.Module, *, topk: float, blkq: int, blkk: int, use_bf16: bool):
        super().__init__()
        self.dense_attention = dense_attention
        self._upstream_module = importlib.import_module(dense_attention.__class__.__module__)
        from mindiesd.layers import SparseLinearAttention

        self.sla = SparseLinearAttention(
            head_dim=dense_attention.head_dim,
            topk=topk,
            BLKQ=blkq,
            BLKK=blkk,
            use_bf16=use_bf16,
        )

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
        if output_attentions:
            raise NotImplementedError("SLA recovery training does not return attention weights.")
        if attention_mask is not None:
            raise ValueError(
                "MindIE-SD SparseLinearAttention has no arbitrary attention-mask input. "
                "Provide diffusion training batches with attention_mask=None."
            )

        dense = self.dense_attention
        bsz, q_len, _ = hidden_states.size()
        qkv_states = dense.qkv_proj(hidden_states).reshape(
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
        return dense.o_proj(attn_output), None, past_key_value


@dataclass
class _Replacement:
    parent: nn.Module
    attribute: str
    dense: nn.Module
    sla: HunyuanImage3SLAAttention


class SLAReplacementManager:
    """Installs SLA modules and can temporarily restore Dense attention for the teacher."""

    def __init__(self, model: nn.Module, *, topk: float, blkq: int, blkk: int, use_bf16: bool):
        self.model = model
        self.replacements: list[_Replacement] = []
        self._install(topk=topk, blkq=blkq, blkk=blkk, use_bf16=use_bf16)
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
        for item in self.replacements:
            setattr(item.parent, item.attribute, item.dense)
        try:
            yield
        finally:
            for item in self.replacements:
                setattr(item.parent, item.attribute, item.sla)

    def trainable_parameters(self):
        for item in self.replacements:
            yield from item.sla.sla.proj_l.parameters()

    def trainable_parameter_names(self):
        for module_name, module in self.model.named_modules():
            if isinstance(module, HunyuanImage3SLAAttention):
                for name, _ in module.sla.proj_l.named_parameters():
                    yield f"{module_name}.sla.proj_l.{name}"
