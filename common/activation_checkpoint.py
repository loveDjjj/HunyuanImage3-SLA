"""Activation checkpoint wrappers for frozen Hunyuan decoder layers."""

from __future__ import annotations

import contextlib

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from common.hunyuan import redirect_legacy_cuda_runtime
from common.sla_context import current_sla_full_attention_spans, sla_full_attention_spans


@contextlib.contextmanager
def _hunyuan_checkpoint_context(spans):
    with redirect_legacy_cuda_runtime(), sla_full_attention_spans(spans):
        yield


def _hunyuan_checkpoint_contexts():
    spans = current_sla_full_attention_spans()
    return _hunyuan_checkpoint_context(spans), _hunyuan_checkpoint_context(spans)


class ActivationCheckpointWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        if not torch.is_grad_enabled():
            with redirect_legacy_cuda_runtime():
                return self.module(*args, **kwargs)
        return checkpoint(
            self.module,
            *args,
            use_reentrant=False,
            context_fn=_hunyuan_checkpoint_contexts,
            **kwargs,
        )


def enable_hunyuan_activation_checkpointing(model: nn.Module) -> int:
    """Wrap every Hunyuan decoder layer without modifying its implementation."""
    wrapped = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if child.__class__.__name__ != "HunyuanImage3DecoderLayer":
                continue
            setattr(parent, name, ActivationCheckpointWrapper(child))
            wrapped += 1
    if wrapped == 0:
        raise RuntimeError("No HunyuanImage3DecoderLayer modules found for activation checkpointing.")
    return wrapped
