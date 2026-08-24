"""Activation checkpoint wrappers for frozen Hunyuan decoder layers."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class ActivationCheckpointWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        if not torch.is_grad_enabled():
            return self.module(*args, **kwargs)
        return checkpoint(self.module, *args, use_reentrant=False, **kwargs)


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
