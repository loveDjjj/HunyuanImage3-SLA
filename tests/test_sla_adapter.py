import sys
import types

import torch
from torch import nn

from train.sla_adapter import HunyuanImage3SLAAttention, SLAReplacementManager


class _SparseLinearAttention(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.proj_l = nn.Linear(1, 1)

    def forward(self, query, key, value):
        return query


class HunyuanImage3SDPAAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = nn.Linear(4, 6)
        self.o_proj = nn.Linear(4, 4)
        self.num_heads = 2
        self.num_key_value_heads = 1
        self.num_key_value_groups = 2
        self.head_dim = 2
        self.dense_calls = 0

    def forward(self, hidden_states, **kwargs):
        self.dense_calls += 1
        return hidden_states + 1, None, None


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = HunyuanImage3SDPAAttention()


def test_dense_teacher_keeps_zero3_module_tree_stable(monkeypatch):
    layers = types.ModuleType("mindiesd.layers")
    layers.SparseLinearAttention = _SparseLinearAttention
    package = types.ModuleType("mindiesd")
    package.layers = layers
    monkeypatch.setitem(sys.modules, "mindiesd", package)
    monkeypatch.setitem(sys.modules, "mindiesd.layers", layers)

    model = _Model()
    original_dense = model.attn
    manager = SLAReplacementManager(model, topk=0.125, blkq=64, blkk=128, use_bf16=True)
    wrapper = model.attn
    assert isinstance(wrapper, HunyuanImage3SLAAttention)

    with manager.dense_teacher():
        assert model.attn is wrapper
        output, _, _ = model.attn(torch.zeros(1, 1, 4))
        assert torch.equal(output, torch.ones(1, 1, 4))
        assert original_dense.dense_calls == 1

    assert model.attn is wrapper
    assert wrapper._attention_mode == "sla"
