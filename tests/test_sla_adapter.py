import sys
import types

import pytest
import torch
from torch import nn

from train.sla_adapter import (
    HunyuanImage3SLAAttention,
    SLAReplacementManager,
    _configure_training_backend,
)


def repeat_kv(tensor, groups):
    return tensor.repeat_interleave(groups, dim=1)


class _SparseLinearAttention(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.proj_l = nn.Linear(1, 1)

    def forward(self, query, key, value):
        return query


class HunyuanImage3SDPAAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = nn.Linear(4, 8)
        self.o_proj = nn.Linear(4, 4)
        self.num_heads = 2
        self.num_key_value_heads = 1
        self.num_key_value_groups = 2
        self.head_dim = 2
        self.hidden_size = 4
        self.hidden_size_q = 4
        self.hidden_size_kv = 2
        self.use_rotary_pos_emb = False
        self.use_qk_norm = False
        self.layer_idx = 0
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


def test_qkv_o_deltas_are_zero_initialized_and_student_only(monkeypatch):
    layers = types.ModuleType("mindiesd.layers")
    layers.SparseLinearAttention = _SparseLinearAttention
    package = types.ModuleType("mindiesd")
    package.layers = layers
    monkeypatch.setitem(sys.modules, "mindiesd", package)
    monkeypatch.setitem(sys.modules, "mindiesd.layers", layers)

    model = _Model()
    manager = SLAReplacementManager(
        model,
        topk=0.125,
        blkq=64,
        blkk=128,
        use_bf16=True,
        trainable_components=("proj_l", "qkv_delta", "o_delta"),
    )
    wrapper = model.attn
    assert torch.count_nonzero(wrapper.qkv_delta.weight) == 0
    assert torch.count_nonzero(wrapper.o_delta.weight) == 0
    assert set(manager.trainable_parameter_groups()) == {"proj_l", "qkv_delta", "o_delta"}

    with manager.dense_teacher():
        teacher, _, _ = wrapper(torch.zeros(1, 1, 4))
    torch.testing.assert_close(teacher, torch.ones_like(teacher))

    with torch.no_grad():
        wrapper.qkv_delta.weight.fill_(0.25)
        wrapper.o_delta.weight.fill_(0.5)
    student, _, _ = wrapper(torch.ones(1, 1, 4))
    assert not torch.equal(student, teacher)


def test_training_backend_can_force_triton(monkeypatch):
    calls = []
    sparse_module = types.ModuleType("mindiesd.layers.flash_attn.sparse_linear_attn")
    sparse_module._triton_shape_supported = lambda head_dim, blkq, blkk: (head_dim, blkq, blkk) == (2, 64, 128)

    def original(*args, **kwargs):
        return "ascendc"

    sparse_module._resolve_sparse_attn_backend = original
    flash_module = types.ModuleType("mindiesd.layers.flash_attn")
    layers = types.ModuleType("mindiesd.layers")

    class RecordingSLA(_SparseLinearAttention):
        def __init__(self, **kwargs):
            calls.append(sparse_module._resolve_sparse_attn_backend(2, 64, 128))
            super().__init__(**kwargs)

    layers.SparseLinearAttention = RecordingSLA
    package = types.ModuleType("mindiesd")
    package.layers = layers
    monkeypatch.setitem(sys.modules, "mindiesd", package)
    monkeypatch.setitem(sys.modules, "mindiesd.layers", layers)
    monkeypatch.setitem(sys.modules, "mindiesd.layers.flash_attn", flash_module)
    monkeypatch.setitem(sys.modules, "mindiesd.layers.flash_attn.sparse_linear_attn", sparse_module)

    SLAReplacementManager(
        _Model(),
        topk=0.125,
        blkq=64,
        blkk=128,
        use_bf16=True,
        training_backend="triton",
    )
    assert calls == ["triton"]


def test_known_910c_triton_ub_overflow_shape_is_rejected():
    with pytest.raises(ValueError, match="UB limit"):
        _configure_training_backend("triton", head_dim=128, blkq=64, blkk=128)
