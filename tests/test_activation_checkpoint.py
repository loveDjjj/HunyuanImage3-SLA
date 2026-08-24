import torch
from torch import nn

from common.activation_checkpoint import (
    ActivationCheckpointWrapper,
    enable_hunyuan_activation_checkpointing,
)


class HunyuanImage3DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.calls = 0

    def forward(self, hidden_states, scale=1.0):
        self.calls += 1
        torch.cuda.set_device(hidden_states.device.index or 0)
        with torch.cuda.nvtx.range("MoE"):
            return (self.proj(hidden_states) * scale,)


def test_decoder_layer_is_recomputed_during_backward():
    model = nn.Module()
    model.layers = nn.ModuleList([HunyuanImage3DecoderLayer()])

    assert enable_hunyuan_activation_checkpointing(model) == 1
    assert isinstance(model.layers[0], ActivationCheckpointWrapper)

    inputs = torch.randn(2, 4, requires_grad=True)
    model.layers[0](inputs, scale=2.0)[0].sum().backward()

    assert model.layers[0].module.calls == 2
    assert inputs.grad is not None


def test_no_grad_teacher_does_not_checkpoint():
    layer = HunyuanImage3DecoderLayer()
    wrapper = ActivationCheckpointWrapper(layer)

    with torch.no_grad():
        wrapper(torch.randn(2, 4))

    assert layer.calls == 1
