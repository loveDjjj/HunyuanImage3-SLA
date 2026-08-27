import torch
from torch import nn
import yaml
from pathlib import Path

from train.lora import LowRankDelta, MoEDownLoRAManager


def test_low_rank_delta_uses_random_a_and_zero_b():
    delta = LowRankDelta(4, 6, rank=2, alpha=2)
    assert torch.count_nonzero(delta.a.weight) > 0
    assert torch.count_nonzero(delta.b.weight) == 0
    assert torch.count_nonzero(delta(torch.randn(3, 4))) == 0
    assert sum(parameter.numel() for parameter in delta.parameters()) == 20


class HunyuanMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_and_up_proj = nn.Linear(4, 6, bias=False)
        self.down_proj = nn.Linear(3, 4, bias=False)

    def forward(self, value):
        return self.down_proj(value)


class HunyuanMoE(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.gate = nn.Linear(4, 2, bias=False)
        self.shared_mlp = HunyuanMLP()
        self.experts = nn.ModuleList([HunyuanMLP(), HunyuanMLP()])


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([HunyuanMoE(0), HunyuanMoE(1)])


def test_moe_manager_wraps_only_routed_expert_down_projections():
    model = Model()
    shared = [layer.shared_mlp.down_proj for layer in model.layers]
    inputs = torch.randn(2, 3)
    expected = model.layers[0].experts[0].down_proj(inputs)
    manager = MoEDownLoRAManager(
        model,
        rank=2,
        alpha=2,
        expected_layers=2,
        expected_experts_per_layer=2,
        expected_in_features=3,
        expected_out_features=4,
    )

    assert [layer.shared_mlp.down_proj for layer in model.layers] == shared
    assert manager.geometry.parameter_count == 4 * 2 * (3 + 4)
    assert sum(parameter.numel() for parameter in manager.parameters()) == 56
    torch.testing.assert_close(model.layers[0].experts[0].down_proj(inputs), expected)

    with torch.no_grad():
        model.layers[0].experts[0].down_proj.lora.b.weight.fill_(1)
    assert not torch.equal(model.layers[0].experts[0].down_proj(inputs), expected)
    with manager.disabled():
        torch.testing.assert_close(model.layers[0].experts[0].down_proj(inputs), expected)


def test_moe_manager_rejects_geometry_mismatch():
    try:
        MoEDownLoRAManager(
            Model(),
            rank=2,
            alpha=2,
            expected_layers=2,
            expected_experts_per_layer=64,
            expected_in_features=3,
            expected_out_features=4,
        )
    except RuntimeError as error:
        assert "expert audit failed" in str(error)
    else:
        raise AssertionError("Expected strict MoE geometry audit to fail")


def test_recommended_configuration_has_expected_155m_parameters():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/train_sla_attention_moe_lora.yaml").read_text())
    attention_rank = config["sla"]["attention_lora_rank"]
    moe = config["moe_lora"]
    proj_l = 32 * (128 * 128 + 128)
    attention = 32 * attention_rank * ((4096 + 6144) + (4096 + 4096))
    experts = (
        moe["expected_layers"]
        * moe["expected_experts_per_layer"]
        * moe["rank"]
        * (moe["expected_in_features"] + moe["expected_out_features"])
    )
    assert proj_l == 528_384
    assert attention == 37_748_736
    assert experts == 117_440_512
    assert proj_l + attention + experts == 155_717_632
