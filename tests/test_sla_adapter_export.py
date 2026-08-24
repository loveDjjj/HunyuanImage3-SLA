import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from common.sla_adapter_schema import extract_adapter_tensors, parameter_count
from tools.export_sla_adapter import export_adapter, load_training_state
from tools.inspect_sla_adapter import inspect_adapter


def training_state(num_layers=2, head_dim=4):
    state = {"unrelated.weight": torch.ones(1)}
    for layer in range(num_layers):
        prefix = f"model.model.layers.{layer}.module.self_attn.sla.proj_l"
        state[f"{prefix}.weight"] = torch.full((head_dim, head_dim), layer + 1, dtype=torch.bfloat16)
        state[f"{prefix}.bias"] = torch.full((head_dim,), layer + 1, dtype=torch.bfloat16)
    return state


def test_extract_normalizes_training_names_and_preserves_fp32():
    tensors = extract_adapter_tensors(training_state(), num_layers=2, head_dim=4)
    assert set(tensors) == {
        "layers.0.sla.proj_l.weight",
        "layers.0.sla.proj_l.bias",
        "layers.1.sla.proj_l.weight",
        "layers.1.sla.proj_l.bias",
    }
    assert all(tensor.dtype == torch.float32 for tensor in tensors.values())
    assert parameter_count(tensors) == 40


def test_extract_rejects_missing_and_nonfinite_tensors():
    state = training_state()
    state.pop("model.model.layers.1.module.self_attn.sla.proj_l.bias")
    with pytest.raises(ValueError, match="missing"):
        extract_adapter_tensors(state, num_layers=2, head_dim=4)

    state = training_state()
    state["model.model.layers.0.module.self_attn.sla.proj_l.bias"][0] = torch.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        extract_adapter_tensors(state, num_layers=2, head_dim=4)


def test_export_pt_checkpoint_writes_valid_interchange_artifact(tmp_path: Path, monkeypatch):
    checkpoint = tmp_path / "sla-step-12.pt"
    torch.save({"trainable_state_dict": training_state()}, checkpoint)
    output = tmp_path / "adapter"
    monkeypatch.setattr("tools.export_sla_adapter.repository_commit", lambda: "abc123")

    config = export_adapter(
        checkpoint,
        output,
        training_config={
            "model_path": "/models/HunyuanImage-3.0-Instruct-Distil",
            "sla": {"topk": 0.125, "blkq": 64, "blkk": 128, "use_bf16": True},
        },
        num_layers=2,
        head_dim=4,
    )

    assert config["training_step"] == 12
    assert config["tensor_count"] == 4
    assert config["parameter_count"] == 40
    assert (output / "SHA256SUMS").is_file()
    assert len(load_file(str(output / "adapter.safetensors"))) == 4
    assert inspect_adapter(output)["valid"] is True
    saved_config = json.loads((output / "adapter_config.json").read_text())
    assert saved_config["training_repo_commit"] == "abc123"


def test_zero_checkpoint_uses_parent_and_tag_with_lazy_consolidation(tmp_path: Path, monkeypatch):
    checkpoint = tmp_path / "training" / "sla-step-20"
    checkpoint.mkdir(parents=True)
    calls = {}

    def fake_loader(root, **kwargs):
        calls.update(root=root, **kwargs)
        return training_state()

    zero_module = types.ModuleType("deepspeed.utils.zero_to_fp32")
    zero_module.get_fp32_state_dict_from_zero_checkpoint = fake_loader
    utils_module = types.ModuleType("deepspeed.utils")
    utils_module.zero_to_fp32 = zero_module
    deepspeed_module = types.ModuleType("deepspeed")
    deepspeed_module.utils = utils_module
    monkeypatch.setitem(sys.modules, "deepspeed", deepspeed_module)
    monkeypatch.setitem(sys.modules, "deepspeed.utils", utils_module)
    monkeypatch.setitem(sys.modules, "deepspeed.utils.zero_to_fp32", zero_module)

    loaded = load_training_state(checkpoint)
    expected = training_state()
    assert loaded.keys() == expected.keys()
    assert all(torch.equal(loaded[name], expected[name]) for name in expected)
    assert calls == {
        "root": str(checkpoint.parent),
        "tag": "sla-step-20",
        "exclude_frozen_parameters": True,
        "lazy_mode": True,
    }


def test_export_cli_runs_outside_repository_working_directory(tmp_path: Path):
    checkpoint = tmp_path / "sla-step-3.pt"
    output = tmp_path / "adapter"
    torch.save({"trainable_state_dict": training_state()}, checkpoint)
    script = Path(__file__).resolve().parents[1] / "tools" / "export_sla_adapter.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--num-layers",
            "2",
            "--head-dim",
            "4",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout)["parameter_count"] == 40
    assert inspect_adapter(output)["valid"] is True
