import json

import pytest
import torch
from safetensors.torch import load_file

from tools.create_zero_sla_adapter import create_zero_adapter
from tools.inspect_sla_adapter import inspect_adapter


def test_create_zero_adapter_is_valid_proj_only_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.create_zero_sla_adapter._repository_commit", lambda: "abc123")
    output = tmp_path / "zero"

    config = create_zero_adapter(
        output,
        training_config={
            "model_path": "/models/HunyuanImage-3.0-Instruct-Distil",
            "sla": {"topk": 0.125, "blkq": 128, "blkk": 128, "use_bf16": True},
        },
        num_layers=2,
        head_dim=4,
        hidden_size=8,
        q_heads=2,
        kv_heads=1,
    )

    tensors = load_file(str(output / "adapter.safetensors"), device="cpu")
    assert len(tensors) == 4
    assert all(tensor.dtype == torch.float32 for tensor in tensors.values())
    assert all(torch.count_nonzero(tensor).item() == 0 for tensor in tensors.values())
    assert config["baseline_type"] == "sla_zero_init"
    assert config["trained_components"] == ["proj_l"]
    assert config["parameter_count"] == 40
    assert inspect_adapter(output)["baseline_type"] == "sla_zero_init"


def test_zero_baseline_inspector_rejects_nonzero_tensor(tmp_path):
    output = tmp_path / "zero"
    create_zero_adapter(
        output,
        training_config={"sla": {}},
        num_layers=1,
        head_dim=2,
        hidden_size=4,
        q_heads=2,
        kv_heads=1,
    )
    config = json.loads((output / "adapter_config.json").read_text())
    from safetensors.torch import save_file

    tensors = load_file(str(output / "adapter.safetensors"))
    tensors["layers.0.sla.proj_l.bias"][0] = 1
    save_file(tensors, str(output / "adapter.safetensors"))
    import hashlib

    config["adapter_sha256"] = hashlib.sha256((output / "adapter.safetensors").read_bytes()).hexdigest()
    (output / "adapter_config.json").write_text(json.dumps(config))

    with pytest.raises(ValueError, match="non-zero"):
        inspect_adapter(output)
