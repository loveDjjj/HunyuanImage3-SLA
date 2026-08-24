import sys
from types import SimpleNamespace

import torch

from common.hunyuan import load_hunyuan, redirect_legacy_cuda_empty


def test_legacy_cuda_empty_is_redirected_to_requested_device():
    with redirect_legacy_cuda_empty(torch.device("cpu")):
        value = torch.empty(0, device="cuda")

    assert value.device.type == "cpu"


def test_loader_forwards_upstream_skip_modules(monkeypatch):
    calls = []

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            calls.append((model_path, kwargs))
            return torch.nn.Linear(1, 1)

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModelForCausalLM=FakeAutoModel))

    load_hunyuan("checkpoint", None, "bf16", skip_load_modules=("vae", "vit"))

    assert calls[0][1]["skip_load_module"] == {"vae", "vit"}
