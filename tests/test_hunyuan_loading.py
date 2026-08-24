import sys
from types import SimpleNamespace

import torch

from common.hunyuan import (
    load_hunyuan,
    prepare_diffusion_runtime,
    redirect_legacy_cuda_empty,
    redirect_legacy_cuda_runtime,
)


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

    assert calls[0][1]["skip_load_module"] == ["vae", "vit"]


def test_legacy_cuda_runtime_calls_are_safe_without_cuda():
    with redirect_legacy_cuda_runtime(torch.device("cpu")):
        torch.cuda.set_device(0)
        with torch.cuda.nvtx.range("MoE"):
            pass


def test_diffusion_runtime_matches_distilled_image_layout():
    model = torch.nn.Module()
    kwargs = {
        "image_mask": torch.ones(1, 4096, dtype=torch.bool),
        "timesteps_index": torch.tensor([[1]]),
        "guidance_index": torch.tensor([[2]]),
        "timesteps_r_index": torch.tensor([[3]]),
    }

    prepare_diffusion_runtime(model, kwargs)

    assert model.post_token_len is None
    assert model.num_image_tokens == 4096
    assert model.num_special_tokens == 3
