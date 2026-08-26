import sys
from types import ModuleType, SimpleNamespace

import torch

from common.hunyuan import (
    infer_hunyuan_model_version,
    load_hunyuan,
    patch_remote_generation_contract,
    patch_remote_static_cache,
    prepare_diffusion_runtime,
    redirect_legacy_cuda_empty,
    redirect_legacy_cuda_runtime,
    redirect_remote_vae_cuda_empty,
)


def test_legacy_cuda_empty_is_redirected_to_requested_device():
    with redirect_legacy_cuda_empty(torch.device("cpu")):
        value = torch.empty(0, device="cuda")

    assert value.device.type == "cpu"


def test_loader_forwards_upstream_skip_modules(monkeypatch):
    calls = []

    class FakeAutoConfig:
        @classmethod
        def from_pretrained(cls, _model_path, **_kwargs):
            return SimpleNamespace(auto_map={})

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            calls.append((model_path, kwargs))
            return torch.nn.Linear(1, 1)

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoConfig=FakeAutoConfig, AutoModelForCausalLM=FakeAutoModel),
    )

    load_hunyuan("checkpoint", None, "bf16", skip_load_modules=("vae", "vit"))

    assert calls[0][1]["skip_load_module"] == ["vae", "vit"]


def test_loader_restores_missing_distilled_model_version(monkeypatch):
    calls = []
    config = SimpleNamespace(auto_map={}, cfg_distilled=True)

    class FakeAutoConfig:
        @classmethod
        def from_pretrained(cls, _model_path, **_kwargs):
            return config

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, _model_path, **kwargs):
            calls.append(kwargs)
            return torch.nn.Linear(1, 1)

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoConfig=FakeAutoConfig, AutoModelForCausalLM=FakeAutoModel),
    )

    load_hunyuan("HunyuanImage-3.0-Instruct-Distil", None, "bf16")

    assert config.model_version == "HunyuanImage-3.0-Instruct"
    assert calls[0]["config"] is config


def test_model_version_inference_preserves_explicit_config():
    config = SimpleNamespace(model_version="custom", cfg_distilled=True)
    assert infer_hunyuan_model_version(config, "Instruct-Distil") == "custom"


def test_remote_vae_redirect_survives_global_empty_replacement(monkeypatch):
    model_module = ModuleType("fake_remote.modeling")
    vae_module = ModuleType("fake_remote.autoencoder")

    class FakeModel(torch.nn.Module):
        pass

    class FakeVAE:
        pass

    FakeModel.__module__ = model_module.__name__
    FakeVAE.__module__ = vae_module.__name__
    model_module.AutoencoderKLConv3D = FakeVAE
    vae_module.torch = torch
    monkeypatch.setitem(sys.modules, model_module.__name__, model_module)
    monkeypatch.setitem(sys.modules, vae_module.__name__, vae_module)

    native_empty = torch.empty
    with redirect_remote_vae_cuda_empty(FakeModel, torch.device("cpu")):
        # Simulate ZeRO-3 replacing the process-global constructor after the
        # compatibility context has already been entered.
        monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: native_empty(*args, **kwargs))
        value = vae_module.torch.empty(0, device="cuda")

    assert value.device.type == "cpu"
    assert vae_module.torch is torch


def test_remote_static_cache_initializes_current_transformers_layer(monkeypatch):
    model_module = ModuleType("fake_remote.cache_model")

    class FakeModel(torch.nn.Module):
        pass

    class CurrentLayer:
        def __init__(self):
            self.keys = None
            self.values = None
            self.initialized_with = None

        def lazy_initialization(self, key_states, value_states):
            self.initialized_with = (key_states, value_states)
            self.keys = torch.empty_like(key_states)
            self.values = torch.empty_like(value_states)

    class LegacyHunyuanStaticCache:
        def __init__(self):
            self.layers = [CurrentLayer()]

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            layer = self.layers[layer_idx]
            if layer.keys is None:
                layer.lazy_initialization(key_states)
            layer.keys.copy_(key_states)
            layer.values.copy_(value_states)
            return layer.keys, layer.values

    FakeModel.__module__ = model_module.__name__
    model_module.HunyuanStaticCache = LegacyHunyuanStaticCache
    monkeypatch.setitem(sys.modules, model_module.__name__, model_module)
    model = FakeModel()
    keys = torch.randn(1, 2, 3, 4)
    values = torch.randn(1, 2, 3, 4)

    assert patch_remote_static_cache(model)
    cache = LegacyHunyuanStaticCache()
    cached_keys, cached_values = cache.update(keys, values, 0, {})

    assert cache.layers[0].initialized_with[0] is keys
    assert cache.layers[0].initialized_with[1] is values
    assert torch.equal(cached_keys, keys)
    assert torch.equal(cached_values, values)
    assert not patch_remote_static_cache(model)


def test_remote_static_cache_keeps_legacy_layer_contract(monkeypatch):
    model_module = ModuleType("fake_remote.legacy_cache_model")

    class FakeModel(torch.nn.Module):
        pass

    class LegacyLayer:
        def __init__(self):
            self.keys = None
            self.values = None

        def lazy_initialization(self, key_states):
            self.keys = torch.empty_like(key_states)
            self.values = torch.empty_like(key_states)

    class LegacyCache:
        def __init__(self):
            self.layers = [LegacyLayer()]

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            layer = self.layers[layer_idx]
            if layer.keys is None:
                layer.lazy_initialization(key_states)
            return layer.keys, layer.values

    FakeModel.__module__ = model_module.__name__
    model_module.HunyuanStaticCache = LegacyCache
    monkeypatch.setitem(sys.modules, model_module.__name__, model_module)

    assert patch_remote_static_cache(FakeModel())
    cache = LegacyCache()
    keys = torch.randn(1, 2, 3, 4)
    cache.update(keys, keys, 0, {})
    assert cache.layers[0].keys.shape == keys.shape


def test_remote_generation_preserves_use_cache_and_cache_position():
    class LegacyGenerationModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(use_cache=True)
            self.generation_config = SimpleNamespace(use_cache=True)

        def generate(self, inputs=None, generation_config=None, **kwargs):
            return kwargs

        def _update_model_kwargs_for_generation(
            self, outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1
        ):
            return {"past_key_values": outputs.past_key_values}

    model = LegacyGenerationModel()
    assert patch_remote_generation_contract(model)
    assert model.generate()["use_cache"] is True

    cache_position = torch.arange(4)
    updated = model._update_model_kwargs_for_generation(
        SimpleNamespace(past_key_values="cache"),
        {"use_cache": True, "cache_position": cache_position},
        num_new_tokens=2,
    )
    assert updated["use_cache"] is True
    assert torch.equal(updated["cache_position"], torch.tensor([5]))
    assert updated["past_key_values"] == "cache"
    assert not patch_remote_generation_contract(model)


def test_remote_generation_respects_disabled_cache():
    class LegacyGenerationModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(use_cache=True)
            self.generation_config = SimpleNamespace(use_cache=False)

        def generate(self, **kwargs):
            return kwargs

        def _update_model_kwargs_for_generation(
            self, outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1
        ):
            return {}

    model = LegacyGenerationModel()
    assert patch_remote_generation_contract(model)
    assert model.generate()["use_cache"] is False
    updated = model._update_model_kwargs_for_generation(
        SimpleNamespace(),
        {"use_cache": False, "cache_position": torch.arange(3)},
        num_new_tokens=2,
    )
    assert updated["use_cache"] is False
    assert torch.equal(updated["cache_position"], torch.arange(5))


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
