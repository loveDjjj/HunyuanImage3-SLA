from pathlib import Path

import torch

from sampling.vae_only import _construct_vae, _infer_model_version


def test_distilled_config_without_model_version_uses_instruct_tokenizer():
    config = {"model_type": "hunyuan_image_3_moe", "cfg_distilled": True}
    assert _infer_model_version(config, Path("HunyuanImage-3.0-Instruct-Distil")) == "HunyuanImage-3.0-Instruct"


def test_explicit_model_version_is_preserved():
    config = {"model_version": "custom-version", "cfg_distilled": True}
    assert _infer_model_version(config, Path("checkpoint")) == "custom-version"


def test_base_checkpoint_uses_base_tokenizer_layout():
    config = {"model_type": "hunyuan_image_3_moe", "cfg_distilled": False}
    assert _infer_model_version(config, Path("HunyuanImage-3.0")) == "HunyuanImage-3.0"


def test_legacy_cuda_sentinel_is_redirected_during_vae_construction():
    class LegacyVAE:
        @classmethod
        def from_config(cls, config):
            instance = cls()
            instance.empty_cache = torch.empty(0, device="cuda")
            return instance

    original_empty = torch.empty
    vae = _construct_vae(LegacyVAE, {}, torch.device("cpu"))
    assert vae.empty_cache.device.type == "cpu"
    assert torch.empty is original_empty
