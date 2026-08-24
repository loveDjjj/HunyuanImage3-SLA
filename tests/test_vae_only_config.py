from pathlib import Path

from sampling.vae_only import _infer_model_version


def test_distilled_config_without_model_version_uses_instruct_tokenizer():
    config = {"model_type": "hunyuan_image_3_moe", "cfg_distilled": True}
    assert _infer_model_version(config, Path("HunyuanImage-3.0-Instruct-Distil")) == "HunyuanImage-3.0-Instruct"


def test_explicit_model_version_is_preserved():
    config = {"model_version": "custom-version", "cfg_distilled": True}
    assert _infer_model_version(config, Path("checkpoint")) == "custom-version"


def test_base_checkpoint_uses_base_tokenizer_layout():
    config = {"model_type": "hunyuan_image_3_moe", "cfg_distilled": False}
    assert _infer_model_version(config, Path("HunyuanImage-3.0")) == "HunyuanImage-3.0"
