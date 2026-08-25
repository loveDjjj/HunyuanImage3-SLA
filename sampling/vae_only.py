"""VAE-only Hunyuan components for offline latent extraction.

The facade intentionally exposes only tokenizer/image preprocessing methods used to
build static conditions.  It never creates the 80B Transformer backbone.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import GenerationConfig

from common.hunyuan import infer_hunyuan_model_version, redirect_legacy_cuda_empty


def _infer_model_version(raw_config: dict, model_path: Path) -> str:
    return infer_hunyuan_model_version(raw_config, model_path)


def _load_local_config(model_path: Path):
    """Parse checkpoint metadata with the checked-out upstream config class.

    Released Instruct-Distil checkpoints can omit ``model_version`` and bundle
    older remote-code files. Loading through AutoConfig then creates a config
    incompatible with the current tokenizer. The checkpoint JSON is data, so it
    is safe to parse it with the upstream revision used by this project.
    """
    from hunyuan_image_3.configuration_hunyuan_image_3 import HunyuanImage3Config

    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Hunyuan config does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        raw_config = json.load(handle)
    raw_config["model_version"] = _infer_model_version(raw_config, model_path)
    return HunyuanImage3Config.from_dict(raw_config)


def _construct_vae(vae_class, vae_config, device: torch.device):
    # AutoencoderKLConv3D currently creates a decode-only empty sentinel on
    # literal CUDA. Keep the compatibility workaround local to construction.
    with redirect_legacy_cuda_empty(device):
        return vae_class.from_config(vae_config)


class VAEOnlyHunyuan:
    def __init__(self, config, tokenizer, image_processor, vae, generation_config):
        self.config = config
        self._tokenizer = tokenizer
        self.image_processor = image_processor
        self.vae = vae
        self.generation_config = generation_config


def _bind_preprocessing_methods() -> None:
    from hunyuan_image_3.modeling_hunyuan_image_3 import HunyuanImage3ForCausalMM

    for name in (
        "check_inputs", "_validate_and_batchify_text", "_validate_and_batchify_image",
        "prepare_message_list", "preprocess_inputs", "build_batch_rope_image_info",
    ):
        setattr(VAEOnlyHunyuan, name, HunyuanImage3ForCausalMM.__dict__[name])


def _load_vae_weights(vae: torch.nn.Module, model_path: Path) -> None:
    targets = set(vae.state_dict())
    loaded: set[str] = set()
    for shard in sorted(model_path.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for source_name in handle.keys():
                name = source_name.removeprefix("vae.")
                if name in targets:
                    with torch.no_grad():
                        vae.state_dict()[name].copy_(handle.get_tensor(source_name))
                    loaded.add(name)
    missing = targets - loaded
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise RuntimeError(f"VAE checkpoint is incomplete: {len(missing)} missing tensors, including {preview}")


def load_vae_only(model_path: str, device: torch.device, dtype: str) -> VAEOnlyHunyuan:
    """Load tokenizer, image processor and VAE without Transformer/MoE weights."""
    from hunyuan_image_3.autoencoder_kl_3d import AutoencoderKLConv3D
    from hunyuan_image_3.image_processor import HunyuanImage3ImageProcessor
    from hunyuan_image_3.tokenization_hunyuan_image_3 import HunyuanImage3TokenizerFast

    root = Path(model_path)
    config = _load_local_config(root)
    tokenizer = HunyuanImage3TokenizerFast.from_pretrained(root, model_version=config.model_version)
    if not isinstance(tokenizer, HunyuanImage3TokenizerFast):
        raise TypeError(f"Expected HunyuanImage3TokenizerFast, got {type(tokenizer).__name__}")
    image_processor = HunyuanImage3ImageProcessor(config)
    vae = _construct_vae(AutoencoderKLConv3D, config.vae, device)
    _load_vae_weights(vae, root)
    vae = vae.to(device=device, dtype={"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    try:
        generation_config = GenerationConfig.from_pretrained(root)
    except OSError:
        generation_config = SimpleNamespace(max_length=4096, sequence_template="pretrain", drop_think=False)
    _bind_preprocessing_methods()
    return VAEOnlyHunyuan(config, tokenizer, image_processor, vae, generation_config)
