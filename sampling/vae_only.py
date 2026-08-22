"""VAE-only Hunyuan components for offline latent extraction.

The facade intentionally exposes only tokenizer/image preprocessing methods used to
build static conditions.  It never creates the 80B Transformer backbone.
"""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, GenerationConfig


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
    config = AutoConfig.from_pretrained(root, trust_remote_code=True)
    tokenizer = HunyuanImage3TokenizerFast.from_pretrained(root, model_version=config.model_version)
    image_processor = HunyuanImage3ImageProcessor(config)
    vae = AutoencoderKLConv3D.from_config(config.vae)
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
