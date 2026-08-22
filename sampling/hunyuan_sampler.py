"""Hunyuan preprocessing and VAE encoding used only by the offline sampling line."""

from __future__ import annotations

from PIL import Image
import torch

from .condition_packer import pack_condition


def preprocess_image(model, image: Image.Image, height: int, width: int) -> torch.Tensor:
    processed = model.image_processor.vae_process_image(image.convert("RGB"), (width, height), random_crop="center")
    return processed.unsqueeze(0)


@torch.no_grad()
def encode_z0(model, image_tensor: torch.Tensor, device: torch.device, dtype: torch.dtype, seed: int) -> torch.Tensor:
    """Encode and apply the upstream VAE latent normalization exactly once."""
    generator = torch.Generator(device=device).manual_seed(seed)
    image_tensor = image_tensor.to(device=device, dtype=dtype)
    result = model.vae.encode(image_tensor)
    latent = result if isinstance(result, torch.Tensor) else result.latent_dist.sample(generator)
    config = model.vae.config
    if getattr(config, "shift_factor", None):
        latent = latent - config.shift_factor
    if getattr(config, "scaling_factor", None):
        latent = latent * config.scaling_factor
    if hasattr(model.vae, "ffactor_temporal"):
        if latent.shape[2] != 1:
            raise ValueError(f"Expected image VAE temporal dimension 1, got {latent.shape}")
        latent = latent.squeeze(2)
    return latent.squeeze(0).detach().cpu()


def sample_record(model, image: Image.Image, caption: str, height: int, width: int, device, dtype, seed: int):
    image_tensor = preprocess_image(model, image, height, width)
    latent = encode_z0(model, image_tensor, device, dtype, seed)
    condition, metadata = pack_condition(model, caption, height, width)
    condition["latent_z0"] = latent
    return condition, metadata
