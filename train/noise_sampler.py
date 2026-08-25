"""Deterministic flow-matching perturbation for SLA recovery inputs."""

from __future__ import annotations

import hashlib

import torch


def sample_seed(global_seed: int, sample_id: str, epoch: int, view: int) -> int:
    payload = f"{global_seed}:{sample_id}:{epoch}:{view}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def flow_match_input(z0: torch.Tensor, seed: int, sigma_min: float, sigma_max: float, train_timesteps: int):
    """Return x_t plus Hunyuan MeanFlow's ordered ``t`` and ``r`` timesteps.

    The range is explicit in config because the released upstream scheduler exposes
    inference stepping but no training ``add_noise`` method.
    """
    generator = torch.Generator(device=z0.device).manual_seed(seed)
    sigma = torch.empty((), device=z0.device).uniform_(sigma_min, sigma_max, generator=generator)
    ratio = torch.rand((), device=z0.device, generator=generator)
    sigma_r = sigma_min + ratio * (sigma - sigma_min)
    noise = torch.randn(z0.shape, device=z0.device, dtype=z0.dtype, generator=generator)
    return (
        (1 - sigma) * z0 + sigma * noise,
        sigma * train_timesteps,
        sigma_r * train_timesteps,
    )


def flow_match_batch(
    z0: torch.Tensor,
    sample_ids: list[str],
    *,
    global_seed: int,
    epoch: int,
    view: int,
    sigma_min: float,
    sigma_max: float,
    train_timesteps: int,
):
    if z0.ndim != 4 or z0.shape[0] != len(sample_ids):
        raise ValueError(f"Expected batched latents matching sample_ids; got {z0.shape=} and {len(sample_ids)=}.")
    samples = [
        flow_match_input(
            latent,
            sample_seed(global_seed, sample_id, epoch, view),
            sigma_min,
            sigma_max,
            train_timesteps,
        )
        for latent, sample_id in zip(z0, sample_ids)
    ]
    x_t, timestep, timestep_r = zip(*samples)
    return torch.stack(x_t), torch.stack(timestep), torch.stack(timestep_r)
