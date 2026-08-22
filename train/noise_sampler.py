"""Deterministic flow-matching perturbation for SLA recovery inputs."""

from __future__ import annotations

import hashlib

import torch


def sample_seed(global_seed: int, sample_id: str, epoch: int, view: int) -> int:
    payload = f"{global_seed}:{sample_id}:{epoch}:{view}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def flow_match_input(z0: torch.Tensor, seed: int, sigma_min: float, sigma_max: float, train_timesteps: int):
    """Return x_sigma=(1-sigma)z0+sigma*eps and Hunyuan's float timestep.

    The range is explicit in config because the released upstream scheduler exposes
    inference stepping but no training ``add_noise`` method.
    """
    generator = torch.Generator(device=z0.device).manual_seed(seed)
    sigma = torch.empty((), device=z0.device).uniform_(sigma_min, sigma_max, generator=generator)
    noise = torch.randn(z0.shape, device=z0.device, dtype=z0.dtype, generator=generator)
    return (1 - sigma) * z0 + sigma * noise, sigma * train_timesteps
