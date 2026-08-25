"""Versioned, resumable storage for official Hunyuan Dense trajectories."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file


TRAJECTORY_VERSION = 1
STEP_COUNT = 8
REQUIRED_TENSORS = {
    "latents",
    "teacher_predictions",
    "timesteps",
    "timesteps_r",
    "input_ids",
    "position_ids",
    "image_mask",
    "timesteps_index",
    "guidance_index",
    "guidance",
    "timesteps_r_index",
    "gen_timestep_scatter_index",
    "attention_mask_packed",
    "ar_generated_token_ids",
}


def pack_bool_mask(mask: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
    shape = list(mask.shape)
    flat = mask.detach().to(device="cpu", dtype=torch.bool).reshape(-1)
    padding = (-flat.numel()) % 8
    if padding:
        flat = torch.cat([flat, torch.zeros(padding, dtype=torch.bool)])
    bits = flat.reshape(-1, 8).to(torch.uint8)
    shifts = torch.arange(8, dtype=torch.uint8)
    packed = torch.sum(bits << shifts, dim=1).to(torch.uint8).contiguous()
    return packed, shape


def unpack_bool_mask(packed: torch.Tensor, shape: list[int] | tuple[int, ...]) -> torch.Tensor:
    count = math.prod(shape)
    shifts = torch.arange(8, dtype=torch.uint8)
    bits = ((packed.detach().cpu().reshape(-1, 1) >> shifts) & 1).reshape(-1)
    if bits.numel() < count:
        raise ValueError(f"Packed attention mask has {bits.numel()} bits, expected at least {count}.")
    return bits[:count].to(torch.bool).reshape(tuple(shape))


def validate_trajectory(metadata: Mapping[str, Any], tensors: Mapping[str, torch.Tensor]) -> None:
    if int(metadata.get("trajectory_version", 0)) != TRAJECTORY_VERSION:
        raise ValueError(f"Unsupported trajectory_version={metadata.get('trajectory_version')!r}.")
    if int(metadata.get("step_count", 0)) != STEP_COUNT:
        raise ValueError(f"Expected {STEP_COUNT} trajectory steps, got {metadata.get('step_count')!r}.")
    missing = sorted(REQUIRED_TENSORS - set(tensors))
    if missing:
        raise ValueError(f"Trajectory is missing tensors: {missing}")

    latents = tensors["latents"]
    predictions = tensors["teacher_predictions"]
    if latents.ndim != 4 or latents.shape[0] != STEP_COUNT + 1:
        raise ValueError(f"latents must be [9,C,H,W], got {tuple(latents.shape)}")
    if predictions.shape != (STEP_COUNT, *latents.shape[1:]):
        raise ValueError(
            f"teacher_predictions must be [8,C,H,W] matching latents, got {tuple(predictions.shape)}"
        )
    if latents.dtype != torch.float32:
        raise ValueError(f"latents must be FP32 for exact scheduler replay, got {latents.dtype}")
    if predictions.dtype != torch.float32:
        raise ValueError(f"teacher_predictions must be FP32, got {predictions.dtype}")
    for name in ("timesteps", "timesteps_r"):
        if tensors[name].shape != (STEP_COUNT,) or tensors[name].dtype != torch.float32:
            raise ValueError(f"{name} must be FP32 [8], got {tensors[name].dtype} {tuple(tensors[name].shape)}")

    input_ids = tensors["input_ids"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"input_ids must be [1,L], got {tuple(input_ids.shape)}")
    mask_shape = metadata.get("attention_mask_shape")
    if not isinstance(mask_shape, list) or len(mask_shape) != 4:
        raise ValueError("metadata.attention_mask_shape must contain four dimensions.")
    attention_mask = unpack_bool_mask(tensors["attention_mask_packed"], mask_shape)
    if attention_mask.shape[0] != 1 or attention_mask.shape[-1] != input_ids.shape[1]:
        raise ValueError(
            f"Attention mask/input mismatch: mask={tuple(attention_mask.shape)}, input={tuple(input_ids.shape)}"
        )
    for name, tensor in tensors.items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
            raise ValueError(f"Trajectory tensor contains NaN/Inf: {name}")


def load_trajectory(sample_dir: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    ready_path = sample_dir / "READY.json"
    if not ready_path.is_file():
        raise FileNotFoundError(f"Trajectory is incomplete: {ready_path}")
    metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    tensors = load_file(str(sample_dir / "trajectory.safetensors"), device="cpu")
    validate_trajectory(metadata, tensors)
    return metadata, tensors


def write_trajectory_atomic(
    sample_dir: Path,
    metadata: dict[str, Any],
    tensors: dict[str, torch.Tensor],
    *,
    final_image=None,
) -> None:
    validate_trajectory(metadata, tensors)
    sample_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{sample_dir.name}.incomplete-", dir=sample_dir.parent))
    try:
        serializable = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in tensors.items()
        }
        save_file(serializable, str(temporary / "trajectory.safetensors"))
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if final_image is not None:
            final_image.save(temporary / "final.png")
        (temporary / "READY.json").write_text(
            json.dumps(
                {
                    "trajectory_version": TRAJECTORY_VERSION,
                    "sample_id": metadata["sample_id"],
                    "step_count": STEP_COUNT,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if sample_dir.exists():
            shutil.rmtree(sample_dir)
        os.replace(temporary, sample_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
