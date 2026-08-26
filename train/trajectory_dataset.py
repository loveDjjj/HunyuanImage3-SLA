"""Dataset exposing individual steps from verified Dense trajectories."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from torch.utils.data import Dataset

from common.trajectory_schema import STEP_COUNT, unpack_bool_mask
from sampling.condition_packer import decode_rope_image_info


class HunyuanTrajectoryDataset(Dataset):
    def __init__(self, root: str, dtype: str = "bf16"):
        self.root = Path(root)
        self.compute_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[dtype]
        manifest = self.root / "manifest.jsonl"
        self.rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
        if not self.rows:
            raise RuntimeError(f"No trajectories in {manifest}")

    def __len__(self):
        return len(self.rows) * STEP_COUNT

    def __getitem__(self, index):
        row = self.rows[index // STEP_COUNT]
        step = index % STEP_COUNT
        sample_dir = self.root / row["path"]
        if not (sample_dir / "READY.json").is_file():
            raise FileNotFoundError(f"Trajectory sample is incomplete: {sample_dir}")
        metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
        with safe_open(str(sample_dir / "trajectory.safetensors"), framework="pt", device="cpu") as handle:
            tensors = {
                name: handle.get_tensor(name)
                for name in (
                    "input_ids",
                    "position_ids",
                    "image_mask",
                    "timesteps_index",
                    "guidance_index",
                    "timesteps_r_index",
                    "gen_timestep_scatter_index",
                    "guidance",
                    "attention_mask_packed",
                )
            }
            latent = handle.get_slice("latents")[step:step + 1]
            teacher = handle.get_slice("teacher_predictions")[step:step + 1]
            timestep = handle.get_slice("timesteps")[step:step + 1]
            timestep_r = handle.get_slice("timesteps_r")[step:step + 1]
        return {
            "sample_id": f"{metadata['sample_id']}:step{step}",
            "input_ids": tensors["input_ids"],
            "position_ids": tensors["position_ids"],
            "rope_image_info": decode_rope_image_info(metadata["rope_image_info"]),
            # Artifacts retain FP32 latents for exact scheduler replay. The
            # actual Dense/SLA model runs patch_embed under reduced-precision
            # autocast, so materialize the model-facing input in that dtype.
            "images": latent.to(self.compute_dtype),
            "image_mask": tensors["image_mask"],
            "timesteps": timestep,
            "timesteps_index": tensors["timesteps_index"],
            "timesteps_r": timestep_r,
            "timesteps_r_index": tensors["timesteps_r_index"],
            "guidance": tensors["guidance"],
            "guidance_index": tensors["guidance_index"],
            "gen_timestep_scatter_index": tensors["gen_timestep_scatter_index"],
            "attention_mask": unpack_bool_mask(
                tensors["attention_mask_packed"], metadata["attention_mask_shape"]
            ),
            "full_attention_spans": metadata["full_attention_spans"],
            "mode": "gen_image",
            "first_step": True,
            "return_dict": True,
            "use_cache": False,
            "teacher_diffusion_prediction": teacher,
        }
