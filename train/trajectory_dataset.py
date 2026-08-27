"""Dataset exposing individual steps from verified Dense trajectories."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from safetensors import safe_open
from torch.utils.data import Dataset

from common.trajectory_schema import STEP_COUNT, unpack_bool_mask
from sampling.condition_packer import decode_rope_image_info


class HunyuanTrajectoryDataset(Dataset):
    def __init__(self, root: str, dtype: str = "bf16", max_prompts: int | None = None):
        self.root = Path(root)
        self.compute_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[dtype]
        manifest = self.root / "manifest.jsonl"
        self.rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
        if max_prompts is not None:
            if max_prompts < 1:
                raise ValueError("max_prompts must be positive when provided.")
            self.rows = self.rows[:max_prompts]
        if not self.rows:
            raise RuntimeError(f"No trajectories in {manifest}")
        self.items = [
            (row_index, step)
            for row_index in range(len(self.rows))
            for step in range(STEP_COUNT)
        ]
        self.dropped_for_batching = 0

    def __len__(self):
        return len(self.items)

    def prepare_exact_length_batches(self, batch_size: int, seed: int) -> None:
        if batch_size < 1:
            raise ValueError("Trajectory batch size must be at least 1.")
        buckets: dict[tuple, list[tuple[int, int]]] = {}
        for row_index, row in enumerate(self.rows):
            sample_dir = self.root / row["path"]
            metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
            key = (
                tuple(metadata["attention_mask_shape"]),
                json.dumps(metadata["rope_image_info"], sort_keys=True),
                json.dumps(metadata["full_attention_spans"], sort_keys=True),
            )
            buckets.setdefault(key, []).extend((row_index, step) for step in range(STEP_COUNT))

        generator = random.Random(seed)
        batches = []
        dropped = 0
        for items in buckets.values():
            generator.shuffle(items)
            usable = len(items) - len(items) % batch_size
            batches.extend(items[index:index + batch_size] for index in range(0, usable, batch_size))
            dropped += len(items) - usable
        generator.shuffle(batches)
        self.items = [item for batch in batches for item in batch]
        self.dropped_for_batching = dropped
        if not self.items:
            raise RuntimeError(f"No exact-layout trajectory batches can be formed with batch_size={batch_size}.")

    def __getitem__(self, index):
        row_index, step = self.items[index]
        row = self.rows[row_index]
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
            "trajectory_step": torch.tensor([step], dtype=torch.long),
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


def collate_trajectory_records(records: list[dict]) -> dict:
    if not records:
        raise ValueError("Cannot collate an empty trajectory batch.")
    result = {}
    tensor_names = (
        "input_ids",
        "position_ids",
        "images",
        "image_mask",
        "timesteps",
        "timesteps_index",
        "timesteps_r",
        "timesteps_r_index",
        "guidance",
        "guidance_index",
        "gen_timestep_scatter_index",
        "attention_mask",
        "teacher_diffusion_prediction",
        "trajectory_step",
    )
    for name in tensor_names:
        result[name] = torch.cat([record[name] for record in records], dim=0)
    result["sample_id"] = [record["sample_id"] for record in records]
    result["rope_image_info"] = [item for record in records for item in record["rope_image_info"]]
    result["full_attention_spans"] = [
        item for record in records for item in record["full_attention_spans"]
    ]
    for name in ("mode", "first_step", "return_dict", "use_cache"):
        values = {record[name] for record in records}
        if len(values) != 1:
            raise ValueError(f"Trajectory batch has inconsistent {name}: {values}")
        result[name] = records[0][name]
    return result
