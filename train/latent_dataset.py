"""Read-only dataset for verified Hunyuan offline latent caches."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.cache_schema import cache_ready
from sampling.condition_packer import decode_rope_image_info


def unwrap_single_record(records: list[dict]) -> dict:
    """Keep cached tensor shapes while exposing a standard batch size to loaders."""
    if len(records) != 1:
        raise ValueError(f"Expected one cache record per micro batch, got {len(records)}.")
    return records[0]


class HunyuanLatentDataset(Dataset):
    def __init__(self, cache_dir: str, split: str | None = None):
        self.cache_dir = Path(cache_dir)
        self.ready = cache_ready(self.cache_dir)
        manifest = self.cache_dir / "manifest.jsonl"
        self.rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
        if split is not None:
            self.rows = [row for row in self.rows if row.get("split", "train") == split]
        if not self.rows:
            raise RuntimeError(f"No {split or 'any'} records in {manifest}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        prefix = row["tensor_prefix"]
        with safe_open(str(self.cache_dir / row["shard"]), framework="pt", device="cpu") as handle:
            record = {
                "sample_id": row["sample_id"],
                "rope_image_info": decode_rope_image_info(row["rope_image_info"]),
                "height": row["height"],
                "width": row["width"],
            }
            for field in (
                "latent_z0", "input_ids", "image_mask", "timesteps_index",
                "guidance_index", "timesteps_r_index", "gen_timestep_scatter_index",
            ):
                key = f"{prefix}/{field}"
                if key in handle.keys():
                    record[field] = handle.get_tensor(key)
        return record


def model_kwargs_from_latent(record: dict, x_t: torch.Tensor, timestep: torch.Tensor) -> dict:
    """Build the exact model-facing subset without image loading or VAE execution."""
    kwargs = {
        "input_ids": record["input_ids"],
        "rope_image_info": record["rope_image_info"],
        "images": x_t.unsqueeze(0),
        "image_mask": record["image_mask"],
        "timesteps": timestep.reshape(1),
        "timesteps_index": record["timesteps_index"],
        "mode": "gen_image",
        "first_step": True,
        "return_dict": True,
        "gen_timestep_scatter_index": record.get("gen_timestep_scatter_index"),
    }
    for name in ("guidance_index", "timesteps_r_index"):
        if name in record:
            kwargs[name] = record[name]
    return kwargs
