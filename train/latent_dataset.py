"""Read-only dataset for verified Hunyuan offline latent caches."""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
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


def collate_latent_records(records: list[dict]) -> dict:
    if not records:
        raise ValueError("Cannot collate an empty latent micro batch.")
    tensor_fields = {
        "input_ids",
        "image_mask",
        "timesteps_index",
        "guidance_index",
        "timesteps_r_index",
        "gen_timestep_scatter_index",
    }
    result = {
        "sample_id": [str(record["sample_id"]) for record in records],
        "latent_z0": torch.stack([record["latent_z0"] for record in records]),
        "rope_image_info": [item for record in records for item in record["rope_image_info"]],
        "height": [record["height"] for record in records],
        "width": [record["width"] for record in records],
    }
    common_fields = set.intersection(*(set(record) for record in records))
    for field in tensor_fields & common_fields:
        shapes = {tuple(record[field].shape[1:]) for record in records}
        if len(shapes) != 1:
            raise ValueError(f"Latent micro batch field {field!r} has incompatible shapes: {sorted(shapes)}")
        result[field] = torch.cat([record[field] for record in records], dim=0)
    return result


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
        self.dropped_for_batching = 0

    def prepare_exact_length_batches(self, batch_size: int, seed: int) -> None:
        if batch_size < 1:
            raise ValueError("Latent micro batch size must be at least 1.")
        if batch_size == 1:
            return
        buckets: dict[tuple[int, int, int], list[dict]] = defaultdict(list)
        rows_by_shard: dict[str, list[dict]] = defaultdict(list)
        for row in self.rows:
            rows_by_shard[row["shard"]].append(row)
        for shard, rows in rows_by_shard.items():
            with safe_open(str(self.cache_dir / shard), framework="pt", device="cpu") as handle:
                for row in rows:
                    key = f"{row['tensor_prefix']}/input_ids"
                    packed_length = int(handle.get_slice(key).get_shape()[-1])
                    buckets[(packed_length, int(row["height"]), int(row["width"]))].append(row)

        generator = random.Random(seed)
        batches = []
        dropped = 0
        for rows in buckets.values():
            generator.shuffle(rows)
            usable = len(rows) - len(rows) % batch_size
            batches.extend(rows[index:index + batch_size] for index in range(0, usable, batch_size))
            dropped += len(rows) - usable
        generator.shuffle(batches)
        self.rows = [row for batch in batches for row in batch]
        self.dropped_for_batching = dropped
        if not self.rows:
            raise RuntimeError(
                f"No exact-length latent batches can be formed with micro_batch_size={batch_size}."
            )

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


def model_kwargs_from_latent(
    record: dict,
    x_t: torch.Tensor,
    timestep: torch.Tensor,
    *,
    timestep_r: torch.Tensor | None = None,
    guidance: torch.Tensor | None = None,
) -> dict:
    """Build the exact model-facing subset without image loading or VAE execution."""
    kwargs = {
        "input_ids": record["input_ids"],
        "rope_image_info": record["rope_image_info"],
        "images": x_t.unsqueeze(0) if x_t.ndim == 3 else x_t,
        "image_mask": record["image_mask"],
        "timesteps": timestep.reshape(-1),
        "timesteps_index": record["timesteps_index"],
        "mode": "gen_image",
        "first_step": True,
        "return_dict": True,
        "use_cache": False,
        "gen_timestep_scatter_index": record.get("gen_timestep_scatter_index"),
    }
    if "guidance_index" in record:
        if guidance is None:
            raise ValueError("guidance is required when guidance_index is cached.")
        kwargs["guidance_index"] = record["guidance_index"]
        kwargs["guidance"] = guidance.reshape(-1)
    if "timesteps_r_index" in record:
        if timestep_r is None:
            raise ValueError("timestep_r is required when timesteps_r_index is cached.")
        kwargs["timesteps_r_index"] = record["timesteps_r_index"]
        kwargs["timesteps_r"] = timestep_r.reshape(-1)
    return kwargs
