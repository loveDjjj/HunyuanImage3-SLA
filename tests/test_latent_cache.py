import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common.cache_schema import CACHE_VERSION, write_json, write_shard
from train.latent_dataset import HunyuanLatentDataset, model_kwargs_from_latent, unwrap_single_record


def test_verified_cache_round_trip(tmp_path: Path):
    cache = tmp_path / "cache"
    write_shard(cache / "shards/train-00000.safetensors", {
        "sample_1/latent_z0": torch.randn(4, 8, 8),
        "sample_1/input_ids": torch.tensor([[1, 2, 3]]),
        "sample_1/image_mask": torch.tensor([[False, True, True]]),
        "sample_1/timesteps_index": torch.tensor([[0]]),
        "sample_1/gen_timestep_scatter_index": torch.tensor([[0]]),
    })
    row = {
        "sample_id": "1", "tensor_prefix": "sample_1", "shard": "shards/train-00000.safetensors",
        "rope_image_info": [[[1, 3, 2, 1]]], "height": 64, "width": 64,
    }
    (cache / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    write_json(cache / "READY.json", {"cache_version": CACHE_VERSION, "sample_count": 1})
    record = HunyuanLatentDataset(str(cache))[0]
    kwargs = model_kwargs_from_latent(record, record["latent_z0"], torch.tensor(500.0))
    assert kwargs["images"].shape == (1, 4, 8, 8)
    assert kwargs["rope_image_info"][0][0][0] == slice(1, 3)
    assert kwargs["timesteps"].item() == 500.0


def test_single_record_collate_preserves_cached_tensor_shapes():
    record = {"input_ids": torch.ones(1, 12, dtype=torch.long)}
    loader = DataLoader([record, record], batch_size=1, collate_fn=unwrap_single_record)
    batch = next(iter(loader))

    assert len(loader) == 2
    assert batch["input_ids"].shape == (1, 12)


def test_model_kwargs_include_distilled_guidance_and_meanflow_timestep():
    record = {
        "input_ids": torch.ones(1, 3, dtype=torch.long),
        "rope_image_info": [],
        "image_mask": torch.ones(1, 3, dtype=torch.bool),
        "timesteps_index": torch.tensor([[0]]),
        "guidance_index": torch.tensor([[1]]),
        "timesteps_r_index": torch.tensor([[2]]),
    }

    kwargs = model_kwargs_from_latent(
        record,
        torch.randn(4, 8, 8),
        torch.tensor(700.0),
        timestep_r=torch.tensor(300.0),
        guidance=torch.tensor(2500.0),
    )

    assert kwargs["guidance"].item() == 2500.0
    assert kwargs["timesteps_r"].item() == 300.0
    assert kwargs["use_cache"] is False
