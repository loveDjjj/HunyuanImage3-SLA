from pathlib import Path

import torch
from safetensors import safe_open

from common.cache_schema import write_shard


def test_shard_write_is_readable(tmp_path: Path):
    shard = tmp_path / "shard.safetensors"
    write_shard(shard, {"sample_1/latent_z0": torch.ones(2, 2)})
    assert shard.is_file()
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        assert torch.equal(handle.get_tensor("sample_1/latent_z0"), torch.ones(2, 2))
