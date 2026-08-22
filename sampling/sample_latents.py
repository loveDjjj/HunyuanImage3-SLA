#!/usr/bin/env python3
"""Build a resumable, rank-sharded Hunyuan VAE latent cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "upstream" / "HunyuanImage-3.0")]
from common.cache_schema import CACHE_VERSION, append_jsonl, config_hash, write_json, write_shard
from common.hunyuan import dtype_from_name
from sampling.hunyuan_sampler import sample_record
from sampling.vae_only import load_vae_only
from tools.verify_latent import verify_cache


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "sampling.yaml"))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sample_seed(base_seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def sample_partition(sample_id: str, world_size: int) -> int:
    try:
        return int(sample_id) % world_size
    except ValueError:
        return sample_seed(0, sample_id) % world_size


def distributed_context(backend: str) -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        import torch_npu  # noqa: F401 - registers torch.npu and HCCL backend
        torch.npu.set_device(local_rank)
    if world_size > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=backend)
    return rank, local_rank, world_size


def merge_rank_caches(cache_dir: Path, world_size: int, expected_count: int) -> dict:
    rows: list[dict] = []
    shards_dir = cache_dir / "shards"
    if (cache_dir / "manifest.jsonl").exists():
        raise RuntimeError("Merged manifest already exists; use a new cache_dir for a new sampling run.")
    for rank in range(world_size):
        rank_dir = cache_dir / f"rank-{rank:03d}"
        for row in read_jsonl(rank_dir / "manifest.jsonl"):
            source = rank_dir / row["shard"]
            target_name = f"rank-{rank:03d}-{source.name}"
            target = shards_dir / target_name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copyfile(source, target)
            row["shard"] = f"shards/{target_name}"
            rows.append(row)
    rows.sort(key=lambda row: str(row["sample_id"]))
    if len(rows) != expected_count:
        raise RuntimeError(f"Sampling incomplete: expected {expected_count}, got {len(rows)}. Add valid candidates and resume.")
    append_jsonl(cache_dir / "manifest.jsonl", rows)
    state = {"cache_version": CACHE_VERSION, "sample_count": len(rows), "world_size": world_size}
    write_json(cache_dir / "state.json", state)
    return verify_cache(cache_dir)


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    source, output = cfg["source"], cfg["output"]
    cache_dir = Path(output["cache_dir"])
    rank, local_rank, world_size = distributed_context(cfg.get("distributed", {}).get("backend", "hccl"))
    if args.resume and (cache_dir / "READY.json").exists():
        if rank == 0:
            print(f"cache already verified: {cache_dir}")
        return

    try:
        import torch_npu  # noqa: F401 - registers torch.npu and HCCL backend
    except ImportError as exc:
        raise RuntimeError("Offline sampling requires the torch_npu runtime.") from exc
    device = torch.device(f"npu:{local_rank}")
    if device.type != "npu":
        raise RuntimeError("Offline Hunyuan sampling requires an Ascend NPU.")
    rank_dir = cache_dir / f"rank-{rank:03d}"
    manifest_path = rank_dir / "manifest.jsonl"
    completed = {str(row["sample_id"]) for row in read_jsonl(manifest_path)} if args.resume and manifest_path.exists() else set()
    target_count, shard_size = int(source["target_count"]), int(output["shard_size"])
    rank_quota = target_count // world_size + int(rank < target_count % world_size)
    model = load_vae_only(cfg["model_path"], device, cfg["dtype"])
    image_root, rows = Path(source["image_root"]), list(read_jsonl(Path(source["manifest_path"])))
    pending_tensors, pending_rows = {}, []
    shard_index = len(list((rank_dir / "shards").glob("*.safetensors")))
    progress = tqdm(rows, desc=f"sampling rank={rank}", unit="sample", disable=rank != 0)
    for row in progress:
        if len(completed) >= rank_quota:
            break
        sample_id = str(row["id"])
        if sample_partition(sample_id, world_size) != rank or sample_id in completed:
            continue
        path = image_root / row["image_path"]
        try:
            with Image.open(path) as image:
                tensors, metadata = sample_record(model, image, row["caption"], int(cfg["image"]["height"]), int(cfg["image"]["width"]), device, dtype_from_name(cfg["dtype"]), sample_seed(int(cfg["seed"]), sample_id))
        except Exception as exc:
            progress.write(f"skip sample_id={sample_id}: {exc}")
            continue
        prefix = f"sample_{sample_id}"
        pending_tensors.update({f"{prefix}/{name}": value for name, value in tensors.items()})
        pending_rows.append({"sample_id": sample_id, "caption": row["caption"], "source_image": str(path), "tensor_prefix": prefix, "shard": f"shards/train-{shard_index:05d}.safetensors", **metadata})
        completed.add(sample_id)
        if len(pending_rows) == shard_size:
            write_shard(rank_dir / pending_rows[0]["shard"], pending_tensors)
            append_jsonl(manifest_path, pending_rows)
            pending_tensors, pending_rows, shard_index = {}, [], shard_index + 1
    if pending_rows:
        write_shard(rank_dir / pending_rows[0]["shard"], pending_tensors)
        append_jsonl(manifest_path, pending_rows)
    write_json(rank_dir / "state.json", {"cache_version": CACHE_VERSION, "rank": rank, "world_size": world_size, "sample_count": len(completed), "target_count": rank_quota, "config_hash": config_hash(cfg)})
    if world_size > 1:
        torch.distributed.barrier()
    if rank == 0:
        ready = merge_rank_caches(cache_dir, world_size, target_count)
        print(json.dumps(ready, indent=2))


if __name__ == "__main__":
    main()
