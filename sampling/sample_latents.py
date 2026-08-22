#!/usr/bin/env python3
"""Build a resumable Hunyuan latent cache from an external image-text manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "train")]
from common.cache_schema import CACHE_VERSION, append_jsonl, config_hash, write_json, write_shard
from common.hunyuan import dtype_from_name, load_hunyuan
from sampling.hunyuan_sampler import sample_record


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


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    source, output = cfg["source"], cfg["output"]
    cache_dir = Path(output["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.jsonl"
    completed = set()
    if args.resume and manifest_path.exists():
        completed = {str(row["sample_id"]) for row in read_jsonl(manifest_path)}
    device = torch.device(cfg.get("device", "npu:0"))
    if device.type != "npu":
        raise RuntimeError("Offline Hunyuan sampling requires an Ascend NPU.")
    model = load_hunyuan(cfg["model_path"], device, cfg["dtype"])
    model.load_tokenizer(cfg["model_path"])
    model.eval()
    image_root = Path(source["image_root"])
    rows = list(read_jsonl(Path(source["manifest_path"])))
    target_count, shard_size = int(source["target_count"]), int(output["shard_size"])
    pending_tensors, pending_rows, shard_index = {}, [], len(list((cache_dir / "shards").glob("*.safetensors")))
    progress = tqdm(rows, desc="sampling", unit="sample")
    for row in progress:
        if len(completed) >= target_count:
            break
        sample_id = str(row["id"])
        if sample_id in completed:
            continue
        path = image_root / row["image_path"]
        try:
            with Image.open(path) as image:
                tensors, metadata = sample_record(
                    model, image, row["caption"], int(cfg["image"]["height"]), int(cfg["image"]["width"]),
                    device, dtype_from_name(cfg["dtype"]), sample_seed(int(cfg["seed"]), sample_id),
                )
        except Exception as exc:
            progress.write(f"skip sample_id={sample_id}: {exc}")
            continue
        prefix = f"sample_{sample_id}"
        for name, value in tensors.items():
            pending_tensors[f"{prefix}/{name}"] = value
        pending_rows.append({
            "sample_id": sample_id, "caption": row["caption"], "source_image": str(path),
            "tensor_prefix": prefix, "shard": f"shards/train-{shard_index:05d}.safetensors", **metadata,
        })
        completed.add(sample_id)
        progress.set_postfix(completed=len(completed), shard=shard_index)
        if len(pending_rows) == shard_size:
            shard = cache_dir / f"shards/train-{shard_index:05d}.safetensors"
            write_shard(shard, pending_tensors)
            append_jsonl(manifest_path, pending_rows)
            pending_tensors, pending_rows, shard_index = {}, [], shard_index + 1
    if pending_rows:
        shard = cache_dir / f"shards/train-{shard_index:05d}.safetensors"
        write_shard(shard, pending_tensors)
        append_jsonl(manifest_path, pending_rows)
    state = {"cache_version": CACHE_VERSION, "sample_count": len(completed), "config_hash": config_hash(cfg)}
    write_json(cache_dir / "state.json", state)
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
