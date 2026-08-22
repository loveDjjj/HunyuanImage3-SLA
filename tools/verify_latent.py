#!/usr/bin/env python3
"""Validate an offline latent cache and publish its immutable READY marker."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.cache_schema import CACHE_VERSION, config_hash, write_json


def verify_cache(cache_dir: Path) -> dict:
    manifest = cache_dir / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise RuntimeError("No cache records found.")
    for row in rows:
        shard = cache_dir / row["shard"]
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for field in ("latent_z0", "input_ids", "image_mask", "timesteps_index"):
                key = f"{row['tensor_prefix']}/{field}"
                if key not in handle.keys():
                    raise RuntimeError(f"Missing {key} in {shard}")
                tensor = handle.get_tensor(key)
                if field == "latent_z0" and tensor.ndim != 3:
                    raise RuntimeError(f"Invalid latent shape {tensor.shape} for {row['sample_id']}")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    ready = {"cache_version": CACHE_VERSION, "sample_count": len(rows), "manifest_sha256": digest}
    write_json(cache_dir / "READY.json", ready)
    return ready


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(verify_cache(Path(args.cache_dir)), indent=2))


if __name__ == "__main__":
    main()
