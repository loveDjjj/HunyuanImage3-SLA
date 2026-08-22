#!/usr/bin/env python3
"""Print a concise summary of an offline latent cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()
    root = Path(args.cache_dir)
    manifest = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if not manifest:
        raise RuntimeError("manifest.jsonl is empty")
    row = manifest[args.sample_index]
    tensors = load_file(str(root / row["shard"]))
    prefix = f"sample_{row['sample_id']}/"
    payload = {"ready": (root / "READY.json").exists(), "sample": row, "tensor_shapes": {key.removeprefix(prefix): list(value.shape) for key, value in tensors.items() if key.startswith(prefix)}}
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
