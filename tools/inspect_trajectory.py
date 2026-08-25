#!/usr/bin/env python3
"""Inspect one verified Hunyuan Dense trajectory artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.trajectory_schema import load_trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", required=True, type=Path)
    args = parser.parse_args()
    metadata, tensors = load_trajectory(args.sample_dir)
    payload = {
        "valid": True,
        "sample_dir": str(args.sample_dir),
        "sample_id": metadata["sample_id"],
        "prompt": metadata.get("prompt"),
        "seed": metadata.get("seed"),
        "step_count": metadata["step_count"],
        "tensor_bytes": (args.sample_dir / "trajectory.safetensors").stat().st_size,
        "scheduler_replay_max_abs": metadata.get("scheduler_replay_max_abs"),
        "dense_replay_max_abs_per_step": metadata.get("dense_replay_max_abs_per_step"),
        "tensors": {
            name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in sorted(tensors.items())
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
