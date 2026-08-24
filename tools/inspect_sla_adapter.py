#!/usr/bin/env python3
"""Validate and summarize an exported HunyuanImage3 SLA adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.sla_adapter_schema import parameter_count, validate_adapter_tensors

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_adapter(directory: Path) -> dict:
    config_path = directory / "adapter_config.json"
    adapter_path = directory / "adapter.safetensors"
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    tensors = load_file(str(adapter_path), device="cpu")
    validate_adapter_tensors(
        tensors,
        num_layers=int(config["num_layers"]),
        head_dim=int(config["head_dim"]),
    )
    actual_sha256 = sha256_file(adapter_path)
    if actual_sha256 != config["adapter_sha256"]:
        raise ValueError(
            f"adapter SHA256 mismatch: config={config['adapter_sha256']}, actual={actual_sha256}"
        )
    actual_parameters = parameter_count(tensors)
    if len(tensors) != int(config["tensor_count"]):
        raise ValueError(f"tensor_count mismatch: config={config['tensor_count']}, actual={len(tensors)}")
    if actual_parameters != int(config["parameter_count"]):
        raise ValueError(
            f"parameter_count mismatch: config={config['parameter_count']}, actual={actual_parameters}"
        )
    return {
        "valid": True,
        "directory": str(directory),
        "training_step": config.get("training_step"),
        "tensor_count": len(tensors),
        "parameter_count": actual_parameters,
        "dtype": sorted({str(tensor.dtype) for tensor in tensors.values()}),
        "adapter_sha256": actual_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    args = parser.parse_args()
    directory = args.adapter_dir if args.adapter_dir.is_absolute() else ROOT / args.adapter_dir
    print(json.dumps(inspect_adapter(directory), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
