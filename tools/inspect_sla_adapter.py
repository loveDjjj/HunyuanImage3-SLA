#!/usr/bin/env python3
"""Validate and summarize an exported HunyuanImage3 SLA adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.sla_adapter_schema import parameter_count, validate_adapter_tensors  # noqa: E402


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
    attention_ranks = {
        tensor.shape[0] if name.endswith("a.weight") else tensor.shape[1]
        for name, tensor in tensors.items()
        if ".qkv_lora." in name or ".o_lora." in name
    }
    moe_ranks = {
        tensor.shape[0] if name.endswith("a.weight") else tensor.shape[1]
        for name, tensor in tensors.items()
        if ".moe.experts." in name
    }
    if attention_ranks and attention_ranks != {int(config.get("attention_lora_rank", 0))}:
        raise ValueError(
            f"Attention LoRA rank metadata mismatch: config={config.get('attention_lora_rank')}, "
            f"tensors={sorted(attention_ranks)}"
        )
    if moe_ranks and moe_ranks != {int(config.get("moe_down_lora_rank", 0))}:
        raise ValueError(
            f"MoE LoRA rank metadata mismatch: config={config.get('moe_down_lora_rank')}, "
            f"tensors={sorted(moe_ranks)}"
        )
    validate_adapter_tensors(
        tensors,
        num_layers=int(config["num_layers"]),
        head_dim=int(config["head_dim"]),
        hidden_size=int(config.get("hidden_size", 4096)),
        q_heads=int(config.get("q_heads", 32)),
        kv_heads=int(config.get("kv_heads", 8)),
        num_experts=int(config.get("num_experts", 64)),
        moe_intermediate_size=int(config.get("moe_intermediate_size", 3072)),
        components=tuple(config.get("trained_components", ("proj_l",))),
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
    baseline_type = config.get("baseline_type")
    if baseline_type == "sla_zero_init":
        if tuple(config.get("trained_components", ())) != ("proj_l",):
            raise ValueError("ZeroInit SLA baseline must contain only proj_l tensors.")
        nonzero = sum(int(torch.count_nonzero(tensor).item()) for tensor in tensors.values())
        if nonzero:
            raise ValueError(f"ZeroInit SLA baseline contains {nonzero} non-zero parameter values.")
    return {
        "valid": True,
        "directory": str(directory),
        "training_step": config.get("training_step"),
        "format_version": int(config["format_version"]),
        "trained_components": list(config.get("trained_components", ("proj_l",))),
        "baseline_type": baseline_type,
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
