#!/usr/bin/env python3
"""Create a deployment-valid ZeroInit SLA baseline adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.sla_adapter_schema import (  # noqa: E402
    ARCHITECTURE,
    DEFAULT_HEAD_DIM,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_KV_HEADS,
    DEFAULT_NUM_LAYERS,
    DEFAULT_Q_HEADS,
    FORMAT_VERSION,
    canonical_key,
    parameter_count,
    validate_adapter_tensors,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def create_zero_adapter(
    output_dir: Path,
    *,
    training_config: dict[str, Any],
    num_layers: int = DEFAULT_NUM_LAYERS,
    head_dim: int = DEFAULT_HEAD_DIM,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    q_heads: int = DEFAULT_Q_HEADS,
    kv_heads: int = DEFAULT_KV_HEADS,
    force: bool = False,
) -> dict[str, Any]:
    tensors = {}
    for layer in range(num_layers):
        tensors[canonical_key(layer, "proj_l", "weight")] = torch.zeros(
            head_dim, head_dim, dtype=torch.float32
        )
        tensors[canonical_key(layer, "proj_l", "bias")] = torch.zeros(head_dim, dtype=torch.float32)
    validate_adapter_tensors(
        tensors,
        num_layers=num_layers,
        head_dim=head_dim,
        hidden_size=hidden_size,
        q_heads=q_heads,
        kv_heads=kv_heads,
        components=("proj_l",),
    )

    if output_dir.exists() and not force:
        raise FileExistsError(f"Output already exists: {output_dir}. Pass --force to replace it.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        adapter_path = temporary / "adapter.safetensors"
        save_file(
            tensors,
            str(adapter_path),
            metadata={
                "format_version": str(FORMAT_VERSION),
                "architecture": ARCHITECTURE,
                "baseline_type": "sla_zero_init",
            },
        )
        adapter_sha256 = _sha256(adapter_path)
        sla = training_config.get("sla", {}) or {}
        config = {
            "format_version": FORMAT_VERSION,
            "architecture": ARCHITECTURE,
            "base_model": str(training_config.get("model_path", "")),
            "training_step": 0,
            "training_repo_commit": _repository_commit(),
            "source_checkpoint": None,
            "baseline_type": "sla_zero_init",
            "initialization": "zeros",
            "num_layers": num_layers,
            "head_dim": head_dim,
            "hidden_size": hidden_size,
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "topk": float(sla.get("topk", 0.125)),
            "blkq": int(sla.get("blkq", 128)),
            "blkk": int(sla.get("blkk", 128)),
            "parameter_dtype": "float32",
            "parameter_dtypes": ["float32"],
            "compute_dtype": "bfloat16" if bool(sla.get("use_bf16", True)) else "float16",
            "training_backend": "zero_init_baseline",
            "trained_components": ["proj_l"],
            "trained_parameters": ["sla.proj_l.bias", "sla.proj_l.weight"],
            "tensor_count": len(tensors),
            "parameter_count": parameter_count(tensors),
            "adapter_sha256": adapter_sha256,
        }
        config_path = temporary / "adapter_config.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (temporary / "SHA256SUMS").write_text(
            f"{adapter_sha256}  adapter.safetensors\n{_sha256(config_path)}  adapter_config.json\n",
            encoding="ascii",
        )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(temporary, output_dir)
        return config
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/adapters/sla-zero-init")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train_sla_trajectory.yaml")
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--q-heads", type=int, default=DEFAULT_Q_HEADS)
    parser.add_argument("--kv-heads", type=int, default=DEFAULT_KV_HEADS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output = args.output if args.output.is_absolute() else ROOT / args.output
    training_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = create_zero_adapter(
        output,
        training_config=training_config,
        num_layers=args.num_layers,
        head_dim=args.head_dim,
        hidden_size=args.hidden_size,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        force=args.force,
    )
    print(json.dumps({"output": str(output), **config}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
