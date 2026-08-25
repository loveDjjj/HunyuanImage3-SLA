#!/usr/bin/env python3
"""Export a compact, deployment-oriented SLA adapter from a training checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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

from common.sla_adapter_schema import (
    ARCHITECTURE,
    DEFAULT_HEAD_DIM,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_KV_HEADS,
    DEFAULT_NUM_LAYERS,
    DEFAULT_Q_HEADS,
    FORMAT_VERSION,
    extract_adapter_tensors,
    infer_components,
    parameter_count,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _load_zero_state(checkpoint: Path) -> dict[str, object]:
    try:
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    except ImportError as exc:
        raise RuntimeError(
            "Exporting a ZeRO checkpoint requires DeepSpeed. Run this command in the training environment."
        ) from exc

    root, tag = checkpoint.parent, checkpoint.name
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"ZeRO checkpoint tag directory does not exist: {checkpoint}")
    state = get_fp32_state_dict_from_zero_checkpoint(
        str(root),
        tag=tag,
        exclude_frozen_parameters=True,
        lazy_mode=True,
    )
    if not isinstance(state, dict):
        raise TypeError(f"DeepSpeed returned an invalid state dict: {type(state)!r}")
    return state


def load_training_state(checkpoint: Path) -> dict[str, object]:
    if checkpoint.is_dir():
        return _load_zero_state(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint payload must be a dict, got {type(payload)!r}")
    state = payload.get("trainable_state_dict", payload.get("state_dict", payload))
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a state dict")
    return state


def infer_step(checkpoint: Path) -> int | None:
    match = re.search(r"(?:^|-)step-(\d+)(?:$|\.)", checkpoint.name)
    return int(match.group(1)) if match else None


def load_training_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise TypeError(f"Training config must contain a mapping: {path}")
    return config


def export_adapter(
    checkpoint: Path,
    output_dir: Path,
    *,
    training_config: dict[str, Any],
    base_model: str | None = None,
    num_layers: int = DEFAULT_NUM_LAYERS,
    head_dim: int = DEFAULT_HEAD_DIM,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    q_heads: int = DEFAULT_Q_HEADS,
    kv_heads: int = DEFAULT_KV_HEADS,
    force: bool = False,
) -> dict[str, Any]:
    state = load_training_state(checkpoint)
    tensors = extract_adapter_tensors(
        state,
        num_layers=num_layers,
        head_dim=head_dim,
        hidden_size=hidden_size,
        q_heads=q_heads,
        kv_heads=kv_heads,
    )
    trained_components = infer_components(tensors)

    if output_dir.exists() and not force:
        raise FileExistsError(f"Output already exists: {output_dir}. Pass --force to replace it.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        adapter_path = temporary / "adapter.safetensors"
        metadata = {
            "format_version": str(FORMAT_VERSION),
            "architecture": ARCHITECTURE,
            "base_model": base_model or str(training_config.get("model_path", "")),
        }
        save_file(tensors, str(adapter_path), metadata=metadata)
        adapter_sha256 = sha256_file(adapter_path)

        sla = training_config.get("sla", {}) or {}
        parameter_dtypes = sorted({str(tensor.dtype).removeprefix("torch.") for tensor in tensors.values()})
        config = {
            "format_version": FORMAT_VERSION,
            "architecture": ARCHITECTURE,
            "base_model": metadata["base_model"],
            "training_step": infer_step(checkpoint),
            "training_repo_commit": repository_commit(),
            "source_checkpoint": str(checkpoint.resolve()),
            "num_layers": num_layers,
            "head_dim": head_dim,
            "hidden_size": hidden_size,
            "q_heads": q_heads,
            "kv_heads": kv_heads,
            "topk": float(sla.get("topk", 0.125)),
            "blkq": int(sla.get("blkq", 64)),
            "blkk": int(sla.get("blkk", 128)),
            "parameter_dtype": parameter_dtypes[0] if len(parameter_dtypes) == 1 else "mixed",
            "parameter_dtypes": parameter_dtypes,
            "compute_dtype": "bfloat16" if bool(sla.get("use_bf16", True)) else "float16",
            "training_backend": str(sla.get("training_backend", "auto")),
            "trained_components": list(trained_components),
            "trained_parameters": [
                name.split(f"layers.0.", 1)[1]
                for name in sorted(tensors)
                if name.startswith("layers.0.")
            ],
            "tensor_count": len(tensors),
            "parameter_count": parameter_count(tensors),
            "adapter_sha256": adapter_sha256,
        }
        config_path = temporary / "adapter_config.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        config_sha256 = sha256_file(config_path)
        (temporary / "SHA256SUMS").write_text(
            f"{adapter_sha256}  adapter.safetensors\n{config_sha256}  adapter_config.json\n",
            encoding="ascii",
        )

        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(temporary, output_dir)
        return config
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train_sla.yaml")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--q-heads", type=int, default=DEFAULT_Q_HEADS)
    parser.add_argument("--kv-heads", type=int, default=DEFAULT_KV_HEADS)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    output = args.output if args.output.is_absolute() else ROOT / args.output
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = export_adapter(
        checkpoint,
        output,
        training_config=load_training_config(config_path),
        base_model=args.base_model,
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
