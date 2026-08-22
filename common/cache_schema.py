"""Versioned, atomic latent-cache helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


CACHE_VERSION = 1


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_shard(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    if not tensors:
        raise ValueError("Cannot write an empty latent shard.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    save_file({name: value.contiguous().cpu() for name, value in tensors.items()}, str(temporary))
    os.replace(temporary, path)


def cache_ready(cache_dir: Path) -> dict[str, Any]:
    path = cache_dir / "READY.json"
    if not path.is_file():
        raise FileNotFoundError(f"Latent cache is not verified: {path}")
    ready = json.loads(path.read_text(encoding="utf-8"))
    if ready.get("cache_version") != CACHE_VERSION:
        raise RuntimeError(f"Unsupported cache version: {ready.get('cache_version')}")
    return ready
