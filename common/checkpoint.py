"""Shared helpers for distributed checkpoint directories."""

from __future__ import annotations

from pathlib import Path


def resolve_output_dir(root: Path, configured_path: str) -> Path:
    output_dir = Path(configured_path).expanduser()
    return output_dir if output_dir.is_absolute() else root / output_dir


def prepare_rank_checkpoint_dir(accelerator, output_dir: Path, tag: str) -> Path:
    """Create a DeepSpeed tag directory on every rank before collective writes."""
    checkpoint_dir = output_dir / tag
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    return checkpoint_dir
