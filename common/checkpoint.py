"""Shared helpers for distributed checkpoint directories."""

from __future__ import annotations

import re
import shutil
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


def prune_checkpoints(output_dir: Path, stage: str, keep: int) -> list[Path]:
    """Remove complete checkpoint tags older than the newest ``keep`` steps."""
    if keep < 1:
        raise ValueError("Checkpoint retention must keep at least one checkpoint.")
    pattern = re.compile(rf"^{re.escape(stage)}-step-(\d+)(?:\.pt)?$")
    checkpoints = []
    for path in output_dir.iterdir():
        match = pattern.fullmatch(path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    removed = [path for _, path in sorted(checkpoints)[:-keep]]
    for path in removed:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    return removed
