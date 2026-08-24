"""Training-loop schedule calculations shared by launch modes."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingSchedule:
    batches_per_epoch: int
    start_epoch: int
    skip_batches: int
    effective_epochs: int


def build_training_schedule(
    *, completed_steps: int, max_steps: int, batches_per_epoch: int, configured_epochs: int
) -> TrainingSchedule:
    if batches_per_epoch < 1:
        raise ValueError("The prepared DataLoader must contain at least one batch.")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1.")
    if configured_epochs < 1:
        raise ValueError("num_epochs must be at least 1.")
    if completed_steps < 0 or completed_steps > max_steps:
        raise ValueError(f"Checkpoint step {completed_steps} is outside the requested range 0..{max_steps}.")

    start_epoch, skip_batches = divmod(completed_steps, batches_per_epoch)
    required_epochs = math.ceil(max_steps / batches_per_epoch)
    return TrainingSchedule(
        batches_per_epoch=batches_per_epoch,
        start_epoch=start_epoch,
        skip_batches=skip_batches,
        effective_epochs=max(configured_epochs, required_epochs),
    )
