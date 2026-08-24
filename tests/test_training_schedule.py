import pytest

from common.training_schedule import build_training_schedule


def test_max_steps_extends_epochs_for_a_sharded_dataloader():
    schedule = build_training_schedule(
        completed_steps=100,
        max_steps=200,
        batches_per_epoch=125,
        configured_epochs=1,
    )

    assert schedule.start_epoch == 0
    assert schedule.skip_batches == 100
    assert schedule.effective_epochs == 2


def test_resume_at_epoch_boundary_starts_next_epoch():
    schedule = build_training_schedule(
        completed_steps=125,
        max_steps=200,
        batches_per_epoch=125,
        configured_epochs=1,
    )

    assert schedule.start_epoch == 1
    assert schedule.skip_batches == 0
    assert schedule.effective_epochs == 2


def test_configured_epochs_remains_a_minimum():
    schedule = build_training_schedule(
        completed_steps=0,
        max_steps=100,
        batches_per_epoch=125,
        configured_epochs=3,
    )

    assert schedule.effective_epochs == 3


def test_checkpoint_cannot_be_ahead_of_requested_max_steps():
    with pytest.raises(ValueError, match="outside the requested range"):
        build_training_schedule(
            completed_steps=201,
            max_steps=200,
            batches_per_epoch=125,
            configured_epochs=1,
        )
