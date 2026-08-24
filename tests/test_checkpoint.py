from pathlib import Path

from common.checkpoint import prepare_rank_checkpoint_dir, resolve_output_dir


class FakeAccelerator:
    def __init__(self):
        self.barriers = 0

    def wait_for_everyone(self):
        self.barriers += 1


def test_relative_output_dir_is_resolved_from_repository_root(tmp_path):
    assert resolve_output_dir(tmp_path, "results/training/default") == tmp_path / "results/training/default"


def test_absolute_output_dir_is_preserved(tmp_path):
    output_dir = tmp_path / "checkpoints"
    assert resolve_output_dir(Path("/unused"), str(output_dir)) == output_dir


def test_each_rank_prepares_tag_directory_before_checkpoint_write(tmp_path):
    accelerator = FakeAccelerator()

    checkpoint_dir = prepare_rank_checkpoint_dir(accelerator, tmp_path, "sla-step-100")

    assert checkpoint_dir.is_dir()
    assert accelerator.barriers == 1
