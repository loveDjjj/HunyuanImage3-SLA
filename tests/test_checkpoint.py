from pathlib import Path

from common.checkpoint import (
    prepare_rank_checkpoint_dir,
    prune_checkpoints,
    prune_checkpoints_with_milestones,
    resolve_output_dir,
)


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


def test_checkpoint_retention_keeps_latest_five_steps(tmp_path):
    for step in (10, 20, 30, 40, 50, 60, 70):
        (tmp_path / f"sla-step-{step}").mkdir()
    (tmp_path / "latest").write_text("sla-step-70")
    (tmp_path / "unrelated").mkdir()

    removed = prune_checkpoints(tmp_path, "sla", keep=5)

    assert [path.name for path in removed] == ["sla-step-10", "sla-step-20"]
    assert sorted(path.name for path in tmp_path.glob("sla-step-*")) == [
        "sla-step-30",
        "sla-step-40",
        "sla-step-50",
        "sla-step-60",
        "sla-step-70",
    ]
    assert (tmp_path / "latest").is_file()
    assert (tmp_path / "unrelated").is_dir()


def test_milestone_retention_keeps_hundreds_and_latest_rolling_tag(tmp_path):
    for step in range(10, 251, 10):
        (tmp_path / f"sla-step-{step}").mkdir()
    (tmp_path / "latest").write_text("sla-step-250")

    removed = prune_checkpoints_with_milestones(
        tmp_path,
        "sla",
        milestone_every_steps=100,
        keep_latest_non_milestones=1,
    )

    assert len(removed) == 22
    assert sorted(path.name for path in tmp_path.glob("sla-step-*")) == [
        "sla-step-100",
        "sla-step-200",
        "sla-step-250",
    ]
    assert (tmp_path / "latest").read_text() == "sla-step-250"

    (tmp_path / "sla-step-260").mkdir()
    removed = prune_checkpoints_with_milestones(
        tmp_path,
        "sla",
        milestone_every_steps=100,
        keep_latest_non_milestones=1,
    )
    assert [path.name for path in removed] == ["sla-step-250"]
    assert sorted(path.name for path in tmp_path.glob("sla-step-*")) == [
        "sla-step-100",
        "sla-step-200",
        "sla-step-260",
    ]


def test_milestone_retention_supports_single_file_checkpoints(tmp_path):
    for step in (90, 100, 110):
        (tmp_path / f"sla-step-{step}.pt").touch()

    prune_checkpoints_with_milestones(
        tmp_path,
        "sla",
        milestone_every_steps=100,
        keep_latest_non_milestones=1,
    )

    assert sorted(path.name for path in tmp_path.glob("sla-step-*.pt")) == [
        "sla-step-100.pt",
        "sla-step-110.pt",
    ]
