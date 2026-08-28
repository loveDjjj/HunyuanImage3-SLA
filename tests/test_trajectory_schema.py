import json

import torch

from common.trajectory_schema import (
    STEP_COUNT,
    load_trajectory,
    pack_bool_mask,
    unpack_bool_mask,
    validate_trajectory,
    write_trajectory_atomic,
)
from train.trajectory_dataset import (
    HunyuanTrajectoryDataset,
    HunyuanTrajectoryRolloutDataset,
    collate_rollout_records,
    collate_trajectory_records,
)


def fake_trajectory():
    mask = torch.ones(1, 1, 11, 11, dtype=torch.bool).tril()
    packed, shape = pack_bool_mask(mask)
    tensors = {
        "latents": torch.zeros(STEP_COUNT + 1, 4, 8, 8, dtype=torch.float32),
        "teacher_predictions": torch.zeros(STEP_COUNT, 4, 8, 8, dtype=torch.float32),
        "timesteps": torch.arange(STEP_COUNT, dtype=torch.float32),
        "timesteps_r": torch.arange(STEP_COUNT, dtype=torch.float32) - 1,
        "input_ids": torch.arange(11).reshape(1, 11),
        "position_ids": torch.arange(11).reshape(1, 11),
        "image_mask": torch.ones(1, 11, dtype=torch.bool),
        "timesteps_index": torch.tensor([[1]]),
        "guidance_index": torch.tensor([[2]]),
        "guidance": torch.tensor([2500.0], dtype=torch.bfloat16),
        "timesteps_r_index": torch.tensor([[3]]),
        "gen_timestep_scatter_index": torch.tensor([[1]]),
        "attention_mask_packed": packed,
        "ar_generated_token_ids": torch.tensor([4, 5, 6]),
    }
    metadata = {
        "trajectory_version": 1,
        "sample_id": "1",
        "step_count": 8,
        "attention_mask_shape": shape,
        "rope_image_info": [[[4, 10, 2, 3]]],
        "full_attention_spans": [[[4, 10]]],
    }
    return metadata, tensors, mask


def test_mask_pack_round_trip():
    _, tensors, mask = fake_trajectory()
    restored = unpack_bool_mask(tensors["attention_mask_packed"], list(mask.shape))
    assert torch.equal(restored, mask)


def test_atomic_trajectory_round_trip(tmp_path):
    metadata, tensors, mask = fake_trajectory()
    sample_dir = tmp_path / "sample_1"
    write_trajectory_atomic(sample_dir, metadata, tensors)

    loaded_metadata, loaded_tensors = load_trajectory(sample_dir)

    assert loaded_metadata == metadata
    assert (sample_dir / "READY.json").is_file()
    assert json.loads((sample_dir / "READY.json").read_text())["step_count"] == 8
    assert torch.equal(
        unpack_bool_mask(loaded_tensors["attention_mask_packed"], metadata["attention_mask_shape"]),
        mask,
    )
    validate_trajectory(loaded_metadata, loaded_tensors)


def test_trajectory_dataset_exposes_all_eight_steps(tmp_path):
    metadata, tensors, _ = fake_trajectory()
    sample_dir = tmp_path / "samples" / "sample_1"
    write_trajectory_atomic(sample_dir, metadata, tensors)
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps({"sample_id": "1", "path": "samples/sample_1"}) + "\n",
        encoding="utf-8",
    )

    dataset = HunyuanTrajectoryDataset(str(tmp_path))
    first = dataset[0]
    last = dataset[7]

    assert len(dataset) == 8
    assert first["images"].shape == (1, 4, 8, 8)
    assert first["images"].dtype == torch.bfloat16
    assert first["attention_mask"].shape == (1, 1, 11, 11)
    assert first["teacher_diffusion_prediction"].dtype == torch.float32
    assert first["teacher_diffusion_prediction"].shape == (1, 4, 8, 8)
    assert first["timesteps"].item() == 0
    assert first["trajectory_step"].item() == 0
    assert last["timesteps"].item() == 7
    assert last["trajectory_step"].item() == 7


def test_trajectory_dataset_uses_configured_model_input_dtype(tmp_path):
    metadata, tensors, _ = fake_trajectory()
    sample_dir = tmp_path / "samples" / "sample_1"
    write_trajectory_atomic(sample_dir, metadata, tensors)
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps({"sample_id": "1", "path": "samples/sample_1"}) + "\n",
        encoding="utf-8",
    )

    assert HunyuanTrajectoryDataset(str(tmp_path), dtype="fp16")[0]["images"].dtype == torch.float16
    assert HunyuanTrajectoryDataset(str(tmp_path), dtype="fp32")[0]["images"].dtype == torch.float32


def test_trajectory_dataset_forms_exact_layout_batch4(tmp_path):
    metadata, tensors, _ = fake_trajectory()
    sample_dir = tmp_path / "samples" / "sample_1"
    write_trajectory_atomic(sample_dir, metadata, tensors)
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps({"sample_id": "1", "path": "samples/sample_1"}) + "\n",
        encoding="utf-8",
    )
    dataset = HunyuanTrajectoryDataset(str(tmp_path), dtype="bf16")
    dataset.prepare_exact_length_batches(4, seed=7)

    batch = collate_trajectory_records([dataset[index] for index in range(4)])

    assert len(dataset) == 8
    assert dataset.dropped_for_batching == 0
    assert batch["images"].shape == (4, 4, 8, 8)
    assert batch["input_ids"].shape == (4, 11)
    assert batch["attention_mask"].shape == (4, 1, 11, 11)
    assert batch["teacher_diffusion_prediction"].shape == (4, 4, 8, 8)
    assert batch["trajectory_step"].shape == (4,)
    assert len(batch["rope_image_info"]) == 4
    assert len(batch["full_attention_spans"]) == 4


def test_rollout_dataset_exposes_full_trajectory_and_valid_padding(tmp_path):
    metadata, tensors, _ = fake_trajectory()
    tensors["scheduler_dts"] = torch.full((STEP_COUNT,), -0.125)
    metadata["scheduler_latent_dtype"] = "bfloat16"
    sample_dir = tmp_path / "samples" / "sample_1"
    write_trajectory_atomic(sample_dir, metadata, tensors)
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps({"sample_id": "1", "path": "samples/sample_1"}) + "\n",
        encoding="utf-8",
    )

    dataset = HunyuanTrajectoryRolloutDataset(
        str(tmp_path), dtype="bf16", max_prompts=1, world_size=2
    )
    real, padding = dataset[0], dataset[1]
    batch = collate_rollout_records([real])

    assert len(dataset) == 2
    assert dataset.padding_prompts == 1
    assert real["valid"].item() is True
    assert padding["valid"].item() is False
    assert batch["dense_latents"].shape == (1, 9, 4, 8, 8)
    assert batch["rollout_timesteps"].shape == (1, 8)
    assert batch["rollout_timesteps_r"].shape == (1, 8)
    assert batch["scheduler_dts"].shape == (1, 8)
    assert batch["scheduler_latent_dtype"] == ["bfloat16"]
    assert batch["valid"].shape == (1,)
