import torch

from common.trajectory_schema import STEP_COUNT, unpack_bool_mask, validate_trajectory
from sampling.vllm_trajectory_adapter import build_vllm_trajectory_artifact


def test_vllm_payload_converts_to_training_schema():
    length = 11
    mask = torch.ones(1, 1, length, length, dtype=torch.bool).tril()
    condition = {
        "input_ids": torch.arange(length).reshape(1, length),
        "position_ids": torch.arange(length).reshape(1, length),
        "image_mask": torch.ones(1, length, dtype=torch.bool),
        "attention_mask": mask,
        "timesteps_index": torch.tensor([[1]]),
        "guidance_index": torch.tensor([[2]]),
        "timesteps_r_index": torch.tensor([[3]]),
        "gen_timestep_scatter_index": torch.tensor([[1]]),
        "guidance": torch.tensor([2500.0], dtype=torch.bfloat16),
    }
    payload = {
        "latents": torch.zeros(STEP_COUNT + 1, 4, 8, 8, dtype=torch.float32),
        "predictions": torch.zeros(STEP_COUNT, 4, 8, 8, dtype=torch.float32),
        "timesteps": torch.arange(STEP_COUNT, dtype=torch.float32),
        "timesteps_r": torch.arange(STEP_COUNT, dtype=torch.float32) - 1,
        "condition": condition,
        "metadata": {
            "prompt": "test prompt",
            "cot_text": "<think>test</think>",
            "height": 128,
            "width": 128,
            "token_height": 8,
            "token_width": 8,
            "full_attention_spans": [[(4, 10)]],
            "ar_generated_token_ids": [4, 5, 6],
            "guidance_scale": 2.5,
            "scheduler_latent_dtype": "float32",
        },
    }

    metadata, tensors = build_vllm_trajectory_artifact(
        payload,
        sample_id="1",
        seed=42,
        model_path="/model",
        vllm_commit="vllm-commit",
        repository_commit="sla-commit",
        bot_task="think_recaption",
        use_system_prompt="en_unified",
    )

    validate_trajectory(metadata, tensors)
    assert metadata["teacher_backend"] == "vllm-omni-dense"
    assert metadata["scheduler_replay_max_abs"] == 0.0
    assert metadata["scheduler_latent_dtype"] == "float32"
    assert metadata["rope_image_info"] == [[[4, 10, 8, 8]]]
    assert torch.equal(
        unpack_bool_mask(tensors["attention_mask_packed"], metadata["attention_mask_shape"]),
        mask,
    )
    assert tensors["teacher_predictions"].dtype == torch.float32
    assert tensors["ar_generated_token_ids"].tolist() == [4, 5, 6]
