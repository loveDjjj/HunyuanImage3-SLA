"""Convert a vLLM-Omni Hunyuan teacher payload to the training artifact."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import torch

from common.trajectory_schema import STEP_COUNT, pack_bool_mask, validate_trajectory


def _tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"vLLM trajectory field {name!r} must be a tensor, got {type(value)!r}.")
    return value.detach().cpu().contiguous()


def build_vllm_trajectory_artifact(
    payload: Mapping[str, Any],
    *,
    sample_id: str,
    seed: int,
    model_path: str,
    vllm_commit: str | None,
    repository_commit: str | None,
    bot_task: str,
    use_system_prompt: str,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    condition = payload.get("condition")
    source_metadata = payload.get("metadata")
    if not isinstance(condition, Mapping) or not isinstance(source_metadata, Mapping):
        raise ValueError("vLLM teacher payload must contain condition and metadata mappings.")

    latents = _tensor(payload.get("latents"), "latents").float()
    predictions = _tensor(payload.get("predictions"), "predictions").float()
    timesteps = _tensor(payload.get("timesteps"), "timesteps").float().reshape(-1)
    timesteps_r = _tensor(payload.get("timesteps_r"), "timesteps_r").float().reshape(-1)
    if latents.shape[0] != STEP_COUNT + 1 or predictions.shape[0] != STEP_COUNT:
        raise ValueError(
            f"Expected 9 latents and 8 predictions, got {latents.shape[0]} and {predictions.shape[0]}."
        )

    attention_mask = _tensor(condition.get("attention_mask"), "condition.attention_mask").bool()
    packed_mask, mask_shape = pack_bool_mask(attention_mask)
    tensor_names = (
        "input_ids",
        "position_ids",
        "image_mask",
        "timesteps_index",
        "guidance_index",
        "timesteps_r_index",
        "gen_timestep_scatter_index",
        "guidance",
    )
    tensors = {
        "latents": latents,
        "teacher_predictions": predictions,
        "timesteps": timesteps,
        "timesteps_r": timesteps_r,
        "attention_mask_packed": packed_mask,
        "ar_generated_token_ids": torch.tensor(
            source_metadata.get("ar_generated_token_ids") or [], dtype=torch.long
        ),
    }
    for name in tensor_names:
        tensors[name] = _tensor(condition.get(name), f"condition.{name}")

    scheduler_latent_dtype = str(source_metadata.get("scheduler_latent_dtype") or "float32")
    if scheduler_latent_dtype not in {"float32", "bfloat16"}:
        raise ValueError(f"Unsupported vLLM scheduler latent dtype: {scheduler_latent_dtype!r}.")
    replay = tensors["latents"][0:1]
    if scheduler_latent_dtype == "bfloat16":
        replay = replay.to(torch.bfloat16)
    replay_max_abs = 0.0
    for index in range(STEP_COUNT):
        dt = (timesteps_r[index] - timesteps[index]) / 1000.0
        replay = replay.float() + predictions[index:index + 1] * dt
        expected = latents[index + 1:index + 2]
        if scheduler_latent_dtype == "bfloat16":
            replay = replay.to(torch.bfloat16)
            expected = expected.to(torch.bfloat16)
        replay_max_abs = max(replay_max_abs, (replay.float() - expected.float()).abs().max().item())
        torch.testing.assert_close(replay, expected, rtol=0, atol=0)

    raw_spans = source_metadata.get("full_attention_spans") or [[]]
    spans = [
        [[int(span[0]), int(span[1])] for span in batch_spans]
        for batch_spans in raw_spans
    ]
    token_height = int(source_metadata["token_height"])
    token_width = int(source_metadata["token_width"])
    rope_image_info = [
        [[start, stop, token_height, token_width] for start, stop in batch_spans]
        for batch_spans in spans
    ]
    prompt = str(source_metadata.get("prompt") or "")
    cot_text = str(source_metadata.get("cot_text") or "")
    metadata = {
        "trajectory_version": 1,
        "sample_id": str(sample_id),
        "prompt": prompt,
        "seed": int(seed),
        "step_count": STEP_COUNT,
        "guidance_scale": float(source_metadata.get("guidance_scale", 2.5)),
        "bot_task": bot_task,
        "use_system_prompt": use_system_prompt,
        "cot_text": cot_text,
        "system_prompt": str(source_metadata.get("system_prompt") or ""),
        "model_path": model_path,
        "repository_commit": repository_commit,
        "teacher_backend": "vllm-omni-dense",
        "scheduler_latent_dtype": scheduler_latent_dtype,
        "scheduler_replay_max_abs": replay_max_abs,
        "vllm_omni_commit": vllm_commit,
        "attention_mask_shape": mask_shape,
        "rope_image_info": rope_image_info,
        "full_attention_spans": spans,
        "height": int(source_metadata["height"]),
        "width": int(source_metadata["width"]),
        "condition_sha256": hashlib.sha256(
            tensors["input_ids"].numpy().tobytes() + packed_mask.numpy().tobytes()
        ).hexdigest(),
    }
    validate_trajectory(metadata, tensors)
    return metadata, tensors
