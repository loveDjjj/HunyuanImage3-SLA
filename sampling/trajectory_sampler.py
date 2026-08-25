"""Capture the official Dense HunyuanImage3 8-step MeanFlow rollout."""

from __future__ import annotations

import copy
import hashlib
from contextlib import AbstractContextManager
from typing import Any

import torch

from common.hunyuan import redirect_legacy_cuda_runtime
from common.trajectory_schema import STEP_COUNT, pack_bool_mask, unpack_bool_mask, validate_trajectory
from sampling.condition_packer import decode_rope_image_info, encode_rope_image_info


def _prediction_tensor(output: Any) -> torch.Tensor:
    value = getattr(output, "diffusion_prediction", None)
    if value is None and isinstance(output, dict):
        value = output.get("diffusion_prediction")
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Expected one Dense prediction, got {len(value)}")
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Invalid Dense prediction: {type(value)!r}")
    return value[0] if value.ndim == 4 and value.shape[0] == 1 else value


class DenseTrajectoryCapture(AbstractContextManager):
    """Non-invasive hooks around official model.forward and scheduler.step."""

    def __init__(self, model, *, enabled: bool):
        self.model = model
        self.scheduler = model.pipeline.scheduler
        self.enabled = enabled
        self._pre_handle = None
        self._post_handle = None
        self._original_step = None
        self.reset()

    def reset(self) -> None:
        self.x_t: list[torch.Tensor] = []
        self.x_next: list[torch.Tensor] = []
        self.timesteps: list[torch.Tensor] = []
        self.timesteps_r: list[torch.Tensor] = []
        self.predictions: list[torch.Tensor] = []
        self.condition: dict[str, Any] | None = None
        self._pending = False

    def __enter__(self):
        if not self.enabled:
            return self
        self._pre_handle = self.model.register_forward_pre_hook(self._pre_forward, with_kwargs=True)
        self._post_handle = self.model.register_forward_hook(self._post_forward, with_kwargs=True)
        self._original_step = self.scheduler.step

        def capture_step(model_output, timestep, sample, *args, **kwargs):
            result = self._original_step(model_output, timestep, sample, *args, **kwargs)
            next_latent = result[0] if isinstance(result, tuple) else result.prev_sample
            if self._pending:
                self.x_next.append(next_latent[0].detach().cpu().float().contiguous())
                self._pending = False
            return result

        self.scheduler.step = capture_step
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._pre_handle is not None:
            self._pre_handle.remove()
        if self._post_handle is not None:
            self._post_handle.remove()
        if self._original_step is not None:
            self.scheduler.step = self._original_step
        return False

    def _pre_forward(self, module, args, kwargs):
        del module, args
        images = kwargs.get("images")
        if kwargs.get("mode") != "gen_image" or not isinstance(images, torch.Tensor):
            self._pending = False
            return
        if images.ndim != 4 or images.shape[0] != 1:
            raise ValueError(f"Trajectory capture requires one generated image, got {tuple(images.shape)}")
        timestep = kwargs.get("timesteps")
        timestep_r = kwargs.get("timesteps_r")
        if not isinstance(timestep, torch.Tensor) or not isinstance(timestep_r, torch.Tensor):
            raise ValueError("Official Distil forward must provide both timesteps and timesteps_r.")
        self.x_t.append(images[0].detach().cpu().float().contiguous())
        self.timesteps.append(timestep.reshape(-1)[0].detach().cpu().float())
        self.timesteps_r.append(timestep_r.reshape(-1)[0].detach().cpu().float())
        self._pending = True
        if self.condition is None:
            names = (
                "input_ids",
                "position_ids",
                "image_mask",
                "timesteps_index",
                "guidance_index",
                "timesteps_r_index",
                "gen_timestep_scatter_index",
                "guidance",
                "attention_mask",
            )
            condition = {}
            for name in names:
                value = kwargs.get(name)
                if not isinstance(value, torch.Tensor):
                    raise ValueError(f"Official first-step condition is missing tensor {name!r}.")
                condition[name] = value.detach().cpu().contiguous()
            condition["rope_image_info"] = kwargs.get("rope_image_info")
            self.condition = condition

    def _post_forward(self, module, args, kwargs, output):
        del module, args, kwargs
        if self._pending:
            self.predictions.append(_prediction_tensor(output).detach().cpu().float().contiguous())

    def build_artifact(
        self,
        *,
        sample_id: str,
        prompt: str,
        seed: int,
        cot_text: str,
        ar_generated_token_ids: torch.Tensor,
        guidance_scale: float,
        bot_task: str,
        use_system_prompt: str,
        model_path: str,
        repository_commit: str | None,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        lengths = (len(self.x_t), len(self.x_next), len(self.predictions), len(self.timesteps_r))
        if lengths != (STEP_COUNT, STEP_COUNT, STEP_COUNT, STEP_COUNT):
            raise RuntimeError(f"Official rollout is incomplete: {lengths=}")
        if self.condition is None:
            raise RuntimeError("Official rollout did not expose a diffusion condition.")
        for index in range(STEP_COUNT - 1):
            torch.testing.assert_close(self.x_next[index], self.x_t[index + 1], rtol=0, atol=0)

        packed_mask, mask_shape = pack_bool_mask(self.condition["attention_mask"])
        rope = self.condition["rope_image_info"] or [[]]
        tensors = {
            "latents": torch.stack([self.x_t[0], *self.x_next]),
            "teacher_predictions": torch.stack(self.predictions),
            "timesteps": torch.stack(self.timesteps).float(),
            "timesteps_r": torch.stack(self.timesteps_r).float(),
            "attention_mask_packed": packed_mask,
            "ar_generated_token_ids": ar_generated_token_ids.detach().cpu().long().reshape(-1),
        }
        for name in (
            "input_ids",
            "position_ids",
            "image_mask",
            "timesteps_index",
            "guidance_index",
            "timesteps_r_index",
            "gen_timestep_scatter_index",
            "guidance",
        ):
            tensors[name] = self.condition[name]
        metadata = {
            "trajectory_version": 1,
            "sample_id": sample_id,
            "prompt": prompt,
            "seed": int(seed),
            "step_count": STEP_COUNT,
            "guidance_scale": float(guidance_scale),
            "bot_task": bot_task,
            "use_system_prompt": use_system_prompt,
            "cot_text": cot_text,
            "model_path": model_path,
            "repository_commit": repository_commit,
            "attention_mask_shape": mask_shape,
            "rope_image_info": encode_rope_image_info(rope),
            "full_attention_spans": [
                [[int(token_slice.start), int(token_slice.stop)] for token_slice, _ in row]
                for row in rope
            ],
            "condition_sha256": hashlib.sha256(
                tensors["input_ids"].numpy().tobytes() + packed_mask.numpy().tobytes()
            ).hexdigest(),
        }
        validate_trajectory(metadata, tensors)
        return metadata, tensors


def validate_meanflow_schedule(tensors: dict[str, torch.Tensor], scheduler) -> None:
    expected_t = scheduler.timesteps.detach().cpu().float()
    expected_r = scheduler.timesteps_full[1:].detach().cpu().float()
    torch.testing.assert_close(tensors["timesteps"], expected_t, rtol=0, atol=0)
    torch.testing.assert_close(tensors["timesteps_r"], expected_r, rtol=0, atol=0)


def replay_scheduler(tensors: dict[str, torch.Tensor], scheduler) -> float:
    replay = type(scheduler).from_config(copy.deepcopy(scheduler.config))
    replay.set_timesteps(STEP_COUNT, device="cpu")
    current = tensors["latents"][0].unsqueeze(0)
    max_error = 0.0
    for index in range(STEP_COUNT):
        current = replay.step(
            tensors["teacher_predictions"][index].unsqueeze(0),
            tensors["timesteps"][index],
            current,
            return_dict=False,
        )[0].float()
        expected = tensors["latents"][index + 1].unsqueeze(0)
        max_error = max(max_error, (current.float() - expected.float()).abs().max().item())
        torch.testing.assert_close(current, expected, rtol=0, atol=0)
    return max_error


@torch.no_grad()
def replay_dense_predictions(
    model,
    metadata: dict[str, Any],
    tensors: dict[str, torch.Tensor],
    device: torch.device,
    *,
    atol: float,
    rtol: float,
) -> list[float]:
    attention_mask = unpack_bool_mask(
        tensors["attention_mask_packed"], metadata["attention_mask_shape"]
    ).to(device)
    rope_image_info = decode_rope_image_info(metadata["rope_image_info"])
    static_names = (
        "input_ids",
        "position_ids",
        "image_mask",
        "timesteps_index",
        "guidance_index",
        "timesteps_r_index",
        "gen_timestep_scatter_index",
        "guidance",
    )
    static = {name: tensors[name].to(device) for name in static_names}
    errors = []
    with redirect_legacy_cuda_runtime():
        for index in range(STEP_COUNT):
            output = model(
                **static,
                attention_mask=attention_mask,
                rope_image_info=rope_image_info,
                images=tensors["latents"][index:index + 1].to(device),
                timesteps=tensors["timesteps"][index:index + 1].to(device),
                timesteps_r=tensors["timesteps_r"][index:index + 1].to(device),
                mode="gen_image",
                first_step=True,
                use_cache=False,
                return_dict=True,
            )
            actual = _prediction_tensor(output).detach().cpu().float()
            expected = tensors["teacher_predictions"][index]
            error = (actual - expected).abs().max().item()
            errors.append(error)
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    return errors
