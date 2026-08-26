from types import SimpleNamespace

import torch
from torch import nn

from common.trajectory_schema import STEP_COUNT
from sampling.sample_trajectories import TrajectoryTokenStreamer
from sampling.trajectory_sampler import (
    DenseTrajectoryCapture,
    replay_scheduler,
    replay_vllm_scheduler,
    validate_meanflow_schedule,
)


class FakeScheduler:
    def __init__(self):
        self.config = {"scale": 0.125}
        self.set_timesteps(STEP_COUNT)

    @classmethod
    def from_config(cls, config):
        instance = cls()
        instance.config = dict(config)
        return instance

    def set_timesteps(self, steps, device=None):
        self.timesteps_full = torch.linspace(1000, 0, steps + 1, device=device)
        self.timesteps = self.timesteps_full[:-1]

    def step(self, prediction, timestep, sample, return_dict=False):
        del timestep
        value = sample - prediction * self.config["scale"]
        return (value,) if not return_dict else SimpleNamespace(prev_sample=value)


class FakeDenseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pipeline = SimpleNamespace(scheduler=FakeScheduler())

    def forward(self, **kwargs):
        return SimpleNamespace(diffusion_prediction=kwargs["images"] * 0.5)


def test_stage0_streamer_skips_prompt_and_counts_generated_tokens():
    streamer = TrajectoryTokenStreamer(4, enabled=True)
    streamer.put(torch.ones(1, 12, dtype=torch.long))
    streamer.put(torch.ones(1, dtype=torch.long))
    streamer.put(torch.ones(1, 2, dtype=torch.long))
    assert streamer._progress.n == 3
    streamer.end()


def condition(step, scheduler):
    length = 11
    return {
        "mode": "gen_image",
        "images": step,
        "timesteps": scheduler.timesteps[:1],
        "timesteps_r": scheduler.timesteps_full[1:2],
        "input_ids": torch.arange(length).reshape(1, length),
        "position_ids": torch.arange(length).reshape(1, length),
        "image_mask": torch.ones(1, length, dtype=torch.bool),
        "timesteps_index": torch.tensor([[1]]),
        "guidance_index": torch.tensor([[2]]),
        "timesteps_r_index": torch.tensor([[3]]),
        "gen_timestep_scatter_index": torch.tensor([[1]]),
        "guidance": torch.tensor([2500.0], dtype=torch.bfloat16),
        "attention_mask": torch.ones(1, 1, length, length, dtype=torch.bool).tril(),
        "rope_image_info": [[(slice(4, 10), (2, 3))]],
    }


def test_capture_builds_replayable_eight_step_artifact():
    model = FakeDenseModel()
    scheduler = model.pipeline.scheduler
    latent = torch.ones(1, 4, 8, 8, dtype=torch.float32)
    with DenseTrajectoryCapture(model, enabled=True) as capture:
        for index in range(STEP_COUNT):
            kwargs = condition(latent, scheduler)
            kwargs["timesteps"] = scheduler.timesteps[index:index + 1]
            kwargs["timesteps_r"] = scheduler.timesteps_full[index + 1:index + 2]
            output = model(**kwargs)
            latent = scheduler.step(
                output.diffusion_prediction,
                kwargs["timesteps"][0],
                latent,
                return_dict=False,
            )[0]

    metadata, tensors = capture.build_artifact(
        sample_id="1",
        prompt="test",
        seed=42,
        cot_text="<think>test</think>",
        ar_generated_token_ids=torch.tensor([1, 2]),
        guidance_scale=2.5,
        bot_task="think",
        use_system_prompt="en_unified",
        model_path="/model",
        repository_commit="abc",
    )

    assert tensors["latents"].shape == (9, 4, 8, 8)
    assert tensors["teacher_predictions"].shape == (8, 4, 8, 8)
    assert metadata["full_attention_spans"] == [[[4, 10]]]
    validate_meanflow_schedule(tensors, scheduler)
    assert replay_scheduler(tensors, scheduler) == 0.0


def test_vllm_scheduler_replay_applies_bf16_between_steps():
    scheduler = FakeScheduler()
    latent = torch.full((1, 4, 8, 8), 0.333, dtype=torch.bfloat16)
    latents = [latent[0].float()]
    predictions = []
    for index in range(STEP_COUNT):
        prediction = torch.full_like(latent, 0.117, dtype=torch.float32)
        predictions.append(prediction[0])
        latent = scheduler.step(
            prediction,
            scheduler.timesteps[index],
            latent,
            return_dict=False,
        )[0].to(torch.bfloat16)
        latents.append(latent[0].float())
    tensors = {
        "latents": torch.stack(latents),
        "teacher_predictions": torch.stack(predictions),
        "timesteps": scheduler.timesteps.float(),
    }
    assert replay_vllm_scheduler(tensors, scheduler) == 0.0
