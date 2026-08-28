#!/usr/bin/env python3
"""Single- and multi-NPU Dense/SLA recovery training entrypoint."""

from __future__ import annotations

import argparse
import glob
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "upstream" / "DiffSynth-Studio"))

from diffsynth.diffusion import DiffusionTrainingModule
from common.accelerate_config import (
    configure_deepspeed_micro_batch,
    create_accelerator,
    deepspeed_offload_devices,
)
from common.checkpoint import (
    prepare_rank_checkpoint_dir,
    prune_checkpoints,
    prune_checkpoints_with_milestones,
    resolve_output_dir,
)
from common.gradient import deepspeed_local_gradient, inspect_local_gradients
from common.hunyuan import prepare_diffusion_runtime, redirect_legacy_cuda_runtime
from common.training_metrics import MetricsLogger
from common.training_schedule import build_training_schedule
from hunyuan_adapter import (
    HunyuanSLARecoveryModule,
    freeze_model,
    load_hunyuan,
    unfreeze_matching,
)
from latent_dataset import (
    HunyuanLatentDataset,
    collate_latent_records,
    model_kwargs_from_latent,
    unwrap_single_record,
)
from noise_sampler import flow_match_batch
from trajectory_dataset import (
    HunyuanTrajectoryDataset,
    HunyuanTrajectoryRolloutDataset,
    collate_rollout_records,
    collate_trajectory_records,
)


def _validation_values(statistics: torch.Tensor) -> dict[str, float]:
    squared_error, teacher_squared, dot, student_squared, elements = [
        float(value) for value in statistics.tolist()
    ]
    epsilon = 1.0e-12
    return {
        "mse": squared_error / max(elements, 1.0),
        "relative_mse": squared_error / max(teacher_squared, epsilon),
        "cosine": dot / max(math.sqrt(student_squared * teacher_squared), epsilon),
    }


def _tensor_prediction(value: Any) -> torch.Tensor:
    tensors = []

    def collect(item):
        if isinstance(item, torch.Tensor):
            tensors.append(item)
        elif isinstance(item, (list, tuple)):
            for child in item:
                collect(child)

    collect(value)
    if len(tensors) != 1:
        raise RuntimeError(f"Rollout requires exactly one prediction tensor, got {len(tensors)}.")
    return tensors[0]


def _additive_statistics(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    actual, expected = actual.float(), expected.float()
    difference = actual - expected
    return torch.stack(
        (
            difference.square().sum(),
            expected.square().sum(),
            (actual * expected).sum(),
            actual.square().sum(),
            actual.new_tensor(actual.numel()),
        )
    )


def _laplacian(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 4:
        raise ValueError(f"Laplacian validation requires BCHW latent, got {tuple(value.shape)}.")
    padded = F.pad(value.float(), (1, 1, 1, 1), mode="replicate")
    center = padded[:, :, 1:-1, 1:-1]
    return (
        padded[:, :, :-2, 1:-1]
        + padded[:, :, 2:, 1:-1]
        + padded[:, :, 1:-1, :-2]
        + padded[:, :, 1:-1, 2:]
        - 4.0 * center
    )


@torch.no_grad()
def evaluate_validation(accelerator, training_model, dataloader, device) -> dict[str, Any]:
    training_model.eval()
    aggregate = torch.zeros((9, 5), dtype=torch.float32, device=device)
    for batch in dataloader:
        batch = move(batch, device)
        teacher_prediction = batch.pop("teacher_diffusion_prediction")
        full_attention_spans = batch.pop("full_attention_spans")
        trajectory_steps = batch.pop("trajectory_step")
        batch.pop("sample_id", None)
        statistics = training_model(
            batch,
            teacher_prediction=teacher_prediction,
            full_attention_spans=full_attention_spans,
            return_statistics=True,
        ).float()
        aggregate[0] += statistics
        if trajectory_steps.numel() != 1:
            raise RuntimeError("Validation requires micro_batch_size_per_gpu=1 for per-step metrics.")
        aggregate[int(trajectory_steps.item()) + 1] += statistics
    aggregate = accelerator.reduce(aggregate, reduction="sum").cpu()
    training_model.train()
    total = _validation_values(aggregate[0])
    return {
        "validation_mse": total["mse"],
        "validation_relative_mse": total["relative_mse"],
        "validation_cosine": total["cosine"],
        "validation_cosine_distance": 1.0 - total["cosine"],
        "validation_mse_by_step": [
            _validation_values(aggregate[index])["mse"] for index in range(1, 9)
        ],
        "validation_relative_mse_by_step": [
            _validation_values(aggregate[index])["relative_mse"]
            for index in range(1, 9)
        ],
        "validation_cosine_distance_by_step": [
            1.0 - _validation_values(aggregate[index])["cosine"]
            for index in range(1, 9)
        ],
    }


@torch.no_grad()
def evaluate_rollout(
    accelerator,
    training_model,
    dataloader,
    device,
    *,
    compute_dtype: torch.dtype,
) -> dict[str, Any]:
    training_model.eval()
    aggregate = torch.zeros((8, 5), dtype=torch.float32, device=device)
    laplacian_error = torch.zeros(2, dtype=torch.float32, device=device)
    for batch in dataloader:
        batch = move(batch, device)
        if batch["valid"].numel() != 1:
            raise RuntimeError("Rollout validation requires micro_batch_size_per_gpu=1.")
        valid = bool(batch.pop("valid").item())
        dense_latents = batch.pop("dense_latents")
        timesteps = batch.pop("rollout_timesteps")
        timesteps_r = batch.pop("rollout_timesteps_r")
        scheduler_dts = batch.pop("scheduler_dts")
        scheduler_dtype = batch.pop("scheduler_latent_dtype")[0]
        full_attention_spans = batch.pop("full_attention_spans")
        batch.pop("sample_id", None)
        current = dense_latents[:, 0]
        if scheduler_dtype == "bfloat16":
            current = current.to(torch.bfloat16)
        elif scheduler_dtype != "float32":
            raise ValueError(f"Unsupported rollout scheduler latent dtype: {scheduler_dtype!r}.")

        for trajectory_step in range(8):
            model_kwargs = {
                **batch,
                "images": current.to(compute_dtype),
                "timesteps": timesteps[:, trajectory_step],
                "timesteps_r": timesteps_r[:, trajectory_step],
            }
            prediction = _tensor_prediction(
                training_model(
                    model_kwargs,
                    full_attention_spans=full_attention_spans,
                    return_prediction=True,
                )
            )
            dt = scheduler_dts[:, trajectory_step].reshape(-1, 1, 1, 1)
            current = current.float() + prediction.float() * dt
            if scheduler_dtype == "bfloat16":
                current = current.to(torch.bfloat16)
            expected = dense_latents[:, trajectory_step + 1]
            if valid:
                aggregate[trajectory_step] += _additive_statistics(current, expected)
                if trajectory_step == 7:
                    actual_laplacian = _laplacian(current)
                    expected_laplacian = _laplacian(expected)
                    laplacian_error += torch.stack(
                        (
                            (actual_laplacian - expected_laplacian).square().sum(),
                            expected_laplacian.square().sum(),
                        )
                    )

    aggregate = accelerator.reduce(aggregate, reduction="sum").cpu()
    laplacian_error = accelerator.reduce(laplacian_error, reduction="sum").cpu()
    training_model.train()
    by_step = [_validation_values(row) for row in aggregate]
    final = by_step[-1]
    return {
        "rollout_final_latent_mse": final["mse"],
        "rollout_final_latent_relative_mse": final["relative_mse"],
        "rollout_final_latent_cosine_distance": 1.0 - final["cosine"],
        "rollout_final_laplacian_relative_mse": float(laplacian_error[0])
        / max(float(laplacian_error[1]), 1.0e-12),
        "rollout_latent_mse_by_step": [value["mse"] for value in by_step],
        "rollout_latent_relative_mse_by_step": [
            value["relative_mse"] for value in by_step
        ],
        "rollout_latent_cosine_distance_by_step": [
            1.0 - value["cosine"] for value in by_step
        ],
    }


def _peak_npu_memory(device) -> int:
    return int(torch.npu.max_memory_allocated(device))


def _logical_parameter_count(parameters) -> int:
    return sum(int(getattr(parameter, "ds_numel", parameter.numel())) for parameter in parameters)


class SerializedModelInputs(Dataset):
    """A directory of ``torch.save(dict(model_forward_kwargs))`` diffusion batches."""

    def __init__(self, pattern: str):
        self.paths = sorted(glob.glob(pattern))
        if not self.paths:
            raise FileNotFoundError(f"No serialized model-input batches match: {pattern}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        data = torch.load(self.paths[index], map_location="cpu", weights_only=False)
        if not isinstance(data, dict):
            raise TypeError(f"{self.paths[index]} must contain a dict of Hunyuan model forward arguments")
        return data


def move(value: Any, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, list):
        return [move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move(item, device) for item in value)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    return value


class DenseForwardBackwardModule(DiffusionTrainingModule):
    """Phase-1 verification: train only a configured diffusion-output probe parameter."""

    def __init__(self, model, patterns: list[str]):
        super().__init__()
        self.model = model
        freeze_model(model)
        self.unfrozen_names = unfreeze_matching(model, patterns)

    def forward(self, model_kwargs):
        prepare_diffusion_runtime(self.model, model_kwargs)
        with redirect_legacy_cuda_runtime():
            output = self.model(**model_kwargs).diffusion_prediction
        tensors = []

        def collect(value):
            if isinstance(value, torch.Tensor):
                tensors.append(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)

        collect(output)
        if not tensors:
            raise RuntimeError("Dense probe requires diffusion_prediction tensors. Use mode='gen_image'.")
        return torch.stack([item.float().square().mean() for item in tensors]).mean()

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

    def trainable_parameter_groups(self) -> dict[str, list[torch.nn.Parameter]]:
        return {"dense": [parameter for parameter in self.parameters() if parameter.requires_grad]}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "train_sla.yaml"))
    parser.add_argument("--stage", choices=("dense", "sla"), default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume-from", default=None, help="Checkpoint written by this entrypoint")
    parser.add_argument(
        "--trainable-components",
        nargs="+",
        choices=("proj_l", "qkv_delta", "o_delta", "qkv_lora", "o_lora"),
        default=None,
    )
    parser.add_argument("--training-backend", choices=("auto", "triton"), default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--validation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable cached badcase trajectory validation",
    )
    return parser.parse_args()


def _using_deepspeed(accelerator) -> bool:
    return str(accelerator.distributed_type).upper().endswith("DEEPSPEED")


def _checkpoint_client_state(cfg: dict[str, Any], dataset, step: int) -> dict[str, Any]:
    return {
        "step": step,
        "stage": cfg["stage"],
        "config": cfg,
        "cache": getattr(dataset, "ready", None),
    }


def save_checkpoint(accelerator, training_model, optimizer, dataset, cfg: dict[str, Any], step: int) -> Path:
    output_dir = resolve_output_dir(ROOT, cfg["output_dir"])
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    if _using_deepspeed(accelerator):
        tag = f"{cfg['stage']}-step-{step}"
        path = prepare_rank_checkpoint_dir(accelerator, output_dir, tag)
        # DeepSpeed checkpointing is collective. Excluding frozen parameters keeps
        # the restart data focused on SLA parameters and their optimizer shards.
        training_model.save_checkpoint(
            str(output_dir),
            tag=tag,
            client_state=_checkpoint_client_state(cfg, dataset, step),
            save_latest=True,
            exclude_frozen_parameters=True,
        )
    else:
        unwrapped = accelerator.unwrap_model(training_model)
        checkpoint = {
            **_checkpoint_client_state(cfg, dataset, step),
            "trainable_parameter_names": unwrapped.trainable_parameter_names(),
            "trainable_state_dict": {
                name: p.detach().cpu() for name, p in unwrapped.named_parameters() if p.requires_grad
            },
            "optimizer": optimizer.state_dict(),
        }
        path = output_dir / f"{cfg['stage']}-step-{step}.pt"
        if accelerator.is_main_process:
            torch.save(checkpoint, path)

    accelerator.wait_for_everyone()
    milestone_every_steps = int(cfg.get("checkpoint_milestone_every_steps", 0))
    max_checkpoints = int(cfg.get("max_checkpoints", 0))
    if accelerator.is_main_process and milestone_every_steps > 0:
        removed = prune_checkpoints_with_milestones(
            output_dir,
            cfg["stage"],
            milestone_every_steps=milestone_every_steps,
            keep_latest_non_milestones=int(
                cfg.get("checkpoint_keep_latest_non_milestones", 1)
            ),
        )
        if removed:
            print("pruned_checkpoints=" + ",".join(path.name for path in removed))
    elif accelerator.is_main_process and max_checkpoints > 0:
        prune_checkpoints(output_dir, cfg["stage"], max_checkpoints)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"checkpoint={path}")
    return path


def load_checkpoint(accelerator, training_model, optimizer, cfg: dict[str, Any], resume_from: str) -> int:
    path = Path(resume_from)
    if not path.is_absolute():
        path = ROOT / path
    if _using_deepspeed(accelerator):
        if path.suffix == ".pt":
            raise ValueError("ZeRO-3 resume requires a DeepSpeed checkpoint directory, not a .pt checkpoint.")
        load_path, client_state = training_model.load_checkpoint(
            str(path.parent),
            tag=path.name,
            load_module_strict=False,
            load_optimizer_states=True,
            load_lr_scheduler_states=False,
        )
        if load_path is None:
            raise RuntimeError(f"DeepSpeed could not load checkpoint: {path}")
        checkpoint = client_state
    else:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        parameters = dict(accelerator.unwrap_model(training_model).named_parameters())
        for name, value in checkpoint["trainable_state_dict"].items():
            if name not in parameters:
                raise RuntimeError(f"Checkpoint parameter is not present: {name}")
            parameters[name].data.copy_(value.to(parameters[name].device, dtype=parameters[name].dtype))
        optimizer.load_state_dict(checkpoint["optimizer"])

    if checkpoint["stage"] != cfg["stage"]:
        raise RuntimeError(f"Checkpoint stage {checkpoint['stage']} does not match {cfg['stage']}.")
    return int(checkpoint["step"])


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if args.stage:
        cfg["stage"] = args.stage
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.resume_from is not None:
        cfg["resume_from"] = args.resume_from
    if args.trainable_components is not None:
        cfg["sla"]["trainable_components"] = args.trainable_components
    if args.training_backend is not None:
        cfg["sla"]["training_backend"] = args.training_backend
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.micro_batch_size is not None:
        cfg["train_micro_batch_size_per_gpu"] = args.micro_batch_size
    if args.activation_checkpointing is not None:
        cfg["activation_checkpointing"] = args.activation_checkpointing
    if args.validation is not None:
        cfg.setdefault("validation", {})["enabled"] = args.validation
        if "rollout_validation" in cfg:
            cfg["rollout_validation"]["enabled"] = args.validation

    import accelerate

    accelerate.utils.set_seed(int(cfg["seed"]), device_specific=False)

    # Keep a standard DataLoader batch size so Accelerate can shard complete
    # per-rank micro batches and DeepSpeed can infer consistent batch semantics.
    accelerator = create_accelerator(accelerate, cfg["gradient_accumulation_steps"])
    using_deepspeed = configure_deepspeed_micro_batch(
        accelerator, int(cfg.get("train_micro_batch_size_per_gpu", 1))
    )
    device = accelerator.device
    if accelerator.is_main_process:
        print(f"distributed_type={accelerator.distributed_type} world_size={accelerator.num_processes}")
        if using_deepspeed:
            value = accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"]
            print(f"deepspeed_train_micro_batch_size_per_gpu={value}")
            param_device, optimizer_device = deepspeed_offload_devices(accelerator)
            print(
                f"deepspeed_offload_param_device={param_device} "
                f"deepspeed_offload_optimizer_device={optimizer_device}"
            )
    if device.type != "npu":
        raise RuntimeError(f"This entrypoint requires an Ascend NPU, got {device}.")
    micro_batch_size = int(cfg.get("train_micro_batch_size_per_gpu", 1))
    trajectory_mode = "trajectory_dir" in cfg["data"]
    if trajectory_mode:
        dataset = HunyuanTrajectoryDataset(cfg["data"]["trajectory_dir"], dtype=cfg["dtype"])
        dataset.prepare_exact_length_batches(micro_batch_size, int(cfg["seed"]))
        latent_mode = False
    elif "cache_dir" in cfg["data"]:
        dataset = HunyuanLatentDataset(cfg["data"]["cache_dir"], split=cfg["data"].get("split", "train"))
        dataset.prepare_exact_length_batches(micro_batch_size, int(cfg["seed"]))
        latent_mode = True
    else:
        dataset = SerializedModelInputs(cfg["data"]["serialized_inputs_glob"])
        latent_mode = False
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=micro_batch_size,
        collate_fn=(
            collate_latent_records
            if latent_mode
            else collate_trajectory_records
            if trajectory_mode
            else unwrap_single_record
        ),
        shuffle=False if (latent_mode or trajectory_mode) else True,
        num_workers=cfg["data"]["num_workers"],
    )
    validation_cfg = cfg.get("validation", {}) or {}
    validation_enabled = bool(validation_cfg.get("enabled", False))
    validation_dataloader = None
    if validation_enabled:
        if not trajectory_mode:
            raise ValueError("Cached badcase validation is only supported with trajectory training data.")
        validation_batch_size = int(validation_cfg.get("micro_batch_size_per_gpu", 1))
        if validation_batch_size != 1:
            raise ValueError("validation.micro_batch_size_per_gpu must be 1 for per-step metrics.")
        validation_dataset = HunyuanTrajectoryDataset(
            validation_cfg["trajectory_dir"],
            dtype=cfg["dtype"],
            max_prompts=int(validation_cfg.get("num_prompts", 4)),
        )
        expected_prompts = int(validation_cfg.get("num_prompts", 4))
        if len(validation_dataset.rows) != expected_prompts:
            raise RuntimeError(
                f"Validation requires exactly {expected_prompts} prompt(s), "
                f"but {validation_cfg['trajectory_dir']} contains {len(validation_dataset.rows)}."
            )
        if len(validation_dataset) % accelerator.num_processes:
            raise RuntimeError(
                f"Validation has {len(validation_dataset)} trajectory point(s), which is not divisible "
                f"by world_size={accelerator.num_processes}; this would duplicate validation records."
            )
        validation_dataloader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=validation_batch_size,
            collate_fn=collate_trajectory_records,
            shuffle=False,
            num_workers=int(validation_cfg.get("num_workers", 0)),
        )
    rollout_cfg = cfg.get("rollout_validation", {}) or {}
    rollout_enabled = bool(rollout_cfg.get("enabled", False))
    rollout_dataloader = None
    if rollout_enabled:
        if not trajectory_mode:
            raise ValueError("Free rollout validation requires trajectory training data.")
        if accelerator.num_processes != 16:
            raise RuntimeError(
                "Configured free rollout validation is restricted to world_size=16; "
                f"got {accelerator.num_processes}."
            )
        rollout_batch_size = int(rollout_cfg.get("micro_batch_size_per_gpu", 1))
        if rollout_batch_size != 1:
            raise ValueError("rollout_validation.micro_batch_size_per_gpu must be 1.")
        rollout_dataset = HunyuanTrajectoryRolloutDataset(
            rollout_cfg["trajectory_dir"],
            dtype=cfg["dtype"],
            max_prompts=int(rollout_cfg.get("num_prompts", 20)),
            world_size=accelerator.num_processes,
        )
        rollout_dataloader = torch.utils.data.DataLoader(
            rollout_dataset,
            batch_size=rollout_batch_size,
            collate_fn=collate_rollout_records,
            shuffle=False,
            num_workers=int(rollout_cfg.get("num_workers", 0)),
        )
    if accelerator.is_main_process and latent_mode:
        print(
            f"latent_micro_batch_size={micro_batch_size} usable_samples={len(dataset)} "
            f"dropped_for_exact_length_batching={dataset.dropped_for_batching}"
        )
    if accelerator.is_main_process and trajectory_mode:
        print(
            f"trajectory_micro_batch_size={micro_batch_size} usable_points={len(dataset)} "
            f"dropped_for_exact_layout_batching={dataset.dropped_for_batching}"
        )
        if validation_dataloader is not None:
            print(
                f"validation_prompts={len(validation_dataset.rows)} "
                f"validation_points={len(validation_dataset)} "
                f"validation_every_steps={validation_cfg.get('every_steps', 0)}"
            )
        if rollout_dataloader is not None:
            print(
                f"rollout_validation_prompts={len(rollout_dataset.rows)} "
                f"rollout_validation_slots={len(rollout_dataset)} "
                f"rollout_padding_prompts={rollout_dataset.padding_prompts} "
                f"rollout_every_steps={rollout_cfg.get('every_steps', 0)}"
            )
    # With ZeRO-3, Transformers/DeepSpeed constructs partitioned parameters while
    # loading. Moving the complete model to one NPU here would defeat that path.
    model = load_hunyuan(
        cfg["model_path"],
        None if _using_deepspeed(accelerator) else device,
        cfg["dtype"],
        skip_load_modules=cfg.get("skip_load_modules", ()),
    )

    if cfg["stage"] == "dense":
        training_model = DenseForwardBackwardModule(model, cfg["dense_trainable_patterns"])
    else:
        training_model = HunyuanSLARecoveryModule(
            model,
            activation_checkpointing=cfg.get("activation_checkpointing", True),
            moe_lora=cfg.get("moe_lora"),
            **cfg["sla"],
        )
    parameter_groups = training_model.trainable_parameter_groups()
    trainable = [parameter for parameters in parameter_groups.values() for parameter in parameters]
    if not trainable:
        raise RuntimeError("No trainable parameters selected.")
    trainable_dtypes = sorted({str(parameter.dtype) for parameter in trainable})
    trainable_elements = _logical_parameter_count(trainable)
    expected_trainable = cfg.get("expected_trainable_parameters")
    if expected_trainable is not None and trainable_elements != int(expected_trainable):
        raise RuntimeError(
            f"Trainable parameter audit failed: expected={int(expected_trainable)}, "
            f"actual={trainable_elements}."
        )
    if using_deepspeed and len(trainable_dtypes) != 1:
        raise RuntimeError(
            "ZeRO-3 requires a uniform trainable parameter dtype for its flat buffer; "
            f"got {trainable_dtypes}."
        )
    if accelerator.is_main_process:
        print(f"trainable_parameter_dtypes={trainable_dtypes}")
        print(f"trainable_parameter_elements={trainable_elements}")
    learning_rates = cfg.get("learning_rates", {}) or {}
    optimizer_group_names = [name for name, parameters in parameter_groups.items() if parameters]
    optimizer_groups = [
        {
            "params": parameters,
            "lr": float(learning_rates.get(name, cfg["learning_rate"])),
        }
        for name, parameters in parameter_groups.items()
        if parameters
    ]
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    prepare_objects = [training_model, optimizer, dataloader]
    if validation_dataloader is not None:
        prepare_objects.append(validation_dataloader)
    if rollout_dataloader is not None:
        prepare_objects.append(rollout_dataloader)
    prepared = list(accelerator.prepare(*prepare_objects))
    training_model, optimizer, dataloader = prepared[:3]
    prepared_index = 3
    if validation_dataloader is not None:
        validation_dataloader = prepared[prepared_index]
        prepared_index += 1
    if rollout_dataloader is not None:
        rollout_dataloader = prepared[prepared_index]

    step = 0
    resume_from = cfg.get("resume_from")
    if resume_from:
        step = load_checkpoint(accelerator, training_model, optimizer, cfg, resume_from)
        if accelerator.is_main_process:
            print(f"resumed_from={resume_from} step={step}")

    metrics_cfg = cfg.get("logging", {}) or {}
    metrics_every_steps = int(metrics_cfg.get("metrics_every_steps", 1))
    plot_every_steps = int(metrics_cfg.get("plot_every_steps", 5))
    metrics_logger = None
    metrics_dir = resolve_output_dir(ROOT, cfg["output_dir"]) / "metrics"
    if accelerator.is_main_process:
        metrics_logger = MetricsLogger(
            metrics_dir,
            resume_step=step,
            ema_decay=float(metrics_cfg.get("ema_decay", 0.9)),
        )

    if accelerator.is_main_process:
        unwrapped_training_model = accelerator.unwrap_model(training_model)
        names = unwrapped_training_model.trainable_parameter_names()
        print(f"stage={cfg['stage']} trainable_parameters={len(names)}")
        if cfg["stage"] == "sla":
            print(f"activation_checkpointed_layers={unwrapped_training_model.checkpointed_layers}")
            print(
                f"sla_training_backend={cfg['sla'].get('training_backend', 'auto')} "
                f"sla_trainable_components={cfg['sla'].get('trainable_components', ['proj_l'])}"
            )
            if unwrapped_training_model.moe_lora is not None:
                print(f"moe_down_lora_geometry={unwrapped_training_model.moe_lora.geometry}")
        print("\n".join(names[:16]))

    training_model.train()
    batches_per_epoch = len(dataloader)
    schedule = build_training_schedule(
        completed_steps=step,
        max_steps=int(cfg["max_steps"]),
        batches_per_epoch=batches_per_epoch,
        configured_epochs=int(cfg["num_epochs"]),
    )
    start_epoch = schedule.start_epoch
    skip_batches = schedule.skip_batches
    if accelerator.is_main_process:
        print(
            f"batches_per_epoch={schedule.batches_per_epoch} configured_epochs={cfg['num_epochs']} "
            f"effective_epochs={schedule.effective_epochs} max_steps={cfg['max_steps']}"
        )
    last_checkpoint_step = None
    save_every_steps = int(cfg.get("save_every_steps", 0))
    progress = tqdm(
        total=int(cfg["max_steps"]),
        initial=min(step, int(cfg["max_steps"])),
        desc=f"{cfg['stage']} training",
        unit="step",
        dynamic_ncols=True,
        disable=not accelerator.is_main_process,
        file=sys.stdout,
    )
    for epoch in range(start_epoch, schedule.effective_epochs):
        for batch_index, batch in enumerate(dataloader):
            if step >= cfg["max_steps"]:
                break
            if epoch == start_epoch and batch_index < skip_batches:
                continue
            step_started = time.perf_counter()
            torch.npu.reset_peak_memory_stats(device)
            batch = move(batch, device)
            if latent_mode:
                noise_cfg = cfg["noise"]
                x_t, timestep, timestep_r = flow_match_batch(
                    batch["latent_z0"],
                    batch["sample_id"],
                    global_seed=int(cfg["seed"]),
                    epoch=epoch,
                    view=step,
                    sigma_min=noise_cfg["sigma_min"],
                    sigma_max=noise_cfg["sigma_max"],
                    train_timesteps=noise_cfg["train_timesteps"],
                )
                guidance = batch["latent_z0"].new_full(
                    (micro_batch_size,), 1000.0 * float(cfg["conditioning"]["guidance_scale"])
                )
                batch = model_kwargs_from_latent(
                    batch,
                    x_t,
                    timestep,
                    timestep_r=timestep_r,
                    guidance=guidance,
                )
            teacher_prediction = batch.pop("teacher_diffusion_prediction", None) if trajectory_mode else None
            full_attention_spans = batch.pop("full_attention_spans", None) if trajectory_mode else None
            if trajectory_mode:
                batch.pop("sample_id", None)
                batch.pop("trajectory_step", None)
            with accelerator.accumulate(training_model):
                loss = training_model(
                    batch,
                    teacher_prediction=teacher_prediction,
                    full_attention_spans=full_attention_spans,
                )
                if accelerator.is_main_process:
                    progress.write(f"step={step + 1} phase=backward")
                accelerator.backward(loss)
                gradient_getter = deepspeed_local_gradient if using_deepspeed else None
                unwrapped = accelerator.unwrap_model(training_model)
                current_groups = unwrapped.trainable_parameter_groups()
                local_group_stats = [
                    inspect_local_gradients(parameters, gradient_getter)
                    for parameters in current_groups.values()
                ]
                packed_stats = torch.tensor(
                    [
                        value
                        for stats in local_group_stats
                        for value in (stats.element_count, stats.nonfinite_count, stats.squared_norm)
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                reduced_stats = accelerator.reduce(packed_stats, reduction="sum").reshape(-1, 3)
                group_gradients = {}
                for group_index, group_name in enumerate(current_groups):
                    elements = int(reduced_stats[group_index, 0].item())
                    nonfinite = int(reduced_stats[group_index, 1].item())
                    norm = math.sqrt(float(reduced_stats[group_index, 2].item())) if nonfinite == 0 else math.nan
                    if elements == 0:
                        raise RuntimeError(f"No gradient was produced for trainable group {group_name!r}.")
                    if nonfinite:
                        raise RuntimeError(
                            f"Found {nonfinite} non-finite gradient values in trainable group {group_name!r}."
                        )
                    if norm <= 0.0:
                        raise RuntimeError(f"Trainable group {group_name!r} produced a zero gradient norm.")
                    group_gradients[group_name] = (elements, norm)
                gradient_elements = sum(elements for elements, _ in group_gradients.values())
                gradient_norm = math.sqrt(sum(norm * norm for _, norm in group_gradients.values()))
                if accelerator.is_main_process:
                    progress.write(f"step={step + 1} phase=optimizer")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            step += 1
            global_loss = accelerator.reduce(loss.detach().float(), reduction="mean")
            local_peak_memory = torch.tensor([float(_peak_npu_memory(device))], device=device)
            peak_memory = float(accelerator.gather(local_peak_memory).max().item())
            step_seconds = time.perf_counter() - step_started
            global_batch_size = micro_batch_size * accelerator.num_processes
            validation_metrics: dict[str, Any] = {}
            validation_every_steps = int(validation_cfg.get("every_steps", 0))
            if (
                validation_dataloader is not None
                and validation_every_steps > 0
                and step % validation_every_steps == 0
            ):
                if accelerator.is_main_process:
                    progress.write(f"step={step} phase=validation")
                validation_metrics = evaluate_validation(
                    accelerator, training_model, validation_dataloader, device
                )
            rollout_every_steps = int(rollout_cfg.get("every_steps", 0))
            if (
                rollout_dataloader is not None
                and rollout_every_steps > 0
                and step % rollout_every_steps == 0
            ):
                if accelerator.is_main_process:
                    progress.write(f"step={step} phase=free_rollout_validation")
                validation_metrics.update(
                    evaluate_rollout(
                        accelerator,
                        training_model,
                        rollout_dataloader,
                        device,
                        compute_dtype={
                            "bf16": torch.bfloat16,
                            "fp16": torch.float16,
                            "fp32": torch.float32,
                        }[cfg["dtype"]],
                    )
                )
            if accelerator.is_main_process:
                loss_value = float(global_loss.item())
                progress.set_postfix(loss=f"{loss_value:.6f}", grad_norm=f"{gradient_norm:.3e}")
                progress.update(1)
                progress.write(
                    f"step={step} loss={loss_value:.8f} gradient_elements={gradient_elements} "
                    f"gradient_norm={gradient_norm:.8e} finite_grad=True "
                    + " ".join(
                        f"{name}_grad_elements={elements} {name}_grad_norm={norm:.8e}"
                        for name, (elements, norm) in group_gradients.items()
                    )
                )
                if validation_metrics:
                    progress.write(
                        f"step={step} validation_mse={validation_metrics['validation_mse']:.8f} "
                        f"validation_relative_mse={validation_metrics['validation_relative_mse']:.8f} "
                        f"validation_cosine_distance="
                        f"{validation_metrics['validation_cosine_distance']:.8f}"
                    )
                    if "rollout_final_latent_relative_mse" in validation_metrics:
                        progress.write(
                            f"step={step} rollout_final_latent_relative_mse="
                            f"{validation_metrics['rollout_final_latent_relative_mse']:.8f} "
                            f"rollout_final_latent_cosine_distance="
                            f"{validation_metrics['rollout_final_latent_cosine_distance']:.8f} "
                            f"rollout_final_laplacian_relative_mse="
                            f"{validation_metrics['rollout_final_laplacian_relative_mse']:.8f}"
                        )
                if metrics_every_steps > 0 and step % metrics_every_steps == 0:
                    record: dict[str, Any] = {
                        "step": step,
                        "epoch": epoch,
                        "loss": loss_value,
                        "gradient_norm": gradient_norm,
                        "gradient_elements": gradient_elements,
                        "step_seconds": step_seconds,
                        "samples_per_second": global_batch_size / max(step_seconds, 1.0e-9),
                        "peak_npu_memory_bytes": int(peak_memory),
                        "learning_rates": {
                            name: float(group["lr"])
                            for name, group in zip(optimizer_group_names, optimizer.param_groups)
                        },
                    }
                    for name, (elements, norm) in group_gradients.items():
                        record[f"{name}_grad_elements"] = elements
                        record[f"{name}_grad_norm"] = norm
                    record.update(validation_metrics)
                    metrics_logger.append(record)
                    if plot_every_steps > 0 and (
                        step % plot_every_steps == 0 or step >= int(cfg["max_steps"])
                    ):
                        try:
                            from tools.plot_training_metrics import plot_metrics

                            plot_metrics(
                                metrics_logger.path,
                                metrics_dir / "training_metrics.png",
                                metrics_dir / "index.html",
                            )
                        except ImportError as exc:
                            progress.write(f"metrics_plot_skipped={exc}")
            if save_every_steps > 0 and step % save_every_steps == 0 and accelerator.sync_gradients:
                save_checkpoint(accelerator, training_model, optimizer, dataset, cfg, step)
                last_checkpoint_step = step
            if step >= cfg["max_steps"]:
                break
        if step >= cfg["max_steps"]:
            break

    progress.close()
    if last_checkpoint_step != step:
        save_checkpoint(accelerator, training_model, optimizer, dataset, cfg, step)


if __name__ == "__main__":
    main()
