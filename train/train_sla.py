#!/usr/bin/env python3
"""Single- and multi-NPU Dense/SLA recovery training entrypoint."""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
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
from common.checkpoint import prepare_rank_checkpoint_dir, resolve_output_dir
from common.gradient import deepspeed_local_gradient, inspect_local_gradients
from common.hunyuan import prepare_diffusion_runtime, redirect_legacy_cuda_runtime
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
from trajectory_dataset import HunyuanTrajectoryDataset


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
        choices=("proj_l", "qkv_delta", "o_delta"),
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

    import accelerate

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
        if micro_batch_size != 1:
            raise ValueError("Trajectory recovery currently requires train_micro_batch_size_per_gpu=1.")
        dataset = HunyuanTrajectoryDataset(cfg["data"]["trajectory_dir"], dtype=cfg["dtype"])
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
        collate_fn=collate_latent_records if latent_mode else unwrap_single_record,
        shuffle=False if latent_mode else True,
        num_workers=cfg["data"]["num_workers"],
    )
    if accelerator.is_main_process and latent_mode:
        print(
            f"latent_micro_batch_size={micro_batch_size} usable_samples={len(dataset)} "
            f"dropped_for_exact_length_batching={dataset.dropped_for_batching}"
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
            **cfg["sla"],
        )
    parameter_groups = training_model.trainable_parameter_groups()
    trainable = [parameter for parameters in parameter_groups.values() for parameter in parameters]
    if not trainable:
        raise RuntimeError("No trainable parameters selected.")
    trainable_dtypes = sorted({str(parameter.dtype) for parameter in trainable})
    if using_deepspeed and len(trainable_dtypes) != 1:
        raise RuntimeError(
            "ZeRO-3 requires a uniform trainable parameter dtype for its flat buffer; "
            f"got {trainable_dtypes}."
        )
    if accelerator.is_main_process:
        print(f"trainable_parameter_dtypes={trainable_dtypes}")
    learning_rates = cfg.get("learning_rates", {}) or {}
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
    training_model, optimizer, dataloader = accelerator.prepare(training_model, optimizer, dataloader)

    step = 0
    resume_from = cfg.get("resume_from")
    if resume_from:
        step = load_checkpoint(accelerator, training_model, optimizer, cfg, resume_from)
        if accelerator.is_main_process:
            print(f"resumed_from={resume_from} step={step}")

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
            if accelerator.is_main_process:
                loss_value = loss.detach().float().item()
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
