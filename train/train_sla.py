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
from common.accelerate_config import configure_deepspeed_micro_batch, create_accelerator
from common.gradient import deepspeed_local_gradient, inspect_local_gradients
from common.hunyuan import prepare_diffusion_runtime, redirect_legacy_cuda_runtime
from hunyuan_adapter import (
    HunyuanSLARecoveryModule,
    freeze_model,
    load_hunyuan,
    unfreeze_matching,
)
from latent_dataset import HunyuanLatentDataset, model_kwargs_from_latent, unwrap_single_record
from noise_sampler import flow_match_input, sample_seed


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "train_sla.yaml"))
    parser.add_argument("--stage", choices=("dense", "sla"), default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume-from", default=None, help="Checkpoint written by this entrypoint")
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
    output_dir = Path(cfg["output_dir"])
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    if _using_deepspeed(accelerator):
        tag = f"{cfg['stage']}-step-{step}"
        # DeepSpeed checkpointing is collective. Excluding frozen parameters keeps
        # the restart data focused on SLA parameters and their optimizer shards.
        training_model.save_checkpoint(
            str(output_dir),
            tag=tag,
            client_state=_checkpoint_client_state(cfg, dataset, step),
            save_latest=True,
            exclude_frozen_parameters=True,
        )
        path = output_dir / tag
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

    import accelerate

    # Each latent-cache record is already one complete batch (batch_size=None).
    # Accelerate cannot pad such a batch sampler to an even number of batches.
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
    if device.type != "npu":
        raise RuntimeError(f"This entrypoint requires an Ascend NPU, got {device}.")
    if "cache_dir" in cfg["data"]:
        dataset = HunyuanLatentDataset(cfg["data"]["cache_dir"], split=cfg["data"].get("split", "train"))
        latent_mode = True
    else:
        dataset = SerializedModelInputs(cfg["data"]["serialized_inputs_glob"])
        latent_mode = False
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn=unwrap_single_record,
        shuffle=not latent_mode,
        num_workers=cfg["data"]["num_workers"],
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
    trainable = [p for p in training_model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters selected.")
    optimizer = torch.optim.AdamW(trainable, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
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
        print("\n".join(names[:16]))

    training_model.train()
    batches_per_epoch = len(dataloader)
    start_epoch, skip_batches = divmod(step, batches_per_epoch)
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
    for epoch in range(start_epoch, cfg["num_epochs"]):
        for batch_index, batch in enumerate(dataloader):
            if epoch == start_epoch and batch_index < skip_batches:
                continue
            batch = move(batch, device)
            if latent_mode:
                noise_cfg = cfg["noise"]
                seed = sample_seed(int(cfg["seed"]), str(batch["sample_id"]), epoch, step)
                x_t, timestep, timestep_r = flow_match_input(
                    batch["latent_z0"], seed, noise_cfg["sigma_min"], noise_cfg["sigma_max"], noise_cfg["train_timesteps"]
                )
                guidance = batch["latent_z0"].new_tensor(
                    1000.0 * float(cfg["conditioning"]["guidance_scale"])
                )
                batch = model_kwargs_from_latent(
                    batch,
                    x_t,
                    timestep,
                    timestep_r=timestep_r,
                    guidance=guidance,
                )
            with accelerator.accumulate(training_model):
                loss = training_model(batch)
                accelerator.backward(loss)
                trainable_parameters = [p for p in training_model.parameters() if p.requires_grad]
                gradient_getter = deepspeed_local_gradient if using_deepspeed else None
                local_grad = inspect_local_gradients(trainable_parameters, gradient_getter)
                gradient_totals = accelerator.reduce(
                    torch.tensor(
                        [local_grad.element_count, local_grad.nonfinite_count, local_grad.squared_norm],
                        dtype=torch.float32,
                        device=device,
                    ),
                    reduction="sum",
                )
                gradient_elements = int(gradient_totals[0].item())
                nonfinite_gradients = int(gradient_totals[1].item())
                gradient_norm = math.sqrt(float(gradient_totals[2].item())) if nonfinite_gradients == 0 else math.nan
                if gradient_elements == 0:
                    raise RuntimeError("No gradient was produced for the selected trainable parameters.")
                if nonfinite_gradients:
                    raise RuntimeError(f"Found {nonfinite_gradients} non-finite gradient values.")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            step += 1
            if accelerator.is_main_process:
                loss_value = loss.detach().float().item()
                progress.set_postfix(loss=f"{loss_value:.6f}", grad_norm=f"{gradient_norm:.3e}")
                progress.update(1)
                progress.write(
                    f"step={step} loss={loss_value:.8f} gradient_elements={gradient_elements} "
                    f"gradient_norm={gradient_norm:.8e} finite_grad=True"
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
