#!/usr/bin/env python3
"""Single- and multi-NPU Dense/SLA recovery training entrypoint."""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "upstream" / "DiffSynth-Studio"))

from diffsynth.diffusion import DiffusionTrainingModule
from hunyuan_adapter import HunyuanSLARecoveryModule, freeze_model, load_hunyuan, unfreeze_matching


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
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if args.stage:
        cfg["stage"] = args.stage
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps

    import accelerate

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=cfg["gradient_accumulation_steps"])
    device = accelerator.device
    if device.type != "npu":
        raise RuntimeError(f"This entrypoint requires an Ascend NPU, got {device}.")
    dataset = SerializedModelInputs(cfg["data"]["serialized_inputs_glob"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=True, num_workers=cfg["data"]["num_workers"])
    model = load_hunyuan(cfg["model_path"], device, cfg["dtype"])

    if cfg["stage"] == "dense":
        training_model = DenseForwardBackwardModule(model, cfg["dense_trainable_patterns"])
    else:
        training_model = HunyuanSLARecoveryModule(model, **cfg["sla"])
    trainable = [p for p in training_model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters selected.")
    optimizer = torch.optim.AdamW(trainable, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    training_model, optimizer, dataloader = accelerator.prepare(training_model, optimizer, dataloader)

    if accelerator.is_main_process:
        names = accelerator.unwrap_model(training_model).trainable_parameter_names()
        print(f"stage={cfg['stage']} trainable_parameters={len(names)}")
        print("\n".join(names[:16]))

    training_model.train()
    step = 0
    for batch in dataloader:
        batch = move(batch, device)
        with accelerator.accumulate(training_model):
            loss = training_model(batch)
            accelerator.backward(loss)
            gradients = [p.grad for p in training_model.parameters() if p.requires_grad]
            has_finite_grad = any(g is not None and torch.isfinite(g).all() for g in gradients)
            if not has_finite_grad:
                raise RuntimeError("No finite gradient was produced for the selected trainable parameters.")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        step += 1
        if accelerator.is_main_process:
            print(f"step={step} loss={loss.detach().float().item():.8f} finite_grad={has_finite_grad}")
        if step >= cfg["max_steps"]:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(training_model)
        output_dir = Path(cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "step": step,
            "stage": cfg["stage"],
            "trainable_parameter_names": unwrapped.trainable_parameter_names(),
            "trainable_state_dict": {name: p.detach().cpu() for name, p in unwrapped.named_parameters() if p.requires_grad},
            "optimizer": optimizer.state_dict(),
            "config": cfg,
        }
        path = output_dir / f"{cfg['stage']}-step-{step}.pt"
        torch.save(checkpoint, path)
        print(f"checkpoint={path}")


if __name__ == "__main__":
    main()
