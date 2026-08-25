"""Minimal HunyuanImage-3.0 adapter used by the DiffSynth training loop."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
for source in (ROOT / "upstream" / "DiffSynth-Studio", ROOT / "upstream" / "MindIE-SD", ROOT / "upstream" / "HunyuanImage-3.0"):
    if source.is_dir() and str(source) not in sys.path:
        sys.path.insert(0, str(source))

from diffsynth.diffusion import DiffusionTrainingModule
from common.activation_checkpoint import enable_hunyuan_activation_checkpointing
from common.hunyuan import (
    dtype_from_name as _dtype,
    load_hunyuan,
    prepare_diffusion_runtime,
    redirect_legacy_cuda_runtime,
)
from sla_adapter import SLAReplacementManager


def freeze_model(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def unfreeze_matching(model: nn.Module, patterns: list[str]) -> list[str]:
    names = []
    for name, parameter in model.named_parameters():
        if any(pattern in name for pattern in patterns):
            parameter.requires_grad_(True)
            names.append(name)
    if not names:
        raise RuntimeError(f"No parameters match dense_trainable_patterns={patterns}")
    return names


def _tensor_leaves(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensor_leaves(item)


def diffusion_output(output: Any) -> Any:
    value = getattr(output, "diffusion_prediction", None)
    if value is None:
        raise RuntimeError("The Hunyuan forward did not return diffusion_prediction. Use mode='gen_image'.")
    return value


def mse_tree(student: Any, teacher: Any) -> torch.Tensor:
    student_tensors, teacher_tensors = list(_tensor_leaves(student)), list(_tensor_leaves(teacher))
    if not student_tensors or len(student_tensors) != len(teacher_tensors):
        raise RuntimeError("Teacher and student diffusion_prediction structures do not match.")
    return torch.stack([F.mse_loss(s, t.to(dtype=s.dtype)) for s, t in zip(student_tensors, teacher_tensors)]).mean()


class HunyuanSLARecoveryModule(DiffusionTrainingModule):
    """Model-level Dense teacher / SLA student recovery objective."""

    def __init__(
        self,
        model: nn.Module,
        *,
        topk: float,
        blkq: int,
        blkk: int,
        use_bf16: bool,
        training_backend: str = "auto",
        trainable_components: tuple[str, ...] = ("proj_l",),
        activation_checkpointing: bool = True,
        log_phases: bool = True,
    ):
        super().__init__()
        self.model = model
        self.log_phases = log_phases
        self.forward_step = 0
        freeze_model(self.model)
        self.checkpointed_layers = (
            enable_hunyuan_activation_checkpointing(self.model)
            if activation_checkpointing
            else 0
        )
        self.replacements = SLAReplacementManager(
            self.model,
            topk=topk,
            blkq=blkq,
            blkk=blkk,
            use_bf16=use_bf16,
            training_backend=training_backend,
            trainable_components=tuple(trainable_components),
        )
        for parameter in self.replacements.trainable_parameters():
            parameter.requires_grad_(True)

    def forward(self, model_kwargs: dict[str, Any]) -> torch.Tensor:
        self.forward_step += 1
        step = self.forward_step
        prepare_diffusion_runtime(self.model, model_kwargs)
        with redirect_legacy_cuda_runtime():
            self._log_phase(step, "dense_teacher_forward")
            with self.replacements.dense_teacher(), torch.no_grad():
                teacher = diffusion_output(self.model(**model_kwargs))
            self._log_phase(step, "sla_student_forward")
            student = diffusion_output(self.model(**model_kwargs))
        self._log_phase(step, "recovery_loss")
        return mse_tree(student, teacher)

    def _log_phase(self, step: int, phase: str) -> None:
        if self.log_phases and int(os.environ.get("RANK", "0")) == 0:
            print(f"step={step} phase={phase}", flush=True)

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, p in self.named_parameters() if p.requires_grad]

    def trainable_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        return self.replacements.trainable_parameter_groups()
