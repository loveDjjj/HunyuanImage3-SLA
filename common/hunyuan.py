"""Minimal Hunyuan model loading shared by sampling and training."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "upstream" / "HunyuanImage-3.0"
if SOURCE.is_dir() and str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))


def dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _current_accelerator_device() -> torch.device:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device(f"npu:{torch.npu.current_device()}")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


@contextmanager
def redirect_legacy_cuda_empty(device: torch.device | None = None):
    """Redirect Hunyuan VAE's constructor-only CUDA sentinel off CUDA builds."""
    target = device or _current_accelerator_device()
    original_empty = torch.empty

    def empty_on_current_device(*args, **kwargs):
        if str(kwargs.get("device")).startswith("cuda") and target.type != "cuda":
            kwargs["device"] = target
        return original_empty(*args, **kwargs)

    torch.empty = empty_on_current_device
    try:
        yield
    finally:
        torch.empty = original_empty


def load_hunyuan(
    model_path: str,
    device: torch.device | None,
    dtype: str,
    skip_load_modules: Iterable[str] = (),
) -> nn.Module:
    from transformers import AutoModelForCausalLM

    # Offline-latent training skips VAE/ViT through the upstream constructor API.
    # The scoped redirect remains a fallback for checkpoints that still construct
    # the VAE's hard-coded CUDA empty sentinel.
    with redirect_legacy_cuda_empty(device):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype_from_name(dtype),
            low_cpu_mem_usage=True,
            skip_load_module=set(skip_load_modules),
        )
    if device is not None:
        model = model.to(device)
    return model
