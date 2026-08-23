"""Minimal Hunyuan model loading shared by sampling and training."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "upstream" / "HunyuanImage-3.0"
if SOURCE.is_dir() and str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))


def dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def load_hunyuan(model_path: str, device: torch.device | None, dtype: str) -> nn.Module:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=dtype_from_name(dtype), low_cpu_mem_usage=True
    )
    if device is not None:
        model = model.to(device)
    return model
