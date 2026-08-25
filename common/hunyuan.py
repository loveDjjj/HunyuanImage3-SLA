"""Minimal Hunyuan model loading shared by sampling and training."""

from __future__ import annotations

import sys
from contextlib import contextmanager, nullcontext
from inspect import signature
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "upstream" / "HunyuanImage-3.0"
if SOURCE.is_dir() and str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))


def dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def infer_hunyuan_model_version(config: Any, model_path: str | Path) -> str:
    """Infer tokenizer layout for released checkpoints missing model_version."""
    if isinstance(config, dict):
        configured = config.get("model_version")
        cfg_distilled = config.get("cfg_distilled", False)
    else:
        configured = getattr(config, "model_version", None)
        cfg_distilled = getattr(config, "cfg_distilled", False)
    if configured:
        return str(configured)
    if cfg_distilled or "instruct" in Path(model_path).name.lower():
        return "HunyuanImage-3.0-Instruct"
    return "HunyuanImage-3.0"


def prepare_diffusion_runtime(model: nn.Module, model_kwargs: dict[str, Any]) -> None:
    """Initialize state normally populated by Hunyuan's generation wrapper."""
    image_token_counts = model_kwargs["image_mask"].sum(dim=1)
    if not torch.equal(image_token_counts, image_token_counts[:1].expand_as(image_token_counts)):
        raise ValueError("A recovery micro batch must use one image token count.")
    model.post_token_len = None
    model.num_image_tokens = int(image_token_counts[0].item())
    model.num_special_tokens = sum(
        model_kwargs.get(name) is not None
        for name in ("timesteps_index", "guidance_index", "timesteps_r_index")
    )


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


class _TorchDeviceRedirect:
    """Proxy a remote-code torch module without touching DeepSpeed's globals."""

    def __init__(self, target: torch.device):
        self._target = target

    def __getattr__(self, name: str):
        return getattr(torch, name)

    def empty(self, *args, **kwargs):
        if str(kwargs.get("device")).startswith("cuda") and self._target.type != "cuda":
            kwargs["device"] = self._target
        # Resolve this at call time because ZeRO-3 replaces torch.empty while
        # constructing partitioned parameters.
        return torch.empty(*args, **kwargs)


def _remote_vae_modules(model_class: type[nn.Module]) -> list[ModuleType]:
    model_module = sys.modules.get(model_class.__module__)
    if model_module is None:
        return []
    modules = []
    for class_name in ("AutoencoderKLConv3D", "AutoencoderKLConv3D_Dist"):
        vae_class = getattr(model_module, class_name, None)
        vae_module = sys.modules.get(getattr(vae_class, "__module__", ""))
        if vae_module is not None and vae_module not in modules:
            modules.append(vae_module)
    return modules


@contextmanager
def redirect_remote_vae_cuda_empty(
    model_class: type[nn.Module], device: torch.device | None = None
):
    """Redirect the remote VAE sentinel after ZeRO replaces torch.empty."""
    target = device or _current_accelerator_device()
    modules = _remote_vae_modules(model_class)
    originals = [(module, module.torch) for module in modules]
    for module, _ in originals:
        module.torch = _TorchDeviceRedirect(target)
    try:
        yield
    finally:
        for module, original in originals:
            module.torch = original


def patch_remote_static_cache(model: nn.Module) -> bool:
    """Bridge Hunyuan's legacy cache update to current Transformers layers."""
    model_module = sys.modules.get(model.__class__.__module__)
    cache_class = getattr(model_module, "HunyuanStaticCache", None)
    if cache_class is None or getattr(cache_class, "_sla_cache_compat", False):
        return False

    original_update = cache_class.update

    def compatible_update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        layer = self.layers[layer_idx]
        if layer.keys is None:
            lazy_initialization = layer.lazy_initialization
            # Bound methods omit ``self``. Transformers <=4.49 accepts only
            # key_states; current StaticLayer also requires value_states.
            if len(signature(lazy_initialization).parameters) >= 2:
                lazy_initialization(key_states, value_states)
            else:
                lazy_initialization(key_states)
        return original_update(self, key_states, value_states, layer_idx, cache_kwargs)

    cache_class.update = compatible_update
    cache_class._sla_cache_compat = True
    return True


@contextmanager
def redirect_legacy_cuda_runtime(device: torch.device | None = None):
    """Map upstream MoE's CUDA-only device/profiling calls during NPU forward."""
    target = device or _current_accelerator_device()
    if target.type == "cuda":
        yield
        return

    original_set_device = torch.cuda.set_device
    original_nvtx_range = torch.cuda.nvtx.range

    def set_current_device(index):
        if target.type == "npu":
            torch.npu.set_device(index)

    torch.cuda.set_device = set_current_device
    torch.cuda.nvtx.range = lambda _message: nullcontext()
    try:
        yield
    finally:
        torch.cuda.set_device = original_set_device
        torch.cuda.nvtx.range = original_nvtx_range


def load_hunyuan(
    model_path: str,
    device: torch.device | None,
    dtype: str,
    skip_load_modules: Iterable[str] = (),
) -> nn.Module:
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if not getattr(config, "model_version", None):
        config.model_version = infer_hunyuan_model_version(config, model_path)
    class_reference = (getattr(config, "auto_map", None) or {}).get("AutoModelForCausalLM")
    if class_reference:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        if isinstance(class_reference, (list, tuple)):
            class_reference = class_reference[0]
        model_class = get_class_from_dynamic_module(class_reference, model_path)
    else:
        model_class = None

    # Offline-latent training skips VAE/ViT through the upstream constructor API.
    # The scoped redirect remains a fallback for checkpoints that still construct
    # the VAE's hard-coded CUDA empty sentinel.
    loader = model_class or AutoModelForCausalLM
    load_kwargs = {
        "config": config,
        "torch_dtype": dtype_from_name(dtype),
        "low_cpu_mem_usage": True,
        # Transformers also forwards model kwargs while resolving the
        # generation config, so this must remain JSON serializable.
        "skip_load_module": list(skip_load_modules),
    }
    if model_class is None:
        load_kwargs["trust_remote_code"] = True
    with redirect_legacy_cuda_empty(device), redirect_remote_vae_cuda_empty(loader, device):
        model = loader.from_pretrained(
            model_path,
            **load_kwargs,
        )
    patch_remote_static_cache(model)
    if device is not None:
        model = model.to(device)
    return model
