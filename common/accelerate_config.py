"""Accelerate and DeepSpeed configuration helpers."""


def create_accelerator(accelerate_module, gradient_accumulation_steps: int):
    return accelerate_module.Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps
    )


def configure_deepspeed_micro_batch(accelerator, micro_batch_size: int) -> bool:
    """Set pre-batched DataLoader semantics on the active DeepSpeed plugin."""
    plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    if plugin is None:
        return False
    if micro_batch_size < 1:
        raise ValueError("train_micro_batch_size_per_gpu must be at least 1.")
    plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = int(micro_batch_size)
    return True


def deepspeed_offload_devices(accelerator) -> tuple[str, str] | None:
    plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    if plugin is None:
        return None
    config = plugin.deepspeed_config
    zero = config.get("zero_optimization", {})

    def device(name: str) -> str:
        value = zero.get(name, config.get(name, config.get(f"{name}_device", "none")))
        if isinstance(value, dict):
            value = value.get("device", "none")
        return str(value).lower()

    return device("offload_param"), device("offload_optimizer")
