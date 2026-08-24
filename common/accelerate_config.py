"""Version-compatible Accelerate construction for pre-batched cache records."""


def create_accelerator(accelerate_module, gradient_accumulation_steps: int):
    kwargs = {"gradient_accumulation_steps": gradient_accumulation_steps}
    if hasattr(accelerate_module, "DataLoaderConfiguration"):
        kwargs["dataloader_config"] = accelerate_module.DataLoaderConfiguration(even_batches=False)
    else:
        # Accelerate < 0.30 exposed this setting directly on Accelerator.
        kwargs["even_batches"] = False
    return accelerate_module.Accelerator(**kwargs)
