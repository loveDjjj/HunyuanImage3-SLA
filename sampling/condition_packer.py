"""Serialize the static Hunyuan generation condition for a latent cache record."""

from __future__ import annotations

from typing import Any

import torch


STATIC_TENSOR_NAMES = (
    "input_ids",
    "image_mask",
    "timesteps_index",
    "guidance_index",
    "timesteps_r_index",
    "gen_timestep_scatter_index",
)


def _batched(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    return value.unsqueeze(0) if value.ndim == 1 else value


def encode_rope_image_info(value: list) -> list[list[list[int]]]:
    """Convert upstream ``[(slice, (height, width))]`` to JSON-compatible spans."""
    result = []
    for batch_item in value:
        spans = []
        for token_slice, shape in batch_item:
            spans.append([token_slice.start, token_slice.stop, int(shape[0]), int(shape[1])])
        result.append(spans)
    return result


def decode_rope_image_info(value: list[list[list[int]]]) -> list[list[tuple[slice, tuple[int, int]]]]:
    return [[(slice(start, stop), (height, width)) for start, stop, height, width in item] for item in value]


def pack_condition(model, caption: str, height: int, width: int) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Use upstream preprocessing once and retain only static, training-safe values."""
    output = model.preprocess_inputs(
        prompt=caption,
        mode="gen_image",
        image_size=(height, width),
        cfg_factor=1,
        bot_task="auto",
    )
    tokenizer_output = output["output"]
    tensors: dict[str, torch.Tensor] = {}
    values = {
        "input_ids": tokenizer_output.tokens,
        "image_mask": tokenizer_output.gen_image_mask,
        "timesteps_index": tokenizer_output.gen_timestep_scatter_index,
        "guidance_index": tokenizer_output.guidance_scatter_index,
        "timesteps_r_index": tokenizer_output.gen_timestep_r_scatter_index,
        "gen_timestep_scatter_index": tokenizer_output.gen_timestep_scatter_index,
    }
    for name, value in values.items():
        value = _batched(value)
        if value is not None:
            tensors[name] = value.cpu()
    rope = model.build_batch_rope_image_info(tokenizer_output, output["sections"])
    metadata = {"rope_image_info": encode_rope_image_info(rope), "height": height, "width": width}
    return tensors, metadata
