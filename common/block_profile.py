"""Low-overhead pooled Q/K block-mass profiling for SLA calibration."""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator

import torch


_ACTIVE_PROFILE: ContextVar[tuple["BlockProfileAccumulator", tuple[int, ...]] | None] = (
    ContextVar("sla_block_profile", default=None)
)


def _mean_pool(value: torch.Tensor, block_size: int) -> torch.Tensor:
    length = value.shape[-2]
    blocks = (length + block_size - 1) // block_size
    padding = blocks * block_size - length
    if padding:
        value = torch.nn.functional.pad(value, (0, 0, 0, padding))
    pooled = value.reshape(*value.shape[:-2], blocks, block_size, value.shape[-1]).sum(-2)
    lengths = torch.full(
        (blocks,), block_size, dtype=pooled.dtype, device=pooled.device
    )
    lengths[-1] = length - (blocks - 1) * block_size
    return pooled / lengths.reshape(*([1] * (pooled.ndim - 2)), blocks, 1)


@dataclass(frozen=True)
class BlockProfileConfig:
    num_layers: int = 32
    num_steps: int = 8
    blkq: int = 128
    blkk: int = 128
    candidate_ratios: tuple[float, ...] = (0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5)
    mass_thresholds: tuple[float, ...] = (0.9, 0.95, 0.99)
    histogram_bins: int = 101

    def __post_init__(self) -> None:
        if self.num_layers < 1 or self.num_steps < 1:
            raise ValueError("Block profile layer and step counts must be positive.")
        if self.blkq < 1 or self.blkk < 1:
            raise ValueError("Block profile block sizes must be positive.")
        if any(not 0 < ratio <= 1 for ratio in self.candidate_ratios):
            raise ValueError("Candidate top-k ratios must be in (0, 1].")
        if any(not 0 < threshold < 1 for threshold in self.mass_thresholds):
            raise ValueError("Mass thresholds must be in (0, 1).")
        if self.histogram_bins < 2:
            raise ValueError("Block profile histogram requires at least two bins.")


class BlockProfileAccumulator:
    """Accumulate fixed-size histograms without retaining Q/K or score matrices."""

    def __init__(self, config: BlockProfileConfig, device: torch.device | str):
        self.config = config
        shape = (config.num_layers, config.num_steps)
        self.query_count = torch.zeros(shape, dtype=torch.float32, device=device)
        self.key_block_sum = torch.zeros(shape, dtype=torch.float32, device=device)
        self.recall_sum = torch.zeros(
            (*shape, len(config.candidate_ratios)), dtype=torch.float32, device=device
        )
        self.recall_hist = torch.zeros(
            (*shape, len(config.candidate_ratios), config.histogram_bins),
            dtype=torch.float32,
            device=device,
        )
        self.required_ratio_sum = torch.zeros(
            (*shape, len(config.mass_thresholds)), dtype=torch.float32, device=device
        )
        self.required_ratio_hist = torch.zeros(
            (*shape, len(config.mass_thresholds), config.histogram_bins),
            dtype=torch.float32,
            device=device,
        )

    @property
    def tensors(self) -> dict[str, torch.Tensor]:
        return {
            "query_count": self.query_count,
            "key_block_sum": self.key_block_sum,
            "recall_sum": self.recall_sum,
            "recall_hist": self.recall_hist,
            "required_ratio_sum": self.required_ratio_sum,
            "required_ratio_hist": self.required_ratio_hist,
        }

    @torch.no_grad()
    def collect(
        self,
        *,
        layer: int,
        steps: tuple[int, ...],
        query: torch.Tensor,
        key: torch.Tensor,
        spans_by_batch: list[list[list[int] | tuple[int, int]]],
    ) -> None:
        if not 0 <= layer < self.config.num_layers:
            raise ValueError(f"Profile layer is out of range: {layer}.")
        if query.ndim != 4 or key.ndim != 4 or query.shape[:2] != key.shape[:2]:
            raise ValueError(f"Expected matching [B,H,L,D] Q/K, got {query.shape}, {key.shape}.")
        if len(steps) != query.shape[0] or len(spans_by_batch) != query.shape[0]:
            raise ValueError("Profile steps/spans must have one entry per batch row.")

        for batch_index, (step, spans) in enumerate(zip(steps, spans_by_batch)):
            if not 0 <= step < self.config.num_steps:
                raise ValueError(f"Profile trajectory step is out of range: {step}.")
            for raw_start, raw_end in spans:
                start, end = int(raw_start), int(raw_end)
                if not 0 <= start < end <= key.shape[-2]:
                    raise ValueError(
                        f"Invalid profile image span {(start, end)} for length={key.shape[-2]}."
                    )
                row_q = query[batch_index : batch_index + 1, :, :end].float()
                row_k = key[batch_index : batch_index + 1, :, :end].float()
                row_k = row_k - row_k.mean(dim=-2, keepdim=True)
                pooled_q = _mean_pool(row_q, self.config.blkq)
                pooled_k = _mean_pool(row_k, self.config.blkk)
                score = torch.matmul(pooled_q, pooled_k.transpose(-1, -2))
                probability = torch.softmax(score / (query.shape[-1] ** 0.5), dim=-1)

                query_start = start // self.config.blkq
                query_end = (end + self.config.blkq - 1) // self.config.blkq
                rows = probability[:, :, query_start:query_end].reshape(-1, probability.shape[-1])
                if rows.numel() == 0:
                    continue
                sorted_probability = rows.sort(dim=-1, descending=True).values
                cumulative = sorted_probability.cumsum(dim=-1)
                count = float(rows.shape[0])
                key_blocks = rows.shape[-1]
                self.query_count[layer, step] += count
                self.key_block_sum[layer, step] += count * key_blocks

                for candidate_index, ratio in enumerate(self.config.candidate_ratios):
                    selected = min(key_blocks, int(ratio * key_blocks))
                    recall = (
                        cumulative[:, selected - 1]
                        if selected > 0
                        else cumulative.new_zeros(rows.shape[0])
                    )
                    self.recall_sum[layer, step, candidate_index] += recall.sum()
                    self.recall_hist[layer, step, candidate_index] += self._histogram(recall)

                for threshold_index, threshold in enumerate(self.config.mass_thresholds):
                    required = (
                        (cumulative >= threshold).to(torch.int32).argmax(dim=-1).float() + 1.0
                    ) / key_blocks
                    self.required_ratio_sum[layer, step, threshold_index] += required.sum()
                    self.required_ratio_hist[layer, step, threshold_index] += self._histogram(required)

    def _histogram(self, value: torch.Tensor) -> torch.Tensor:
        indexes = torch.clamp(
            (value * (self.config.histogram_bins - 1)).round().long(),
            0,
            self.config.histogram_bins - 1,
        )
        return torch.bincount(indexes, minlength=self.config.histogram_bins).float()


@contextlib.contextmanager
def block_profile_context(
    accumulator: BlockProfileAccumulator, steps: tuple[int, ...]
) -> Iterator[None]:
    token = _ACTIVE_PROFILE.set((accumulator, tuple(int(step) for step in steps)))
    try:
        yield
    finally:
        _ACTIVE_PROFILE.reset(token)


def current_block_profile() -> tuple[BlockProfileAccumulator, tuple[int, ...]] | None:
    return _ACTIVE_PROFILE.get()
