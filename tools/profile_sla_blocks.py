#!/usr/bin/env python3
"""Profile guidance-1 Dense Hunyuan pooled Q/K mass for SLA top-k calibration."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "upstream" / "DiffSynth-Studio"))

from common.accelerate_config import configure_deepspeed_micro_batch, create_accelerator  # noqa: E402
from common.block_profile import (  # noqa: E402
    BlockProfileAccumulator,
    BlockProfileConfig,
    block_profile_context,
)
from tools.plot_block_profile import plot_block_profile  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "block_profile_guidance1.yaml")
    )
    return parser.parse_args()


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


def _histogram_quantile(histogram: torch.Tensor, quantile: float) -> float:
    total = float(histogram.sum())
    if total <= 0:
        return math.nan
    target = total * quantile
    index = int(torch.searchsorted(histogram.cumsum(0), torch.tensor(target)).item())
    return index / (histogram.numel() - 1)


def _summarize_slice(
    state: dict[str, torch.Tensor],
    config: BlockProfileConfig,
    layer_slice,
    step_slice,
) -> dict[str, Any]:
    count = state["query_count"][layer_slice, step_slice].sum()
    count_value = max(float(count), 1.0)
    recall_sum = state["recall_sum"][layer_slice, step_slice].reshape(
        -1, len(config.candidate_ratios)
    ).sum(0)
    recall_hist = state["recall_hist"][layer_slice, step_slice].reshape(
        -1, len(config.candidate_ratios), config.histogram_bins
    ).sum(0)
    required_sum = state["required_ratio_sum"][layer_slice, step_slice].reshape(
        -1, len(config.mass_thresholds)
    ).sum(0)
    required_hist = state["required_ratio_hist"][layer_slice, step_slice].reshape(
        -1, len(config.mass_thresholds), config.histogram_bins
    ).sum(0)
    return {
        "query_blocks": int(float(count)),
        "average_key_blocks": float(
            state["key_block_sum"][layer_slice, step_slice].sum() / count_value
        ),
        "candidate_ratios": list(config.candidate_ratios),
        "mean_recall": [float(value / count_value) for value in recall_sum],
        "p10_recall": [
            _histogram_quantile(histogram, 0.10) for histogram in recall_hist
        ],
        "p05_recall": [
            _histogram_quantile(histogram, 0.05) for histogram in recall_hist
        ],
        "mass_thresholds": list(config.mass_thresholds),
        "required_ratio_mean": [float(value / count_value) for value in required_sum],
        "required_ratio_p90": [
            _histogram_quantile(histogram, 0.90) for histogram in required_hist
        ],
        "required_ratio_p95": [
            _histogram_quantile(histogram, 0.95) for histogram in required_hist
        ],
    }


def summarize_profile(
    state: dict[str, torch.Tensor], config: BlockProfileConfig
) -> dict[str, Any]:
    global_stats = _summarize_slice(state, config, slice(None), slice(None))
    by_layer = [
        {"layer": layer, **_summarize_slice(state, config, layer, slice(None))}
        for layer in range(config.num_layers)
    ]
    by_step = [
        {"step": step, **_summarize_slice(state, config, slice(None), step)}
        for step in range(config.num_steps)
    ]
    target_index = min(
        range(len(config.mass_thresholds)),
        key=lambda index: abs(config.mass_thresholds[index] - 0.95),
    )
    required_p90 = global_stats["required_ratio_p90"][target_index]
    eligible = [
        ratio
        for ratio, p10 in zip(config.candidate_ratios, global_stats["p10_recall"])
        if p10 >= config.mass_thresholds[target_index] and ratio >= required_p90
    ]
    recommended = eligible[0] if eligible else config.candidate_ratios[-1]
    return {
        "global": global_stats,
        "by_layer": by_layer,
        "by_step": by_step,
        "recommendation": {
            "topk": recommended,
            "criterion": "smallest candidate with P10 recall >= 0.95 and ratio >= P90 required ratio",
            "proxy_mass_threshold": config.mass_thresholds[target_index],
            "required_ratio_p90": required_p90,
            "criterion_satisfied": bool(eligible),
        },
    }


def _validate_guidance_one(root: Path, rows: list[dict[str, Any]]) -> None:
    from safetensors import safe_open

    for row in rows:
        sample_dir = root / row["path"]
        metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
        guidance_scale = float(metadata.get("guidance_scale", math.nan))
        with safe_open(
            str(sample_dir / "trajectory.safetensors"), framework="pt", device="cpu"
        ) as handle:
            guidance = handle.get_tensor("guidance").float()
        if not math.isclose(guidance_scale, 1.0, rel_tol=0, abs_tol=1e-6):
            raise RuntimeError(
                f"Profile sample {metadata.get('sample_id')} has guidance_scale={guidance_scale}, expected 1.0."
            )
        if not torch.allclose(guidance, torch.full_like(guidance, 1000.0)):
            raise RuntimeError(
                f"Profile sample {metadata.get('sample_id')} guidance tensor is not 1000."
            )


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    import accelerate
    from common.hunyuan import load_hunyuan
    from hunyuan_adapter import HunyuanSLARecoveryModule
    from trajectory_dataset import HunyuanTrajectoryDataset, collate_trajectory_records

    accelerate.utils.set_seed(int(cfg["seed"]), device_specific=False)
    accelerator = create_accelerator(accelerate, 1)
    configure_deepspeed_micro_batch(accelerator, 1)
    if accelerator.num_processes != 16 or accelerator.device.type != "npu":
        raise RuntimeError(
            f"Block profiling requires 16 Ascend ranks, got world={accelerator.num_processes}, "
            f"device={accelerator.device}."
        )

    trajectory_root = Path(cfg["trajectory_dir"])
    if not trajectory_root.is_absolute():
        trajectory_root = ROOT / trajectory_root
    dataset = HunyuanTrajectoryDataset(
        str(trajectory_root), dtype=cfg["dtype"], max_prompts=int(cfg["num_prompts"])
    )
    _validate_guidance_one(trajectory_root, dataset.rows)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collate_trajectory_records,
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
    )

    model = load_hunyuan(
        cfg["model_path"],
        None,
        cfg["dtype"],
        skip_load_modules=cfg.get("skip_load_modules", ("vae", "vit")),
    )
    profile_model = HunyuanSLARecoveryModule(
        model,
        topk=float(cfg["sla"]["topk"]),
        blkq=int(cfg["sla"]["blkq"]),
        blkk=int(cfg["sla"]["blkk"]),
        use_bf16=bool(cfg["sla"].get("use_bf16", True)),
        training_backend="auto",
        trainable_components=("proj_l",),
        activation_checkpointing=False,
        log_phases=False,
    )
    optimizer = torch.optim.AdamW(
        profile_model.trainable_parameter_groups()["proj_l"], lr=0.0
    )
    profile_model, optimizer, dataloader = accelerator.prepare(
        profile_model, optimizer, dataloader
    )
    del optimizer

    profile_config = BlockProfileConfig(
        num_layers=int(cfg["num_layers"]),
        num_steps=8,
        blkq=int(cfg["sla"]["blkq"]),
        blkk=int(cfg["sla"]["blkk"]),
        candidate_ratios=tuple(float(value) for value in cfg["candidate_ratios"]),
        mass_thresholds=tuple(float(value) for value in cfg["mass_thresholds"]),
    )
    accumulator = BlockProfileAccumulator(profile_config, accelerator.device)
    profile_model.eval()
    progress = tqdm(
        dataloader,
        desc="pooled block profile",
        unit="trajectory-point",
        disable=not accelerator.is_main_process,
        dynamic_ncols=True,
    )
    with torch.no_grad():
        for batch in progress:
            batch = move(batch, accelerator.device)
            step = int(batch.pop("trajectory_step").item())
            batch.pop("teacher_diffusion_prediction")
            batch.pop("sample_id", None)
            spans = batch.pop("full_attention_spans")
            with block_profile_context(accumulator, (step,)):
                profile_model(batch, full_attention_spans=spans, profile_dense=True)

    reduced = {
        name: accelerator.reduce(tensor, reduction="sum").cpu()
        for name, tensor in accumulator.tensors.items()
    }
    if accelerator.is_main_process:
        output_dir = Path(cfg["output_dir"])
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "profile_version": 1,
            "model_path": cfg["model_path"],
            "trajectory_dir": str(trajectory_root),
            "guidance_scale": 1.0,
            "num_prompts": int(cfg["num_prompts"]),
            "blkq": profile_config.blkq,
            "blkk": profile_config.blkk,
            "score_definition": "softmax(mean_pool(Q) @ mean_pool(K-mean(K)).T / sqrt(head_dim))",
            "warning": "Pooled router proxy mass, not exact token-level Dense attention mass.",
            **summarize_profile(reduced, profile_config),
        }
        report_path = output_dir / "block_profile.json"
        temporary = report_path.with_suffix(".json.incomplete")
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, report_path)
        plot_block_profile(report_path, output_dir / "block_profile.png")
        print(json.dumps(report["recommendation"], indent=2))
        print(f"report={report_path}")
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
