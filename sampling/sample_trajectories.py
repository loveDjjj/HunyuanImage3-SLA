#!/usr/bin/env python3
"""Collect official HunyuanImage3-Instruct-Distil Dense 8-step trajectories."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "upstream" / "HunyuanImage-3.0"),
]

from common.accelerate_config import configure_deepspeed_micro_batch, create_accelerator
from common.cache_schema import append_jsonl
from common.hunyuan import load_hunyuan, redirect_legacy_cuda_runtime
from common.trajectory_schema import load_trajectory, write_trajectory_atomic
from sampling.trajectory_sampler import (
    DenseTrajectoryCapture,
    replay_dense_predictions,
    replay_scheduler,
    validate_meanflow_schedule,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "trajectory_sampling.yaml"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--prompt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-id", default="000001")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_rows(args) -> list[dict]:
    if args.prompt:
        return [{"id": args.sample_id, "prompt": args.prompt, "seed": args.seed}]
    if args.manifest is None:
        raise ValueError("Pass --prompt or --manifest.")
    rows = []
    with args.manifest.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(
                {
                    "id": str(row.get("id", row.get("sample_id", index))),
                    "prompt": str(row["prompt"]),
                    "seed": int(row.get("seed", args.seed)),
                }
            )
    return rows[: args.limit or None]


def repository_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def safe_sample_id(value: str) -> str:
    result = "".join(character if character.isalnum() or character in "-_" else "_" for character in value)
    if not result:
        raise ValueError(f"Invalid sample id: {value!r}")
    return result


def rebuild_manifest(output_dir: Path) -> None:
    manifest = output_dir / "manifest.jsonl"
    temporary = manifest.with_suffix(".jsonl.incomplete")
    if temporary.exists():
        temporary.unlink()
    rows = []
    for sample_dir in sorted((output_dir / "samples").glob("sample_*")):
        if not (sample_dir / "READY.json").is_file():
            continue
        metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "sample_id": metadata["sample_id"],
                "prompt": metadata["prompt"],
                "seed": metadata["seed"],
                "path": str(sample_dir.relative_to(output_dir)),
            }
        )
    append_jsonl(temporary, rows)
    if not temporary.exists():
        temporary.write_text("", encoding="utf-8")
    temporary.replace(manifest)


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if int(cfg["num_inference_steps"]) != 8:
        raise ValueError("Official Distil trajectory collection requires num_inference_steps=8.")
    rows = read_rows(args)
    import accelerate

    accelerator = create_accelerator(accelerate, 1)
    configure_deepspeed_micro_batch(accelerator, 1)
    device = accelerator.device
    if device.type != "npu":
        raise RuntimeError(f"Trajectory collection requires Ascend NPU, got {device}.")
    model = load_hunyuan(cfg["model_path"], None, cfg["dtype"])
    model.load_tokenizer(cfg["model_path"])
    model.eval()
    model = accelerator.prepare(model)
    official_model = accelerator.unwrap_model(model)
    official_model.generation_config.diff_infer_steps = 8
    official_model.generation_config.diff_guidance_scale = float(cfg["guidance_scale"])

    output_dir = Path(cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    samples_dir = output_dir / "samples"
    if accelerator.is_main_process:
        samples_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    for row in rows:
        sample_id = safe_sample_id(str(row["id"]))
        sample_dir = samples_dir / f"sample_{sample_id}"
        ready = sample_dir / "READY.json"
        skip = bool(args.resume and ready.is_file())
        skip_tensor = torch.tensor(int(skip), device=device)
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(skip_tensor, src=0)
        if skip_tensor.item():
            if accelerator.is_main_process:
                print(f"skip trajectory sample_id={sample_id}")
            continue

        capture = DenseTrajectoryCapture(official_model, enabled=True)
        with capture, redirect_legacy_cuda_runtime():
            cot_texts, images = official_model.generate_image(
                prompt=row["prompt"],
                seed=int(row["seed"]),
                image_size=cfg["image_size"],
                use_system_prompt=cfg["use_system_prompt"],
                bot_task=cfg["bot_task"],
                diff_infer_steps=8,
                diff_guidance_scale=float(cfg["guidance_scale"]),
                max_new_tokens=int(cfg["max_new_tokens"]),
                use_taylor_cache=False,
                verbose=1 if accelerator.is_main_process else 0,
            )
        cot_text = cot_texts[0] if isinstance(cot_texts, list) else str(cot_texts)
        ar_tokens = torch.tensor(official_model._tokenizer.encode(cot_text), dtype=torch.long)
        metadata, tensors = capture.build_artifact(
            sample_id=sample_id,
            prompt=row["prompt"],
            seed=int(row["seed"]),
            cot_text=cot_text,
            ar_generated_token_ids=ar_tokens,
            guidance_scale=float(cfg["guidance_scale"]),
            bot_task=cfg["bot_task"],
            use_system_prompt=cfg["use_system_prompt"],
            model_path=cfg["model_path"],
            repository_commit=repository_commit(),
        )
        validate_meanflow_schedule(tensors, official_model.pipeline.scheduler)
        metadata["scheduler_replay_max_abs"] = replay_scheduler(
            tensors, official_model.pipeline.scheduler
        )
        dense_errors = replay_dense_predictions(
            official_model,
            metadata,
            tensors,
            device,
            atol=float(cfg["dense_replay_atol"]),
            rtol=float(cfg["dense_replay_rtol"]),
        )
        metadata["dense_replay_max_abs_per_step"] = dense_errors
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            write_trajectory_atomic(
                sample_dir,
                metadata,
                tensors,
                final_image=images[0] if cfg.get("save_final_image", False) else None,
            )
            _, loaded = load_trajectory(sample_dir)
            size = (sample_dir / "trajectory.safetensors").stat().st_size
            print(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "steps": int(loaded["teacher_predictions"].shape[0]),
                        "bytes": size,
                        "scheduler_replay_max_abs": metadata["scheduler_replay_max_abs"],
                        "dense_replay_max_abs": max(dense_errors),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        rebuild_manifest(output_dir)


if __name__ == "__main__":
    main()
