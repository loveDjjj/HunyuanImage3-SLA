#!/usr/bin/env python3
"""Render training metrics JSONL as an atomic PNG and auto-refreshing HTML page."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def read_records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def plot_metrics(metrics_path: Path, png_path: Path, html_path: Path | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = read_records(metrics_path)
    if not records:
        return
    png_path.parent.mkdir(parents=True, exist_ok=True)
    steps = [record["step"] for record in records]
    figure, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)

    axes[0, 0].plot(steps, [record["loss"] for record in records], alpha=0.35, label="train")
    axes[0, 0].plot(steps, [record["loss_ema"] for record in records], label="train EMA")
    validation = [(record["step"], record["validation_mse"]) for record in records if "validation_mse" in record]
    if validation:
        axes[0, 0].plot(*zip(*validation), marker="o", label="validation")
    axes[0, 0].set_title("Recovery MSE")
    axes[0, 0].legend()

    gradient_names = sorted(
        {key.removesuffix("_grad_norm") for record in records for key in record if key.endswith("_grad_norm")}
    )
    axes[0, 1].plot(steps, [record["gradient_norm"] for record in records], label="total")
    for name in gradient_names:
        axes[0, 1].plot(steps, [record.get(f"{name}_grad_norm", float("nan")) for record in records], label=name)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Gradient Norms")
    axes[0, 1].legend(fontsize=8)

    validation_records = [record for record in records if "validation_relative_mse" in record]
    axes[1, 0].plot(
        [record["step"] for record in validation_records],
        [record["validation_relative_mse"] for record in validation_records],
        color="tab:blue",
        marker="o",
        label="relative MSE",
    )
    validation_cosine_axis = axes[1, 0].twinx()
    validation_cosine_axis.plot(
        [record["step"] for record in validation_records],
        [
            record.get(
                "validation_cosine_distance",
                1.0 - record.get("validation_cosine", 1.0),
            )
            for record in validation_records
        ],
        color="tab:red",
        marker=".",
        label="cosine distance",
    )
    axes[1, 0].set_title("Teacher-Forced Validation")
    axes[1, 0].set_ylabel("relative MSE", color="tab:blue")
    validation_cosine_axis.set_ylabel("cosine distance", color="tab:red")
    validation_lines = axes[1, 0].get_lines() + validation_cosine_axis.get_lines()
    if validation_lines:
        axes[1, 0].legend(
            validation_lines,
            [line.get_label() for line in validation_lines],
            fontsize=8,
        )

    validation_step_records = [
        record for record in records if "validation_relative_mse_by_step" in record
    ]
    for timestep in range(8):
        axes[1, 1].plot(
            [record["step"] for record in validation_step_records],
            [
                record["validation_relative_mse_by_step"][timestep]
                for record in validation_step_records
            ],
            marker=".",
            label=f"trajectory {timestep}",
        )
    axes[1, 1].set_title("Teacher-Forced Relative MSE by MeanFlow Step")
    if validation_step_records:
        axes[1, 1].legend(fontsize=7, ncol=2)

    rollout_records = [
        record for record in records if "rollout_final_latent_relative_mse" in record
    ]
    rollout_steps = [record["step"] for record in rollout_records]
    axes[2, 0].plot(
        rollout_steps,
        [record["rollout_final_latent_relative_mse"] for record in rollout_records],
        marker="o",
        label="final latent relative MSE",
    )
    axes[2, 0].plot(
        rollout_steps,
        [
            record["rollout_final_laplacian_relative_mse"]
            for record in rollout_records
        ],
        marker=".",
        label="final latent Laplacian relative MSE",
    )
    rollout_cosine_axis = axes[2, 0].twinx()
    rollout_cosine_axis.plot(
        rollout_steps,
        [record["rollout_final_latent_cosine_distance"] for record in rollout_records],
        color="tab:red",
        marker="x",
        label="final latent cosine distance",
    )
    axes[2, 0].set_title("Eight-Step Free Rollout Final Latent")
    axes[2, 0].set_ylabel("relative error")
    rollout_cosine_axis.set_ylabel("cosine distance", color="tab:red")
    if rollout_records:
        rollout_lines = axes[2, 0].get_lines() + rollout_cosine_axis.get_lines()
        axes[2, 0].legend(
            rollout_lines,
            [line.get_label() for line in rollout_lines],
            fontsize=8,
        )

    for timestep in range(8):
        axes[2, 1].plot(
            rollout_steps,
            [
                record["rollout_latent_relative_mse_by_step"][timestep]
                for record in rollout_records
            ],
            marker=".",
            label=f"rollout {timestep + 1}",
        )
    axes[2, 1].set_title("Free Rollout Relative MSE Growth")
    if rollout_records:
        axes[2, 1].legend(fontsize=7, ncol=2)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.set_xlabel("optimizer step")

    temporary = png_path.with_name(f".{png_path.name}.tmp.png")
    figure.savefig(temporary, dpi=140)
    plt.close(figure)
    os.replace(temporary, png_path)
    if html_path is not None:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_html = html_path.with_suffix(".html.incomplete")
        temporary_html.write_text(
            "<!doctype html><meta charset='utf-8'><meta http-equiv='refresh' content='15'>"
            f"<title>SLA training metrics</title><img src='{png_path.name}' style='max-width:100%'>",
            encoding="utf-8",
        )
        os.replace(temporary_html, html_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--png", type=Path)
    parser.add_argument("--html", type=Path)
    args = parser.parse_args()
    png = args.png or args.metrics.with_name("training_metrics.png")
    plot_metrics(args.metrics, png, args.html or png.with_suffix(".html"))


if __name__ == "__main__":
    main()
