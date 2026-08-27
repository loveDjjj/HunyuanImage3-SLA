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
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

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

    axes[1, 0].plot(steps, [record["step_seconds"] for record in records], label="seconds/step")
    throughput_axis = axes[1, 0].twinx()
    throughput_axis.plot(steps, [record["samples_per_second"] for record in records], color="tab:green", label="samples/s")
    axes[1, 0].set_title("Throughput")

    validation_records = [record for record in records if "validation_mse_by_step" in record]
    for timestep in range(8):
        axes[1, 1].plot(
            [record["step"] for record in validation_records],
            [record["validation_mse_by_step"][timestep] for record in validation_records],
            marker=".",
            label=f"trajectory {timestep}",
        )
    axes[1, 1].set_title("Validation MSE by MeanFlow Step")
    if validation_records:
        axes[1, 1].legend(fontsize=7, ncol=2)
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
