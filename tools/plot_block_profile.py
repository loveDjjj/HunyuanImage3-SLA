#!/usr/bin/env python3
"""Plot SLA pooled-block calibration JSON without loading the model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def plot_block_profile(report_path: Path, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(report_path.read_text(encoding="utf-8"))
    global_stats = report["global"]
    ratios = global_stats["candidate_ratios"]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    axes[0, 0].plot(ratios, global_stats["mean_recall"], marker="o", label="mean")
    axes[0, 0].plot(ratios, global_stats["p10_recall"], marker=".", label="P10")
    axes[0, 0].plot(ratios, global_stats["p05_recall"], marker=".", label="P05")
    axes[0, 0].axhline(0.95, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_title("Router Proxy Mass Recall")
    axes[0, 0].set_xlabel("configured top-k ratio")
    axes[0, 0].set_ylabel("cumulative pooled mass")
    axes[0, 0].legend()

    thresholds = [str(value) for value in global_stats["mass_thresholds"]]
    x = range(len(thresholds))
    axes[0, 1].plot(x, global_stats["required_ratio_mean"], marker="o", label="mean")
    axes[0, 1].plot(x, global_stats["required_ratio_p90"], marker=".", label="P90")
    axes[0, 1].plot(x, global_stats["required_ratio_p95"], marker=".", label="P95")
    axes[0, 1].set_xticks(list(x), thresholds)
    axes[0, 1].set_title("Blocks Required for Target Proxy Mass")
    axes[0, 1].set_xlabel("target cumulative mass")
    axes[0, 1].set_ylabel("required block ratio")
    axes[0, 1].legend()

    selected_index = global_stats["candidate_ratios"].index(report["recommendation"]["topk"])
    layer_values = [row["mean_recall"][selected_index] for row in report["by_layer"]]
    axes[1, 0].plot(range(len(layer_values)), layer_values, marker=".")
    axes[1, 0].set_title(f"Mean Recall by Layer at top-k={report['recommendation']['topk']}")
    axes[1, 0].set_xlabel("layer")
    axes[1, 0].set_ylabel("mean pooled mass recall")

    step_values = [row["mean_recall"][selected_index] for row in report["by_step"]]
    axes[1, 1].plot(range(len(step_values)), step_values, marker="o")
    axes[1, 1].set_title(f"Mean Recall by MeanFlow Step at top-k={report['recommendation']['topk']}")
    axes[1, 1].set_xlabel("MeanFlow step")
    axes[1, 1].set_ylabel("mean pooled mass recall")

    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.set_ylim(bottom=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.png")
    figure.savefig(temporary, dpi=150)
    plt.close(figure)
    os.replace(temporary, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.report.with_suffix(".png")
    plot_block_profile(args.report, output)


if __name__ == "__main__":
    main()
