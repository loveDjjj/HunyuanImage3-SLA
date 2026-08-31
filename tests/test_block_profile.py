import json

import torch

from common.block_profile import (
    BlockProfileAccumulator,
    BlockProfileConfig,
    block_profile_context,
    current_block_profile,
)
from tools.plot_block_profile import plot_block_profile
from tools.profile_sla_blocks import summarize_profile


def test_uniform_pooled_scores_have_expected_mass_and_required_ratio():
    config = BlockProfileConfig(
        num_layers=1,
        num_steps=1,
        blkq=2,
        blkk=2,
        candidate_ratios=(0.25, 0.5),
        mass_thresholds=(0.5, 0.9),
    )
    accumulator = BlockProfileAccumulator(config, "cpu")
    query = torch.zeros(1, 1, 8, 2)
    key = torch.zeros_like(query)

    accumulator.collect(
        layer=0,
        steps=(0,),
        query=query,
        key=key,
        spans_by_batch=[[[0, 8]]],
    )

    assert accumulator.query_count[0, 0].item() == 4
    assert accumulator.key_block_sum[0, 0].item() == 16
    torch.testing.assert_close(accumulator.recall_sum[0, 0], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(
        accumulator.required_ratio_sum[0, 0], torch.tensor([2.0, 4.0])
    )


def test_profile_context_restores_previous_state():
    accumulator = BlockProfileAccumulator(
        BlockProfileConfig(num_layers=1, num_steps=1), "cpu"
    )
    assert current_block_profile() is None
    with block_profile_context(accumulator, (0,)):
        assert current_block_profile() == (accumulator, (0,))
    assert current_block_profile() is None


def test_profile_summary_recommends_smallest_candidate_meeting_tail_criterion():
    config = BlockProfileConfig(
        num_layers=1,
        num_steps=1,
        blkq=2,
        blkk=2,
        candidate_ratios=(0.25, 0.5),
        mass_thresholds=(0.5,),
    )
    accumulator = BlockProfileAccumulator(config, "cpu")
    accumulator.collect(
        layer=0,
        steps=(0,),
        query=torch.zeros(1, 1, 8, 2),
        key=torch.zeros(1, 1, 8, 2),
        spans_by_batch=[[[0, 8]]],
    )

    report = summarize_profile(accumulator.tensors, config)

    assert report["global"]["mean_recall"] == [0.25, 0.5]
    assert report["recommendation"]["topk"] == 0.5
    assert report["recommendation"]["criterion_satisfied"] is True


def test_block_profile_plot_writes_png(tmp_path):
    report = {
        "global": {
            "candidate_ratios": [0.125, 0.25],
            "mean_recall": [0.9, 0.98],
            "p10_recall": [0.8, 0.95],
            "p05_recall": [0.7, 0.92],
            "mass_thresholds": [0.9, 0.95, 0.99],
            "required_ratio_mean": [0.1, 0.2, 0.4],
            "required_ratio_p90": [0.2, 0.25, 0.5],
            "required_ratio_p95": [0.25, 0.3, 0.6],
        },
        "by_layer": [
            {"layer": 0, "mean_recall": [0.9, 0.98]},
            {"layer": 1, "mean_recall": [0.85, 0.96]},
        ],
        "by_step": [
            {"step": 0, "mean_recall": [0.9, 0.98]},
            {"step": 1, "mean_recall": [0.85, 0.96]},
        ],
        "recommendation": {"topk": 0.25},
    }
    source = tmp_path / "profile.json"
    output = tmp_path / "profile.png"
    source.write_text(json.dumps(report), encoding="utf-8")

    plot_block_profile(source, output)

    assert output.stat().st_size > 0
