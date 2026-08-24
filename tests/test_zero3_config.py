from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_zero3_config_declares_prebatched_micro_batch_size():
    config = yaml.safe_load((ROOT / "configs/accelerate_zero3_16npu.yaml").read_text(encoding="utf-8"))

    assert config["distributed_type"] == "DEEPSPEED"
    assert config["num_processes"] == 16
    assert config["deepspeed_config"]["zero_stage"] == 3
    assert config["deepspeed_config"]["train_micro_batch_size_per_gpu"] == 1
