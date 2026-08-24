from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_zero3_config_declares_prebatched_micro_batch_size():
    accelerate_config = yaml.safe_load((ROOT / "configs/accelerate_zero3_16npu.yaml").read_text(encoding="utf-8"))
    training_config = yaml.safe_load((ROOT / "configs/train_sla.yaml").read_text(encoding="utf-8"))

    assert accelerate_config["distributed_type"] == "DEEPSPEED"
    assert accelerate_config["num_processes"] == 16
    assert accelerate_config["deepspeed_config"]["zero_stage"] == 3
    assert training_config["train_micro_batch_size_per_gpu"] == 1
