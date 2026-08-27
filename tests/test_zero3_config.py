from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_zero3_resident_config_declares_prebatched_micro_batch_size():
    accelerate_config = yaml.safe_load((ROOT / "configs/accelerate_zero3_16npu.yaml").read_text(encoding="utf-8"))
    training_config = yaml.safe_load((ROOT / "configs/train_sla.yaml").read_text(encoding="utf-8"))

    assert accelerate_config["distributed_type"] == "DEEPSPEED"
    assert accelerate_config["num_processes"] == 16
    assert accelerate_config["deepspeed_config"]["zero_stage"] == 3
    assert accelerate_config["deepspeed_config"]["offload_param_device"] == "none"
    assert accelerate_config["deepspeed_config"]["offload_optimizer_device"] == "none"
    assert training_config["train_micro_batch_size_per_gpu"] == 1
    assert training_config["activation_checkpointing"] is True


def test_zero3_offload_fallback_keeps_cpu_devices():
    config = yaml.safe_load(
        (ROOT / "configs/accelerate_zero3_16npu_offload.yaml").read_text(encoding="utf-8")
    )
    assert config["deepspeed_config"]["offload_param_device"] == "cpu"
    assert config["deepspeed_config"]["offload_optimizer_device"] == "cpu"


def test_14npu_lora_configs_have_compatible_global_and_validation_batches():
    accelerate_config = yaml.safe_load(
        (ROOT / "configs/accelerate_zero3_14npu.yaml").read_text(encoding="utf-8")
    )
    training_config = yaml.safe_load(
        (ROOT / "configs/train_sla_attention_moe_lora_14npu.yaml").read_text(
            encoding="utf-8"
        )
    )
    world_size = accelerate_config["num_processes"]
    micro_batch = training_config["train_micro_batch_size_per_gpu"]
    validation_points = training_config["validation"]["num_prompts"] * 8

    assert world_size == 14
    assert accelerate_config["deepspeed_config"]["zero_stage"] == 3
    assert accelerate_config["deepspeed_config"]["offload_param_device"] == "none"
    assert accelerate_config["deepspeed_config"]["offload_optimizer_device"] == "none"
    assert world_size * micro_batch == 56
    assert validation_points == 56
    assert validation_points % world_size == 0
    assert training_config["expected_trainable_parameters"] == 155_717_632
