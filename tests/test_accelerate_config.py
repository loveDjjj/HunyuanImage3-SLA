from types import SimpleNamespace

from common.accelerate_config import configure_deepspeed_micro_batch, create_accelerator


def test_accelerator_receives_gradient_accumulation_steps():
    calls = []
    module = SimpleNamespace(Accelerator=lambda **kwargs: calls.append(kwargs) or kwargs)

    create_accelerator(module, 2)

    assert calls[0] == {"gradient_accumulation_steps": 2}


def test_deepspeed_micro_batch_is_written_to_active_plugin():
    plugin = SimpleNamespace(deepspeed_config={})
    accelerator = SimpleNamespace(state=SimpleNamespace(deepspeed_plugin=plugin))

    assert configure_deepspeed_micro_batch(accelerator, 1)
    assert plugin.deepspeed_config["train_micro_batch_size_per_gpu"] == 1


def test_micro_batch_configuration_is_a_noop_without_deepspeed():
    accelerator = SimpleNamespace(state=SimpleNamespace(deepspeed_plugin=None))

    assert not configure_deepspeed_micro_batch(accelerator, 1)
