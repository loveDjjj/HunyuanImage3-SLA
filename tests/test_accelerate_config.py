from types import SimpleNamespace

from common.accelerate_config import create_accelerator


def test_current_accelerate_uses_dataloader_configuration():
    calls = []

    class DataLoaderConfiguration:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    module = SimpleNamespace(
        DataLoaderConfiguration=DataLoaderConfiguration,
        Accelerator=lambda **kwargs: calls.append(kwargs) or kwargs,
    )

    create_accelerator(module, 2)

    assert calls[0]["gradient_accumulation_steps"] == 2
    assert calls[0]["dataloader_config"].kwargs == {"even_batches": False}


def test_legacy_accelerate_uses_direct_even_batches():
    calls = []
    module = SimpleNamespace(Accelerator=lambda **kwargs: calls.append(kwargs) or kwargs)

    create_accelerator(module, 1)

    assert calls[0] == {"gradient_accumulation_steps": 1, "even_batches": False}
