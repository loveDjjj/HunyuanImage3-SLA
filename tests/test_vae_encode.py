import torch

from sampling.hunyuan_sampler import encode_z0


class _Distribution:
    def sample(self, generator):
        return torch.ones(1, 2, 1, 3, 4)


class _VAE:
    class config:
        shift_factor = 0.5
        scaling_factor = 2.0

    ffactor_temporal = 4

    def encode(self, image):
        return type("Result", (), {"latent_dist": _Distribution()})()


class _Model:
    vae = _VAE()


def test_encode_z0_applies_upstream_scaling_and_squeezes_time():
    value = encode_z0(_Model(), torch.zeros(1, 3, 8, 8), torch.device("cpu"), torch.float32, 7)
    assert value.shape == (2, 3, 4)
    assert torch.equal(value, torch.ones(2, 3, 4))
