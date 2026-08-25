import torch

from train.noise_sampler import flow_match_batch, flow_match_input, sample_seed


def test_flow_matching_input_is_deterministic():
    z0 = torch.ones(2, 3, 4)
    seed = sample_seed(7, "sample-a", 2, 3)
    first, first_t, first_r = flow_match_input(z0, seed, 0.01, 0.99, 1000)
    second, second_t, second_r = flow_match_input(z0, seed, 0.01, 0.99, 1000)
    assert torch.equal(first, second)
    assert first_t == second_t
    assert first_r == second_r
    assert 10 <= first_t.item() <= 990
    assert 10 <= first_r.item() <= first_t.item()


def test_flow_match_batch_uses_independent_reproducible_sample_seeds():
    z0 = torch.zeros(2, 4, 8, 8)
    kwargs = {
        "global_seed": 17,
        "epoch": 2,
        "view": 3,
        "sigma_min": 0.01,
        "sigma_max": 0.99,
        "train_timesteps": 1000,
    }
    first = flow_match_batch(z0, ["a", "b"], **kwargs)
    second = flow_match_batch(z0, ["a", "b"], **kwargs)

    assert all(torch.equal(left, right) for left, right in zip(first, second))
    assert not torch.equal(first[0][0], first[0][1])
    assert first[1].shape == first[2].shape == (2,)
