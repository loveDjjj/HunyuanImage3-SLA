import torch

from train.noise_sampler import flow_match_input, sample_seed


def test_flow_matching_input_is_deterministic():
    z0 = torch.ones(2, 3, 4)
    seed = sample_seed(7, "sample-a", 2, 3)
    first, first_t = flow_match_input(z0, seed, 0.01, 0.99, 1000)
    second, second_t = flow_match_input(z0, seed, 0.01, 0.99, 1000)
    assert torch.equal(first, second)
    assert first_t == second_t
    assert 10 <= first_t.item() <= 990
