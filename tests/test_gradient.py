import torch

from common.gradient import inspect_local_gradients


def test_regular_gradients_are_inspected_from_parameter_grad():
    parameter = torch.nn.Parameter(torch.ones(2))
    parameter.grad = torch.tensor([3.0, 4.0])

    stats = inspect_local_gradients([parameter])

    assert stats.element_count == 2
    assert stats.nonfinite_count == 0
    assert stats.squared_norm == 25.0


def test_partitioned_gradient_getter_is_used_when_parameter_grad_is_none():
    parameter = torch.nn.Parameter(torch.ones(2))

    stats = inspect_local_gradients([parameter], lambda _: torch.tensor([1.0, 2.0]))

    assert stats.element_count == 2
    assert stats.nonfinite_count == 0
    assert stats.squared_norm == 5.0


def test_nonfinite_gradient_is_reported_without_a_nan_norm():
    parameter = torch.nn.Parameter(torch.ones(2))
    parameter.grad = torch.tensor([float("nan"), 1.0])

    stats = inspect_local_gradients([parameter])

    assert stats.element_count == 2
    assert stats.nonfinite_count == 1
    assert stats.squared_norm == 0.0
