"""Architecture and initialization invariants."""

from __future__ import annotations

import pytest
import torch

from activation_benchmark.models import build_mlp, count_parameters, make_activation


def test_parameter_count_matches_the_published_architecture() -> None:
    model = build_mlp("ReLU", seed=0)
    assert count_parameters(model) == 2241


def test_activation_factory_accepts_aliases() -> None:
    assert isinstance(make_activation("sigmoid"), torch.nn.Sigmoid)
    assert isinstance(make_activation("Tanh"), torch.nn.Tanh)
    assert isinstance(make_activation("relu"), torch.nn.ReLU)
    assert isinstance(make_activation("leaky_relu"), torch.nn.LeakyReLU)
    with pytest.raises(ValueError, match="Unknown activation"):
        make_activation("gelu")


def test_initial_weights_are_identical_across_activations() -> None:
    torch.manual_seed(1234)
    sigmoid = build_mlp("Sigmoid", seed=7)
    relu = build_mlp("ReLU", seed=7)
    leaky = build_mlp("LeakyReLU", seed=7)

    for left, right, third in zip(
        sigmoid.parameters(), relu.parameters(), leaky.parameters(), strict=True
    ):
        assert torch.equal(left, right)
        assert torch.equal(left, third)


def test_forward_pass_returns_one_logit_per_example() -> None:
    model = build_mlp("Tanh", seed=0)
    output = model(torch.randn(13, 2))
    assert output.shape == (13, 1)
    assert torch.isfinite(output).all()
