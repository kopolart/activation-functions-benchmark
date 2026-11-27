"""Activation factory and the shared MLP architecture."""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ExperimentConfig


def make_activation(name: str, negative_slope: float = 0.01) -> nn.Module:
    """Return a fresh activation module from a canonical or loose name."""
    normalized = name.replace("_", "").replace("-", "").replace(" ", "").lower()
    if normalized == "sigmoid":
        return nn.Sigmoid()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "relu":
        return nn.ReLU()
    if normalized in {"leakyrelu", "leaky"}:
        return nn.LeakyReLU(negative_slope=negative_slope)
    raise ValueError(
        f"Unknown activation {name!r}. Choose from Sigmoid, Tanh, ReLU, or LeakyReLU."
    )


def build_mlp(
    activation: str,
    seed: int,
    widths: tuple[int, ...] | None = None,
    negative_slope: float = 0.01,
) -> nn.Sequential:
    """Build the benchmark MLP with Xavier-uniform weights and zero biases.

    Seeding happens before layer construction, and activations consume no random
    numbers, so every activation function receives identical initial weights for
    a given seed.
    """
    if widths is None:
        widths = ExperimentConfig().widths
    if len(widths) < 2:
        raise ValueError("widths must contain at least an input and an output layer")

    torch.manual_seed(seed)
    layers: list[nn.Module] = []
    for index, (n_in, n_out) in enumerate(zip(widths[:-1], widths[1:], strict=True)):
        layers.append(nn.Linear(n_in, n_out))
        if index < len(widths) - 2:
            layers.append(make_activation(activation, negative_slope))

    model = nn.Sequential(*layers)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    return model


def count_parameters(model: nn.Module) -> int:
    """Number of trainable scalar parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
