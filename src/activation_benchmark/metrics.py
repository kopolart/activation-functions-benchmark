"""Evaluation metrics and activation-health diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ActivationHealth:
    """Average fraction of dead and saturated units across hidden layers."""

    dead_fraction: float
    saturated_fraction: float


def binary_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Accuracy for BCE-with-logits outputs at the standard 0.5 probability threshold."""
    predictions = logits >= 0.0
    labels = targets >= 0.5
    return float((predictions == labels).float().mean())


def activation_health(model: nn.Sequential, X: torch.Tensor) -> ActivationHealth:
    """Measure dead ReLU units and saturated Sigmoid/Tanh units on X.

    A ReLU unit is counted as dead when its output is non-positive for every
    sample. A Sigmoid/Tanh unit is counted as saturated when its output falls in
    the flat tail (absolute value above 0.99). The returned fractions are averaged
    over all hidden layers.
    """
    dead: list[float] = []
    saturated: list[float] = []
    h = X
    model.eval()
    with torch.no_grad():
        for layer in model:
            h = layer(h)
            if isinstance(layer, nn.ReLU):
                dead.append(float((h <= 1e-8).all(dim=0).float().mean()))
                saturated.append(0.0)
            elif isinstance(layer, nn.LeakyReLU):
                dead.append(0.0)
                saturated.append(0.0)
            elif isinstance(layer, nn.Sigmoid):
                dead.append(0.0)
                saturated.append(float(((h < 0.01) | (h > 0.99)).float().mean()))
            elif isinstance(layer, nn.Tanh):
                dead.append(0.0)
                saturated.append(float((h.abs() > 0.99).float().mean()))

    if not dead:
        return ActivationHealth(dead_fraction=0.0, saturated_fraction=0.0)
    return ActivationHealth(
        dead_fraction=float(sum(dead) / len(dead)),
        saturated_fraction=float(sum(saturated) / len(saturated)),
    )
