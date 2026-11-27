"""Metric and activation-health checks."""

from __future__ import annotations

import torch
import torch.nn as nn

from activation_benchmark.metrics import activation_health, binary_accuracy


def test_binary_accuracy_uses_the_zero_logit_threshold() -> None:
    logits = torch.tensor([[2.0], [-1.0], [0.5], [-0.1]])
    targets = torch.tensor([[1.0], [0.0], [1.0], [1.0]])
    assert binary_accuracy(logits, targets) == 0.75


def test_dead_relu_units_are_counted() -> None:
    model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.fill_(-1.0)
    health = activation_health(model, torch.zeros(16, 2))
    assert health.dead_fraction == 1.0
    assert health.saturated_fraction == 0.0


def test_saturated_sigmoid_units_are_counted() -> None:
    model = nn.Sequential(nn.Linear(2, 4), nn.Sigmoid(), nn.Linear(4, 1))
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.fill_(6.0)
    health = activation_health(model, torch.zeros(16, 2))
    assert health.dead_fraction == 0.0
    assert health.saturated_fraction == 1.0
