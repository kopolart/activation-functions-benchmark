"""A short end-to-end smoke test of the training loop."""

from __future__ import annotations

import math

from activation_benchmark.config import ExperimentConfig
from activation_benchmark.data import load_moons
from activation_benchmark.training import train_once


def test_two_epoch_run_returns_finite_metrics_and_histories() -> None:
    config = ExperimentConfig(epochs=2, seeds=1)
    data = load_moons(config)
    result = train_once("Tanh", seed=0, data=data, config=config)

    assert 0.0 <= result.test_accuracy <= 1.0
    assert 0.0 <= result.train_accuracy <= 1.0
    assert len(result.train_loss) == 2
    assert len(result.test_loss) == 2
    assert len(result.grad_norm) == 2
    assert all(math.isfinite(value) for value in result.train_loss)
    assert all(math.isfinite(value) for value in result.test_loss)
    assert all(value > 0.0 for value in result.grad_norm)
    assert result.mean_epoch_time_ms > 0.0
