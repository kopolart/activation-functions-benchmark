"""Summary statistics and persistence helpers."""

from __future__ import annotations

import pytest

from activation_benchmark.analysis import statistical_tests, summarize
from activation_benchmark.training import RunResult


def make_result(activation: str, accuracy: float, seed: int = 0) -> RunResult:
    return RunResult(
        activation=activation,
        seed=seed,
        test_accuracy=accuracy,
        train_accuracy=accuracy,
        final_train_loss=0.2,
        final_test_loss=0.2,
        convergence_epoch=4,
        train_time_s=1.0,
        mean_epoch_time_ms=10.0,
        first_layer_grad_norm_last10=0.2,
        dead_fraction=0.0,
        saturated_fraction=0.0,
        train_loss=(0.5, 0.2),
        test_loss=(0.5, 0.2),
        grad_norm=(0.3, 0.2),
    )


def test_summary_reports_mean_and_sample_sd() -> None:
    results = [
        make_result("Sigmoid", 0.90, seed=0),
        make_result("Sigmoid", 0.94, seed=1),
        make_result("ReLU", 0.96, seed=0),
    ]
    summary = summarize(results).set_index("activation")
    assert summary.loc["Sigmoid", "test_accuracy_mean"] == pytest.approx(0.92)
    assert summary.loc["Sigmoid", "test_accuracy_sd"] == pytest.approx(0.0282843, abs=1e-6)
    assert summary.loc["ReLU", "test_accuracy_sd"] == 0.0


def test_statistical_tests_return_probabilities() -> None:
    results = [
        make_result("Sigmoid", 0.90, seed=0),
        make_result("Sigmoid", 0.91, seed=1),
        make_result("ReLU", 0.95, seed=0),
        make_result("ReLU", 0.96, seed=1),
    ]
    tests = statistical_tests(results)
    assert 0.0 <= tests["anova_p"] <= 1.0
    assert 0.0 <= tests["kruskal_p"] <= 1.0
    assert tests["n_per_group"] == 2.0