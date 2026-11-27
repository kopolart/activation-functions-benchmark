"""Aggregation, statistical tests, and JSON/CSV persistence."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, kruskal

from .config import ACTIVATIONS
from .training import RunResult

SUMMARY_COLUMNS = [
    "activation",
    "test_accuracy_mean",
    "test_accuracy_sd",
    "train_accuracy_mean",
    "train_loss_mean",
    "test_loss_mean",
    "convergence_epoch_mean",
    "first_layer_grad_norm_last10_mean",
    "dead_fraction_mean",
    "saturated_fraction_mean",
    "epoch_time_ms_mean",
]


def as_records(results: Iterable[RunResult | Mapping[str, object]]) -> list[dict[str, object]]:
    """Normalize dataclass results and JSON dictionaries to plain dictionaries."""
    records: list[dict[str, object]] = []
    for result in results:
        records.append(result.as_dict() if isinstance(result, RunResult) else dict(result))
    return records


def _mean(values: list[float]) -> float:
    return float(np.mean(values))


def _sd(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def summarize(results: Iterable[RunResult | Mapping[str, object]]) -> pd.DataFrame:
    """Aggregate each activation to mean and sample standard deviation."""
    records = as_records(results)
    rows: list[dict[str, object]] = []
    for activation in ACTIVATIONS:
        subset = [record for record in records if record["activation"] == activation]
        if not subset:
            continue
        accuracies = [float(record["test_accuracy"]) for record in subset]
        rows.append(
            {
                "activation": activation,
                "test_accuracy_mean": _mean(accuracies),
                "test_accuracy_sd": _sd(accuracies),
                "train_accuracy_mean": _mean(
                    [float(record["train_accuracy"]) for record in subset]
                ),
                "train_loss_mean": _mean(
                    [float(record["final_train_loss"]) for record in subset]
                ),
                "test_loss_mean": _mean(
                    [float(record["final_test_loss"]) for record in subset]
                ),
                "convergence_epoch_mean": _mean(
                    [
                        float(record["convergence_epoch"])
                        for record in subset
                        if record["convergence_epoch"] is not None
                    ]
                ),
                "first_layer_grad_norm_last10_mean": _mean(
                    [float(record["first_layer_grad_norm_last10"]) for record in subset]
                ),
                "dead_fraction_mean": _mean(
                    [float(record["dead_fraction"]) for record in subset]
                ),
                "saturated_fraction_mean": _mean(
                    [float(record["saturated_fraction"]) for record in subset]
                ),
                "epoch_time_ms_mean": _mean(
                    [float(record["mean_epoch_time_ms"]) for record in subset]
                ),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def statistical_tests(
    results: Iterable[RunResult | Mapping[str, object]],
) -> dict[str, float]:
    """Omnibus tests for a difference in test accuracy between activations."""
    records = as_records(results)
    groups = [
        np.array(
            [
                float(record["test_accuracy"])
                for record in records
                if record["activation"] == activation
            ],
            dtype=float,
        )
        for activation in ACTIVATIONS
    ]
    groups = [group for group in groups if group.size > 0]
    if len(groups) < 2:
        return {
            "anova_F": float("nan"),
            "anova_p": float("nan"),
            "kruskal_H": float("nan"),
            "kruskal_p": float("nan"),
            "n_per_group": float(groups[0].size if groups else 0),
        }
    anova = f_oneway(*groups)
    kruskal_result = kruskal(*groups)
    return {
        "anova_F": float(anova.statistic),
        "anova_p": float(anova.pvalue),
        "kruskal_H": float(kruskal_result.statistic),
        "kruskal_p": float(kruskal_result.pvalue),
        "n_per_group": float(groups[0].size),
    }


def save_results(results: Iterable[RunResult], path: str | Path) -> Path:
    """Write the canonical JSON artifact."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = [result.as_dict() for result in results]
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def load_results(path: str | Path) -> list[dict[str, object]]:
    """Load a results JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_summary_csv(summary: pd.DataFrame, path: str | Path) -> Path:
    """Write the aggregate table as UTF-8 CSV."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(destination, index=False, encoding="utf-8")
    return destination
