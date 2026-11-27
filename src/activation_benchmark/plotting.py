"""Publication-quality figures for the benchmark."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .analysis import as_records
from .config import ACTIVATION_COLORS, ACTIVATION_LABELS, ACTIVATIONS

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Inter", "DejaVu Sans"],
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "figure.dpi": 120,
        "savefig.dpi": 240,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)


def plot_activation_shapes(path: str | Path) -> Path:
    """Plot the four activation functions and their derivatives."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    x = np.linspace(-6.0, 6.0, 1600)
    alpha = 0.01

    sigmoid = 1.0 / (1.0 + np.exp(-x))
    tanh = np.tanh(x)
    relu = np.maximum(0.0, x)
    leaky = np.where(x > 0.0, x, alpha * x)

    curves = {
        "Sigmoid": (sigmoid, sigmoid * (1.0 - sigmoid)),
        "Tanh": (tanh, 1.0 - tanh**2),
        "ReLU": (relu, (x > 0.0).astype(float)),
        "LeakyReLU": (leaky, np.where(x > 0.0, 1.0, alpha)),
    }

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.2, 4.1), constrained_layout=True)
    for name in ACTIVATIONS:
        values, derivative = curves[name]
        color = ACTIVATION_COLORS[name]
        left.plot(x, values, color=color, lw=2.1, label=ACTIVATION_LABELS[name])
        right.plot(x, derivative, color=color, lw=2.1, label=ACTIVATION_LABELS[name])

    for ax in (left, right):
        ax.axhline(0.0, color="#8A94A6", lw=0.8, zorder=0)
        ax.axvline(0.0, color="#8A94A6", lw=0.8, zorder=0)
        ax.set_xlim(-6.0, 6.0)
        ax.set_xlabel("Pre-activation  z")

    left.set_ylim(-1.18, 1.18)
    left.set_ylabel("Activation  phi(z)")
    left.set_title("Activation functions")
    left.legend(loc="upper left", ncol=2, fontsize=9, frameon=True, framealpha=0.92,
                facecolor="white", edgecolor="none")

    right.set_ylim(-0.12, 1.12)
    right.set_ylabel("Derivative  phi'(z)")
    right.set_title("Derivatives: where gradients survive")

    fig.savefig(destination)
    plt.close(fig)
    return destination


def _moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Centered-ish trailing moving average used to de-noise gradient curves."""
    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="valid")


def _matrix(records: list[dict[str, object]], activation: str, key: str) -> np.ndarray:
    return np.array(
        [record[key] for record in records if record["activation"] == activation],
        dtype=float,
    )


def plot_training_dynamics(
    results: Iterable[Mapping[str, object]],
    path: str | Path,
) -> Path:
    """Mean training loss and first-layer gradient norm over training."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = as_records(results)
    epochs = np.arange(1, len(records[0]["train_loss"]) + 1)

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)
    for name in ACTIVATIONS:
        color = ACTIVATION_COLORS[name]
        losses = _matrix(records, name, "train_loss")
        mean_loss = losses.mean(axis=0)
        sd_loss = losses.std(axis=0, ddof=1) if losses.shape[0] > 1 else np.zeros_like(mean_loss)
        left.plot(epochs, mean_loss, color=color, lw=1.8, label=ACTIVATION_LABELS[name])
        left.fill_between(epochs, mean_loss - sd_loss, mean_loss + sd_loss, color=color, alpha=0.13)

        gradients = _matrix(records, name, "grad_norm")
        window = min(5, len(epochs))
        smoothed = _moving_average(gradients.mean(axis=0), window)
        right.plot(
            epochs[window - 1:],
            smoothed,
            color=color,
            lw=1.8,
            label=ACTIVATION_LABELS[name],
        )

    left.axhline(0.30, color="#5F6B7A", ls=(0, (4, 3)), lw=1.0)
    left.text(94, 0.315, "convergence threshold = 0.30", fontsize=8.5, color="#5F6B7A")
    left.set(
        xlabel="Epoch",
        ylabel="Training BCE",
        title="Training loss (mean ± 1 s.d., n = 5)",
    )
    left.legend(loc="upper right", fontsize=9)

    right.set_yscale("log")
    right.set(
        xlabel="Epoch",
        ylabel="First-layer gradient norm",
        title="Gradient flow (5-epoch moving average, log scale)",
    )

    fig.savefig(destination)
    plt.close(fig)
    return destination


def plot_accuracy_and_convergence(
    results: Iterable[Mapping[str, object]],
    path: str | Path,
) -> Path:
    """Final test accuracy and epochs to reach the convergence threshold."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = as_records(results)

    means: list[float] = []
    sds: list[float] = []
    convergence: list[list[float]] = []
    for name in ACTIVATIONS:
        accuracies = _matrix(records, name, "test_accuracy") * 100.0
        means.append(float(accuracies.mean()))
        sds.append(float(accuracies.std(ddof=1)) if accuracies.size > 1 else 0.0)
        epochs = [
            float(record["convergence_epoch"])
            for record in records
            if record["activation"] == name and record["convergence_epoch"] is not None
        ]
        convergence.append(epochs)

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.2, 4.1), constrained_layout=True)
    y = np.arange(len(ACTIVATIONS))
    left.barh(
        y,
        means,
        xerr=sds,
        height=0.62,
        color=[ACTIVATION_COLORS[name] for name in ACTIVATIONS],
        alpha=0.9,
        capsize=3,
        error_kw={"elinewidth": 1.0, "ecolor": "#374151"},
    )
    left.set_yticks(y, [ACTIVATION_LABELS[name] for name in ACTIVATIONS])
    left.invert_yaxis()
    left.set_xlim(88.0, 100.8)
    left.set(
        xlabel="Test accuracy (%)",
        title="Final test accuracy (mean ± 1 s.d.)",
    )
    left.grid(axis="x", alpha=0.2)
    left.grid(axis="y", visible=False)
    for index, (mean, sd) in enumerate(zip(means, sds, strict=True)):
        left.text(min(mean + sd + 0.35, 100.45), index, f"{mean:.2f}%",
                  va="center", fontsize=9, color="#111827")

    for index, name in enumerate(ACTIVATIONS):
        values = np.array(convergence[index], dtype=float)
        if values.size == 0:
            right.bar(index, 0.0, width=0.62, color=ACTIVATION_COLORS[name], alpha=0.25)
            right.text(index, 1.2, "not reached", ha="center", va="bottom",
                       fontsize=8, rotation=90, color="#6B7280")
            continue
        right.bar(index, values.mean(), width=0.62, color=ACTIVATION_COLORS[name], alpha=0.9)
        jitter = np.linspace(-0.18, 0.18, len(values)) if len(values) > 1 else np.zeros(1)
        right.scatter(
            index + jitter,
            values,
            s=22,
            color="#111827",
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        right.text(index, values.mean() + 1.4, f"{values.mean():.1f}",
                   ha="center", fontsize=9, color="#111827")

    right.set_xticks(range(len(ACTIVATIONS)), [ACTIVATION_LABELS[n] for n in ACTIVATIONS])
    reached = [value for group in convergence for value in group]
    upper = max(reached) + 8 if reached else 20
    right.set_ylim(0, upper)
    right.set(
        ylabel="Epoch",
        title="Epochs to BCE ≤ 0.30 (dots = seeds)",
    )
    right.grid(axis="y", alpha=0.2)
    right.grid(axis="x", visible=False)

    fig.savefig(destination)
    plt.close(fig)
    return destination


def plot_health_profile(
    results: Iterable[Mapping[str, object]],
    path: str | Path,
) -> Path:
    """Dead and saturated hidden-unit fractions by activation."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = as_records(results)
    dead = [_matrix(records, name, "dead_fraction").mean() * 100.0 for name in ACTIVATIONS]
    saturated = [
        _matrix(records, name, "saturated_fraction").mean() * 100.0 for name in ACTIVATIONS
    ]

    fig, ax = plt.subplots(figsize=(8.6, 3.9), constrained_layout=True)
    x = np.arange(len(ACTIVATIONS))
    width = 0.36
    ax.bar(x - width / 2, dead, width, label="Dead units", color="#C44E52", alpha=0.9)
    ax.bar(x + width / 2, saturated, width, label="Saturated units", color="#6A4C93", alpha=0.75)

    for index, value in enumerate(dead):
        ax.text(index - width / 2, value + 0.35, f"{value:.1f}%", ha="center", fontsize=8.5)
    for index, value in enumerate(saturated):
        ax.text(index + width / 2, value + 0.35, f"{value:.1f}%", ha="center", fontsize=8.5)

    ax.set_xticks(x, [ACTIVATION_LABELS[name] for name in ACTIVATIONS])
    ax.set_ylim(0, max(dead + saturated) + 3.5)
    ax.set(
        ylabel="% of hidden units",
        title="Hidden-unit health at the end of training",
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    ax.grid(axis="x", visible=False)

    fig.savefig(destination)
    plt.close(fig)
    return destination


def make_all_figures(
    results: Iterable[Mapping[str, object]],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Generate every figure and return the written paths."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    records = as_records(results)
    return {
        "activation_shapes": plot_activation_shapes(destination / "activation_shapes.png"),
        "training_dynamics": plot_training_dynamics(records, destination / "loss_and_gradient.png"),
        "accuracy_convergence": plot_accuracy_and_convergence(
            records, destination / "accuracy_and_convergence.png"
        ),
        "health_profile": plot_health_profile(records, destination / "health_profile.png"),
    }