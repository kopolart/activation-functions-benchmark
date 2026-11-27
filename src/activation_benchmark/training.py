"""The training loop and experiment runner."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from .config import ACTIVATIONS, ExperimentConfig
from .data import MoonsData, load_moons
from .metrics import activation_health, binary_accuracy
from .models import build_mlp


@dataclass(frozen=True)
class RunResult:
    """One full training run, including complete per-epoch histories."""

    activation: str
    seed: int
    test_accuracy: float
    train_accuracy: float
    final_train_loss: float
    final_test_loss: float
    convergence_epoch: int | None
    train_time_s: float
    mean_epoch_time_ms: float
    first_layer_grad_norm_last10: float
    dead_fraction: float
    saturated_fraction: float
    train_loss: tuple[float, ...]
    test_loss: tuple[float, ...]
    grad_norm: tuple[float, ...]

    def as_dict(self) -> dict[str, object]:
        """JSON-friendly representation (tuples become lists)."""
        payload = asdict(self)
        for key in ("train_loss", "test_loss", "grad_norm"):
            payload[key] = list(payload[key])
        return payload


def train_once(
    activation: str,
    seed: int,
    data: MoonsData,
    config: ExperimentConfig | None = None,
) -> RunResult:
    """Train one MLP and return its metrics and full learning curves."""
    config = config or ExperimentConfig()
    X_train = torch.from_numpy(data.X_train)
    X_test = torch.from_numpy(data.X_test)
    y_train = torch.from_numpy(data.y_train).view(-1, 1)
    y_test = torch.from_numpy(data.y_test).view(-1, 1)

    model = build_mlp(activation, seed, config.widths, config.negative_slope)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(seed + 1000)

    train_losses: list[float] = []
    test_losses: list[float] = []
    grad_norms: list[float] = []
    started = time.perf_counter()

    for _ in range(config.epochs):
        model.train()
        permutation = torch.randperm(len(X_train), generator=generator)
        batch_losses: list[float] = []
        batch_grads: list[float] = []

        for indices in permutation.split(config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(X_train[indices]), y_train[indices])
            loss.backward()
            grad = model[0].weight.grad
            batch_grads.append(float(grad.detach().norm()))
            optimizer.step()
            batch_losses.append(float(loss.detach()) * len(indices))

        train_losses.append(sum(batch_losses) / len(X_train))
        grad_norms.append(sum(batch_grads) / len(batch_grads))

        model.eval()
        with torch.no_grad():
            test_losses.append(float(loss_fn(model(X_test), y_test)))

    elapsed = time.perf_counter() - started

    with torch.no_grad():
        train_accuracy = binary_accuracy(model(X_train), y_train)
        test_accuracy = binary_accuracy(model(X_test), y_test)

    health = activation_health(model, X_train)
    convergence_epoch = next(
        (
            epoch
            for epoch, value in enumerate(train_losses, start=1)
            if value <= config.convergence_threshold
        ),
        None,
    )

    return RunResult(
        activation=activation,
        seed=seed,
        test_accuracy=test_accuracy,
        train_accuracy=train_accuracy,
        final_train_loss=train_losses[-1],
        final_test_loss=test_losses[-1],
        convergence_epoch=convergence_epoch,
        train_time_s=elapsed,
        mean_epoch_time_ms=1000.0 * elapsed / config.epochs,
        first_layer_grad_norm_last10=sum(grad_norms[-10:]) / len(grad_norms[-10:]),
        dead_fraction=health.dead_fraction,
        saturated_fraction=health.saturated_fraction,
        train_loss=tuple(train_losses),
        test_loss=tuple(test_losses),
        grad_norm=tuple(grad_norms),
    )


def run_experiment(
    config: ExperimentConfig | None = None,
    activations: Iterable[str] = ACTIVATIONS,
    data: MoonsData | None = None,
    progress: bool = True,
) -> list[RunResult]:
    """Run the full activation x seed sweep and return results in execution order."""
    config = config or ExperimentConfig()
    dataset = data or load_moons(config)
    results: list[RunResult] = []

    for activation in activations:
        for seed in range(config.seeds):
            result = train_once(activation, seed, dataset, config)
            results.append(result)
            if progress:
                convergence = result.convergence_epoch
                convergence_text = "never" if convergence is None else str(convergence)
                print(
                    f"{activation:<10} seed={seed}  "
                    f"test_acc={result.test_accuracy:.4f}  "
                    f"train_loss={result.final_train_loss:.4f}  "
                    f"convergence={convergence_text}",
                    flush=True,
                )
    return results
