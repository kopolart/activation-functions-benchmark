"""Experiment-wide constants and the single configuration object."""

from __future__ import annotations

from dataclasses import dataclass

ACTIVATIONS: tuple[str, ...] = ("Sigmoid", "Tanh", "ReLU", "LeakyReLU")

ACTIVATION_LABELS: dict[str, str] = {
    "Sigmoid": "Sigmoid",
    "Tanh": "Tanh",
    "ReLU": "ReLU",
    "LeakyReLU": "Leaky ReLU",
}

ACTIVATION_COLORS: dict[str, str] = {
    "Sigmoid": "#6A4C93",
    "Tanh": "#2A9D8F",
    "ReLU": "#C44E52",
    "LeakyReLU": "#D9A441",
}


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Everything that is held constant across the four activation functions.

    The defaults reproduce the published results in results/results.json.
    """

    epochs: int = 150
    seeds: int = 5
    batch_size: int = 64
    learning_rate: float = 0.01
    convergence_threshold: float = 0.30
    hidden_width: int = 32
    hidden_layers: int = 3
    negative_slope: float = 0.01
    data_seed: int = 42
    n_samples: int = 600
    noise: float = 0.22
    test_size: float = 0.30

    @property
    def widths(self) -> tuple[int, ...]:
        """Layer widths, including the two input features and one output logit."""
        return (2,) + (self.hidden_width,) * self.hidden_layers + (1,)

    @property
    def n_runs(self) -> int:
        """Total number of training runs in a full sweep."""
        return len(ACTIVATIONS) * self.seeds
