"""A controlled PyTorch benchmark of activation functions in a deep MLP."""

from .analysis import as_records, load_results, save_results, statistical_tests, summarize
from .config import ACTIVATION_COLORS, ACTIVATION_LABELS, ACTIVATIONS, ExperimentConfig
from .data import MoonsData, load_moons
from .metrics import ActivationHealth, activation_health, binary_accuracy
from .models import build_mlp, count_parameters, make_activation
from .training import RunResult, run_experiment, train_once

__all__ = [
    "ACTIVATIONS",
    "ACTIVATION_COLORS",
    "ACTIVATION_LABELS",
    "ActivationHealth",
    "ExperimentConfig",
    "MoonsData",
    "RunResult",
    "activation_health",
    "as_records",
    "binary_accuracy",
    "build_mlp",
    "count_parameters",
    "load_moons",
    "load_results",
    "make_activation",
    "run_experiment",
    "save_results",
    "statistical_tests",
    "summarize",
    "train_once",
]

__version__ = "1.0.0"
