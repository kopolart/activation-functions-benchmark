"""Dataset construction for the two-moons benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import ExperimentConfig


@dataclass(frozen=True)
class MoonsData:
    """A stratified train/test split with a scaler fitted only on the training set."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler


def load_moons(config: ExperimentConfig | None = None) -> MoonsData:
    """Create the exact dataset used by every run.

    The scaler is fitted on X_train only, so the test set never leaks into
    preprocessing. Both splits are returned as float32 arrays.
    """
    config = config or ExperimentConfig()
    X, y = make_moons(
        n_samples=config.n_samples,
        noise=config.noise,
        random_state=config.data_seed,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.data_seed,
        stratify=y,
    )
    scaler = StandardScaler().fit(X_train)
    return MoonsData(
        X_train=scaler.transform(X_train).astype(np.float32),
        X_test=scaler.transform(X_test).astype(np.float32),
        y_train=y_train.astype(np.float32),
        y_test=y_test.astype(np.float32),
        scaler=scaler,
    )
