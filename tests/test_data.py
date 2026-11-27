"""Dataset split and leakage checks."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from activation_benchmark.config import ExperimentConfig
from activation_benchmark.data import load_moons


def test_split_shapes_and_scaler_statistics() -> None:
    config = ExperimentConfig()
    data = load_moons(config)
    assert data.X_train.shape == (420, 2)
    assert data.X_test.shape == (180, 2)
    assert data.y_train.shape == (420,)
    assert data.y_test.shape == (180,)
    assert data.X_train.dtype == np.float32

    X, y = make_moons(n_samples=config.n_samples, noise=config.noise, random_state=config.data_seed)
    X_train, _, _, _ = train_test_split(
        X, y, test_size=config.test_size, random_state=config.data_seed, stratify=y
    )
    assert np.allclose(data.scaler.mean_, X_train.mean(axis=0), atol=1e-6)
    assert np.allclose(data.scaler.scale_, X_train.std(axis=0), atol=1e-6)
    assert abs(float(data.X_train.mean())) < 1e-6
