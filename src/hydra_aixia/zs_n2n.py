from __future__ import annotations

import numpy as np


def normalize_features(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (x - x.mean(axis=0)) / (x.std(axis=0) + eps)


def nearest_neighbor_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    train_x = normalize_features(train_x)
    test_x = normalize_features(test_x)
    distances = ((test_x[:, None, :] - train_x[None, :, :]) ** 2).sum(axis=2)
    return train_y[np.argmin(distances, axis=1)]
