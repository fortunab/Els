import numpy as np
from multih.metrics.classification import compute_metrics


def test_compute_metrics_binary():
    y = np.array([0, 1, 0, 1])
    p = np.array([[0.9, 0.1], [0.1, 0.9], [0.6, 0.4], [0.2, 0.8]])
    m = compute_metrics(y, p)
    assert m["accuracy"] == 1.0
    assert m["mcc"] == 1.0
