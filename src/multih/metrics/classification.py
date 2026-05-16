from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
    brier_score_loss,
)


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = predictions == y_true
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (confidences > lower) & (confidences <= upper)
        if np.any(mask):
            ece += np.abs(accuracies[mask].mean() - confidences[mask].mean()) * mask.mean()
    return float(ece)


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro") -> float:
    labels = np.unique(y_true)
    specs = []
    for label in labels:
        y_true_bin = y_true == label
        y_pred_bin = y_pred == label
        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[False, True]).ravel()
        specs.append(tn / (tn + fp + 1e-12))
    return float(np.mean(specs)) if average == "macro" else float(specs[0])


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, average: str = "macro") -> dict[str, float]:
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    y_pred = probs.argmax(axis=1)
    num_classes = probs.shape[1]
    out: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "specificity": specificity_score(y_true, y_pred, average=average),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "ece": expected_calibration_error(y_true, probs),
    }
    try:
        if num_classes == 2:
            out["roc_auc"] = float(roc_auc_score(y_true, probs[:, 1]))
            out["pr_auc"] = float(average_precision_score(y_true, probs[:, 1]))
            out["brier"] = float(brier_score_loss(y_true, probs[:, 1]))
        else:
            out["roc_auc"] = float(roc_auc_score(y_true, probs, multi_class="ovr", average=average))
            y_onehot = np.eye(num_classes)[y_true]
            out["pr_auc"] = float(average_precision_score(y_onehot, probs, average=average))
            out["brier"] = float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))
    except ValueError:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
        out["brier"] = float("nan")
    return out


def entropy(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, 1e-12, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)
