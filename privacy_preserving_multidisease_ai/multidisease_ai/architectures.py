"""
Deep Learning Neural Architectures & Benchmarks
Section 3.4 & Section 4.7-4.8:
- Baseline CNN
- Vision Transformer (ViT) with Multi-Head Self-Attention (Eq. 9)
- Hybrid CNN-ViT Network
- NAS-Optimized Global Architecture MH
- SOTA Models: ResNet-50, EfficientNet-B4, FedAvg-CNN
"""

import math
import numpy as np

def compute_self_attention(query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
    """
    Self-Attention Mechanism (Eq. 9):
    Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) ) * V
    """
    d_k = query.shape[-1]
    scores = np.matmul(query, key.T) / math.sqrt(d_k)
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return np.matmul(weights, value)

ARCHITECTURE_METRICS = {
    "Baseline CNN": {
        "params_m": 34.72,
        "flops_g": 9.84,
        "latency_ms": 42.16,
        "accuracy": 0.9015,
        "f1_score": 0.8912,
        "auc": 0.9145,
        "ece": 0.0385
    },
    "Vision transformer": {
        "params_m": 48.36,
        "flops_g": 12.91,
        "latency_ms": 57.43,
        "accuracy": 0.9178,
        "f1_score": 0.9064,
        "auc": 0.9317,
        "ece": 0.0335
    },
    "Hybrid CNN-ViT": {
        "params_m": 39.14,
        "flops_g": 10.27,
        "latency_ms": 46.82,
        "accuracy": 0.9286,
        "f1_score": 0.9175,
        "auc": 0.9448,
        "ece": 0.0272
    },
    "NAS-optimized model": {
        "params_m": 28.63,
        "flops_g": 7.92,
        "latency_ms": 34.57,
        "accuracy": 0.9342,
        "f1_score": 0.9238,
        "auc": 0.9516,
        "ece": 0.0217
    }
}

SOTA_COMPARISONS = {
    "ResNet-50": {"accuracy": 0.8927, "f1_score": 0.8816, "auc": 0.9048, "ece": 0.0412, "pr_auc": 0.8914, "froc": 0.8647, "mcc": 0.7816},
    "EfficientNet-B4": {"accuracy": 0.9073, "f1_score": 0.8951, "auc": 0.9186, "ece": 0.0368, "pr_auc": 0.9078, "froc": 0.8819, "mcc": 0.8127},
    "Vision transformer": {"accuracy": 0.9178, "f1_score": 0.9064, "auc": 0.9317, "ece": 0.0335, "pr_auc": 0.9216, "froc": 0.8964, "mcc": 0.8368},
    "FedAvg-CNN": {"accuracy": 0.9218, "f1_score": 0.9109, "auc": 0.9386, "ece": 0.0304, "pr_auc": 0.9297, "froc": 0.9043, "mcc": 0.8512},
    "Hybrid CNN-ViT": {"accuracy": 0.9286, "f1_score": 0.9175, "auc": 0.9448, "ece": 0.0272, "pr_auc": 0.9378, "froc": 0.9135, "mcc": 0.8729},
    "Architecture MH": {"accuracy": 0.9342, "f1_score": 0.9238, "auc": 0.9516, "ece": 0.0217, "pr_auc": 0.9442, "froc": 0.9126, "mcc": 0.8873}
}
