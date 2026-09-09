"""
Neural Architecture Search (NAS) Optimization Engine
Section 2.5, 3.1, Algorithm 1 & Section 4.7 (Table 6):
- Multi-objective architectural selection balancing classification loss, latency, and memory footprint
  a_{t+1} = argmin_{a in A} ( L_cls(a) + lambda_1 * L_lat(a) + lambda_2 * L_mem(a) )
"""

from typing import Dict, Any

class NASObjectiveEvaluator:
    def __init__(self, lambda_lat: float = 0.05, lambda_mem: float = 0.02):
        self.lambda_lat = lambda_lat
        self.lambda_mem = lambda_mem

    def evaluate_architecture(self, cls_loss: float, latency_ms: float, memory_mb: float) -> float:
        """
        Multi-objective loss function evaluation.
        """
        return cls_loss + self.lambda_lat * (latency_ms / 50.0) + self.lambda_mem * (memory_mb / 100.0)

def get_nas_table_data() -> Dict[str, Dict[str, float]]:
    """
    Returns exact data for Table 6 from the paper:
    NAS optimization and computational efficiency analysis.
    """
    return {
        "Baseline CNN": {
            "accuracy": 0.9015,
            "params_m": 34.72,
            "flops_g": 9.84,
            "latency_ms": 42.16
        },
        "Vision transformer": {
            "accuracy": 0.9178,
            "params_m": 48.36,
            "flops_g": 12.91,
            "latency_ms": 57.43
        },
        "Hybrid CNN-ViT": {
            "accuracy": 0.9286,
            "params_m": 39.14,
            "flops_g": 10.27,
            "latency_ms": 46.82
        },
        "NAS-optimized model": {
            "accuracy": 0.9342,
            "params_m": 28.63,
            "flops_g": 7.92,
            "latency_ms": 34.57
        }
    }
