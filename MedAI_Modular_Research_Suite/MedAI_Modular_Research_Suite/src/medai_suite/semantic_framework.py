from dataclasses import dataclass, asdict
from typing import Dict, List
import time
import numpy as np


@dataclass
class ADResult:
    model_name: str
    score: float
    params: int
    latency_ms: float


def semantic_record(result: ADResult) -> Dict:
    """Create a monitorable semantic record.

    This is a lightweight Python alternative to the RDF/Gradio prototype in the notebook.
    """
    payload = asdict(result)
    payload["timestamp"] = time.time()
    payload["type"] = "ArchitectureDecision"
    return payload


def rank_architectures(records: List[Dict]):
    return sorted(records, key=lambda x: x["score"], reverse=True)


def dummy_histology_batch(n=16, image_size=96, num_classes=4):
    x = np.random.rand(n, 3, image_size, image_size).astype("float32")
    y = np.random.randint(0, num_classes, size=(n,))
    return x, y
