from pathlib import Path
import sys
import random

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from medai_suite.semantic_framework import ADResult, semantic_record, rank_architectures
from medai_suite.utils import save_json


records = []
for name in ["SmallCNN_A", "SmallCNN_B", "SmallCNN_C"]:
    result = ADResult(
        model_name=name,
        score=random.random(),
        params=random.randint(10_000, 100_000),
        latency_ms=random.uniform(1.0, 10.0),
    )
    records.append(semantic_record(result))

ranked = rank_architectures(records)
save_json("outputs/results/semantic_demo.json", ranked)
print(ranked)
