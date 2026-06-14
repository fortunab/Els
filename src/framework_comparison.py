import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIG = ROOT / "configs" / "framework_comparison.json"
RESULTS = ROOT / "results"

RESULTS.mkdir(exist_ok=True)

with open(CONFIG, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for framework, metrics in data.items():

    rows.append({
        "Framework": framework,
        "Architecture type": metrics["Architecture type"],
        "Federated": metrics["Federated"],
        "Explainable": metrics["Explainable"],
        "Accuracy": metrics["Accuracy"],
        "AUC": metrics["AUC"],
        "ECE": metrics["ECE"]
    })

df = pd.DataFrame(rows)

df.to_csv(
    RESULTS / "framework_comparison.csv",
    index=False
)

print("\nFramework Comparison")
print("=" * 100)
print(df.to_string(index=False))