import os
import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIG = ROOT / "configs" / "classification_performance.json"

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

with open(CONFIG, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for disease, metrics in data.items():

    rows.append({
        "Disease domain": disease,
        "Accuracy": metrics["Accuracy"],
        "Precision": metrics["Precision"],
        "Sensitivity": metrics["Sensitivity"],
        "F1-score": metrics["F1-score"],
        "AUC": metrics["AUC"]
    })

df = pd.DataFrame(rows)

csv_file = RESULTS / "classification_performance.csv"

df.to_csv(csv_file, index=False)

print("\nClassification Performance Across Disease Domains")
print("=" * 70)
print(df.to_string(index=False))

print(f"\nSaved: {csv_file}")