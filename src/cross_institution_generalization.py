import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIG = ROOT / "configs" / "cross_institution_generalization.json"
RESULTS = ROOT / "results"

RESULTS.mkdir(exist_ok=True)

with open(CONFIG, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for institution, metrics in data.items():
    rows.append({
        "Institution": institution,
        "Accuracy": metrics["Accuracy"],
        "F1-score": metrics["F1-score"],
        "AUC": metrics["AUC"],
        "ECE": metrics["ECE"]
    })

df = pd.DataFrame(rows)

df.to_csv(
    RESULTS / "cross_institution_generalization.csv",
    index=False
)

print("\nCross-Institution Generalization")
print("=" * 70)
print(df.to_string(index=False))