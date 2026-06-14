import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIG = ROOT / "configs" / "sota_comparison.json"
RESULTS = ROOT / "results"

RESULTS.mkdir(exist_ok=True)

with open(CONFIG, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for method, metrics in data.items():

    rows.append({
        "Method": method,
        "Accuracy": metrics["Accuracy"],
        "F1-score": metrics["F1-score"],
        "AUC": metrics["AUC"],
        "ECE": metrics["ECE"]
    })

df = pd.DataFrame(rows)

output_file = RESULTS / "sota_comparison.csv"

df.to_csv(
    output_file,
    index=False
)

print("\nComparison with State-of-the-Art Methods")
print("=" * 80)
print(df.to_string(index=False))

print(f"\nSaved: {output_file}")