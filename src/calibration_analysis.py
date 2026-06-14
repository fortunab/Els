import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIG = ROOT / "configs" / "calibration_analysis.json"
RESULTS = ROOT / "results"

RESULTS.mkdir(exist_ok=True)

with open(CONFIG, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for disease, metrics in data.items():

    rows.append({
        "Disease domain": disease,
        "ECE": metrics["ECE"],
        "Brier score": metrics["Brier score"],
        "Entropy": metrics["Entropy"]
    })

df = pd.DataFrame(rows)

output_file = RESULTS / "calibration_analysis.csv"

df.to_csv(
    output_file,
    index=False
)

print("\nCalibration and Uncertainty Analysis")
print("=" * 80)
print(df.to_string(index=False))

print(f"\nSaved: {output_file}")