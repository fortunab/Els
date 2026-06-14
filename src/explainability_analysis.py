import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIG = ROOT / "configs" / "explainability_analysis.json"
RESULTS = ROOT / "results"

RESULTS.mkdir(exist_ok=True)

with open(CONFIG, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for disease, metrics in data.items():
    rows.append({
        "Disease domain": disease,
        "Localization accuracy": metrics["Localization accuracy"],
        "IoU": metrics["IoU"],
        "Clinician agreement": metrics["Clinician agreement"]
    })

df = pd.DataFrame(rows)

df.to_csv(
    RESULTS / "explainability_analysis.csv",
    index=False
)

print("\nExplainability and Attention Visualization Analysis")
print("=" * 70)
print(df.to_string(index=False))
