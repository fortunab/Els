import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIG = ROOT / "configs" / "ablation_study.json"
RESULTS = ROOT / "results"

RESULTS.mkdir(exist_ok=True)

with open(CONFIG, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for configuration, metrics in data.items():

    rows.append({
        "Configuration": configuration,
        "Accuracy": metrics["Accuracy"],
        "F1-score": metrics["F1-score"],
        "AUC": metrics["AUC"],
        "ECE": metrics["ECE"]
    })

df = pd.DataFrame(rows)

df.to_csv(
    RESULTS / "ablation_study.csv",
    index=False
)

print("\nAblation Study")
print("=" * 90)
print(df.to_string(index=False))