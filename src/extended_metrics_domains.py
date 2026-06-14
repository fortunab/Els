import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIG = ROOT / "configs" / "extended_metrics_domains.json"
RESULTS = ROOT / "results"

RESULTS.mkdir(exist_ok=True)

with open(CONFIG, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for disease, metrics in data.items():

    rows.append({
        "Disease domain": disease,
        "PR-AUC": metrics["PR-AUC"],
        "FROC": metrics["FROC"],
        "MCC": metrics["MCC"]
    })

df = pd.DataFrame(rows)

df.to_csv(
    RESULTS / "extended_metrics_domains.csv",
    index=False
)

print("\nExtended Evaluation Metrics Across Disease Domains")
print("=" * 80)
print(df.to_string(index=False))