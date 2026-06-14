import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIG = ROOT / "configs" / "extended_sota_metrics.json"
RESULTS = ROOT / "results"

RESULTS.mkdir(exist_ok=True)

with open(CONFIG, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for method, metrics in data.items():

    rows.append({
        "Method": method,
        "PR-AUC": metrics["PR-AUC"],
        "FROC": metrics["FROC"],
        "MCC": metrics["MCC"]
    })

df = pd.DataFrame(rows)

df.to_csv(
    RESULTS / "extended_sota_metrics.csv",
    index=False
)

print("\nExtended Comparative Analysis")
print("=" * 80)
print(df.to_string(index=False))