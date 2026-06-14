import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIG = ROOT / "configs" / "nas_efficiency.json"
RESULTS = ROOT / "results"

RESULTS.mkdir(exist_ok=True)

with open(CONFIG, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for architecture, metrics in data.items():

    rows.append({
        "Architecture": architecture,
        "Accuracy": metrics["Accuracy"],
        "Parameters (M)": metrics["Parameters (M)"],
        "FLOPs (G)": metrics["FLOPs (G)"],
        "Latency (ms)": metrics["Latency (ms)"]
    })

df = pd.DataFrame(rows)

output_file = RESULTS / "nas_efficiency.csv"

df.to_csv(
    output_file,
    index=False
)

print("\nNAS Optimization and Computational Efficiency")
print("=" * 80)
print(df.to_string(index=False))

print(f"\nSaved: {output_file}")