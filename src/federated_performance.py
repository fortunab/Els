import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIG = ROOT / "configs" / "federated_performance.json"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

with open(CONFIG, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []

for method, metrics in data["summary"].items():
    rows.append({
        "Method": method,
        "Accuracy": metrics["Accuracy"],
        "F1-score": metrics["F1-score"],
        "AUC": metrics["AUC"],
        "Rounds": metrics["Rounds"]
    })

df = pd.DataFrame(rows)

df.to_csv(
    RESULTS / "federated_performance.csv",
    index=False
)

print("\nFederated Learning Performance")
print("=" * 70)
print(df.to_string(index=False))

conv = data["convergence"]

rounds = conv["rounds"]

plt.figure(figsize=(7, 5))

plt.plot(rounds, conv["FedAvg"], marker="o", label="FedAvg")
plt.plot(rounds, conv["FedProx"], marker="s", label="FedProx")
plt.plot(rounds, conv["Centralized"], marker="D", label="Centralized")

plt.xlabel("Communication rounds")
plt.ylabel("Validation accuracy")
plt.ylim(0.75, 0.96)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURES / "federated_convergence.png",
    dpi=300
)

plt.show()