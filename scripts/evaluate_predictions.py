from __future__ import annotations

import argparse
import json
import pandas as pd
from multih.metrics.classification import compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="CSV with y_true and probability columns prob_0, prob_1, ...")
    args = parser.parse_args()
    df = pd.read_csv(args.predictions)
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if "y_true" not in df.columns or not prob_cols:
        raise ValueError("CSV must include y_true and prob_* columns.")
    metrics = compute_metrics(df["y_true"].to_numpy(), df[prob_cols].to_numpy())
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
