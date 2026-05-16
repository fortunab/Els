from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch
from multih.utils.config import load_config
from multih.utils.seed import set_seed
from multih.data import SyntheticMedicalDataset, make_loaders
from multih.nas.search import random_search


def serialize(row):
    out = dict(row)
    out["candidate"] = str(out["candidate"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu"
    data_cfg = cfg["data"]
    ds = SyntheticMedicalDataset(n=data_cfg["num_samples"], num_classes=data_cfg["num_classes"], image_size=data_cfg["image_size"], seed=cfg.get("seed", 42))
    train_loader, val_loader, _ = make_loaders(ds, batch_size=cfg["training"]["batch_size"], seed=cfg.get("seed", 42))
    best, results = random_search(train_loader, val_loader, num_classes=data_cfg["num_classes"], image_size=data_cfg["image_size"], trials=cfg["nas"]["trials"], epochs=cfg["training"]["epochs"], lr=cfg["training"]["lr"], device=device)
    out_dir = Path(cfg.get("output_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best.state_dict(), out_dir / "nas_best_model.pt")
    (out_dir / "nas_results.json").write_text(json.dumps([serialize(r) for r in results], indent=2), encoding="utf-8")
    print(json.dumps(serialize(sorted(results, key=lambda x: x["score"])[0]), indent=2))


if __name__ == "__main__":
    main()
