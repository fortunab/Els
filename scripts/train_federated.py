from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import random_split
from multih.utils.config import load_config
from multih.utils.seed import set_seed
from multih.data import SyntheticMedicalDataset, make_loaders
from multih.models.factory import build_model
from multih.fl.strategies import Client, run_federated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu"
    data_cfg = cfg["data"]
    fl_cfg = cfg["federated"]
    clients = []
    for i in range(fl_cfg["num_clients"]):
        ds = SyntheticMedicalDataset(n=data_cfg["num_samples_per_client"], num_classes=data_cfg["num_classes"], image_size=data_cfg["image_size"], seed=cfg.get("seed", 42) + i)
        train_loader, val_loader, _ = make_loaders(ds, batch_size=cfg["training"]["batch_size"], seed=cfg.get("seed", 42) + i)
        clients.append(Client(name=f"hospital_{i+1}", train_loader=train_loader, val_loader=val_loader, num_samples=len(ds)))
    def model_fn():
        return build_model(cfg["model"]["name"], num_classes=data_cfg["num_classes"], image_size=data_cfg["image_size"])
    model, history = run_federated(model_fn, clients, rounds=fl_cfg["rounds"], local_epochs=fl_cfg["local_epochs"], lr=cfg["training"]["lr"], device=device, strategy=fl_cfg.get("strategy", "fedavg"), mu=fl_cfg.get("mu", 0.01))
    out_dir = Path(cfg.get("output_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "federated_model.pt")
    (out_dir / "federated_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(json.dumps(history[-1], indent=2))


if __name__ == "__main__":
    main()
