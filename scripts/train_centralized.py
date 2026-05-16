from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch
from multih.utils.config import load_config
from multih.utils.seed import set_seed
from multih.data import SyntheticMedicalDataset, load_image_folder, make_loaders
from multih.models.factory import build_model
from multih.training.engine import fit, predict
from multih.metrics.classification import compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu"
    data_cfg = cfg["data"]
    if data_cfg.get("synthetic", True):
        dataset = SyntheticMedicalDataset(n=data_cfg["num_samples"], num_classes=data_cfg["num_classes"], image_size=data_cfg["image_size"], seed=cfg.get("seed", 42))
    else:
        dataset = load_image_folder(data_cfg["path"], image_size=data_cfg["image_size"], train=True)
    train_loader, val_loader, test_loader = make_loaders(dataset, batch_size=cfg["training"]["batch_size"], seed=cfg.get("seed", 42))
    model = build_model(cfg["model"]["name"], num_classes=data_cfg["num_classes"], image_size=data_cfg["image_size"])
    result = fit(model, train_loader, val_loader, epochs=cfg["training"]["epochs"], lr=cfg["training"]["lr"], device=device, patience=cfg["training"].get("patience", 5))
    y_true, probs = predict(model, test_loader, device)
    metrics = compute_metrics(y_true, probs)
    out_dir = Path(cfg.get("output_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
