from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from .data import load_datasets
from .model import build_model
from .utils import ensure_dir, load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train EfficientNetB0 Alzheimer's MRI classifier.")
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    img_size = tuple(cfg["img_size"])
    train_ds, test_ds, class_names = load_datasets(
        cfg["train_dir"], cfg["test_dir"], img_size, int(cfg["batch_size"])
    )

    model = build_model(
        img_size=img_size,
        num_classes=int(cfg["num_classes"]),
        fine_tune_from=cfg.get("fine_tune_from", "block5a_expand_activation"),
    )

    model.compile(
        optimizer=Adam(learning_rate=float(cfg["learning_rate"])),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            min_delta=0.01,
            patience=11,
            verbose=1,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        epochs=int(cfg["epochs"]),
        validation_data=test_ds,
        callbacks=callbacks,
    )

    model_path = Path(cfg["model_path"])
    ensure_dir(model_path.parent)
    model.save(model_path)

    reports_dir = ensure_dir(cfg["reports_dir"])
    pd.DataFrame(history.history).to_csv(reports_dir / "training_history.csv", index=False)
    with open(reports_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
