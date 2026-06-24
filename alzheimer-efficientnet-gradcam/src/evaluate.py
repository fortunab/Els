from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize

from .data import load_datasets
from .utils import ensure_dir, load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained Alzheimer's MRI classifier.")
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    img_size = tuple(cfg["img_size"])
    _, test_ds, class_names = load_datasets(
        cfg["train_dir"], cfg["test_dir"], img_size, int(cfg["batch_size"])
    )

    model = tf.keras.models.load_model(cfg["model_path"])
    reports_dir = ensure_dir(cfg["reports_dir"])

    loss, accuracy = model.evaluate(test_ds, verbose=1)
    y_true, y_pred, y_probs = [], [], []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_probs.append(probs)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(probs, axis=1))

    y_probs = np.concatenate(y_probs, axis=0)
    num_classes = len(class_names)
    y_true_one_hot = label_binarize(y_true, classes=range(num_classes))

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    conf = confusion_matrix(y_true, y_pred)

    with open(reports_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"loss": float(loss), "accuracy": float(accuracy), "report": report}, f, indent=2)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(reports_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 8))
    fpr_micro, tpr_micro, _ = roc_curve(y_true_one_hot.ravel(), y_probs.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(fpr_micro, tpr_micro, label=f"Micro-average ROC AUC = {auc_micro:.2f}")

    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_one_hot[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{class_name} AUC = {roc_auc:.2f}")

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-class ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(reports_dir / "roc_curve.png", dpi=300)
    plt.close()

    print(f"Test accuracy: {accuracy * 100:.2f}%")
    print(f"Test loss: {loss:.4f}")
    print(f"Reports saved to: {reports_dir}")


if __name__ == "__main__":
    main()
