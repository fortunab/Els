from __future__ import annotations

import argparse
import glob
import os
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .utils import ensure_dir, load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Eigen-CAM style heatmap overlays.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--images-per-class", type=int, default=3)
    return parser.parse_args()


def find_last_conv_layer(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except Exception:
            continue
    raise ValueError("Could not find a 4D convolutional feature layer.")


def load_image(path: str, img_size: tuple[int, int]):
    img = tf.keras.utils.load_img(path, target_size=img_size)
    img = tf.keras.utils.img_to_array(img).astype(np.float32)
    return np.expand_dims(img, axis=0), img.astype(np.uint8)


def eigen_attention_map(img_array: np.ndarray, feature_model: tf.keras.Model, img_size: tuple[int, int]):
    features = feature_model(img_array)[0].numpy()
    features = np.maximum(features, 0)

    h, w, c = features.shape
    reshaped = features.reshape((-1, c))
    reshaped = reshaped - reshaped.mean(axis=0)

    _, _, vt = np.linalg.svd(reshaped, full_matrices=False)
    cam = reshaped @ vt[0]
    cam = cam.reshape(h, w)
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    cam = cv2.resize(cam, img_size)
    cam = cv2.GaussianBlur(cam, (9, 9), 0)
    return cam / (cam.max() + 1e-8)


def full_heatmap_overlay(image: np.ndarray, cam: np.ndarray):
    cam_uint8 = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image, 0.55, heatmap, 0.45, 0)
    return heatmap, overlay


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    img_size = tuple(cfg["img_size"])
    test_dir = cfg["test_dir"]
    model = tf.keras.models.load_model(cfg["model_path"])

    class_names = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    last_conv_layer_name = find_last_conv_layer(model)
    feature_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=model.get_layer(last_conv_layer_name).output,
    )

    selected_images = []
    for class_name in class_names:
        image_files = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            image_files.extend(glob.glob(os.path.join(test_dir, class_name, ext)))
        chosen = random.sample(image_files, min(args.images_per_class, len(image_files)))
        selected_images.extend((path, class_name) for path in chosen)

    if not selected_images:
        raise ValueError(f"No images found in {test_dir}")

    rows = len(selected_images)
    plt.figure(figsize=(18, rows * 4))

    for idx, (img_path, true_class) in enumerate(selected_images):
        img_array, img = load_image(img_path, img_size)
        predictions = model.predict(img_array, verbose=0)[0]
        pred_class = class_names[int(np.argmax(predictions))]

        cam = eigen_attention_map(img_array, feature_model, img_size)
        heatmap, overlay = full_heatmap_overlay(img, cam)

        plt.subplot(rows, 3, idx * 3 + 1)
        plt.imshow(img)
        plt.title(f"Original\nTrue: {true_class}")
        plt.axis("off")

        plt.subplot(rows, 3, idx * 3 + 2)
        plt.imshow(heatmap)
        plt.title("Full Attention Heatmap")
        plt.axis("off")

        plt.subplot(rows, 3, idx * 3 + 3)
        plt.imshow(overlay)
        plt.title(f"Overlay\nPredicted: {pred_class}")
        plt.axis("off")

    out_dir = ensure_dir(cfg["gradcam_dir"])
    out_path = Path(out_dir) / "eigen_cam_grid.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Eigen-CAM grid to: {out_path}")
    print(f"Last convolutional layer used: {last_conv_layer_name}")


if __name__ == "__main__":
    main()
