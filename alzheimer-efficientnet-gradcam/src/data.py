from __future__ import annotations

from pathlib import Path
from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.efficientnet import preprocess_input


def _check_dir(path: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {path}. "
            "Update configs/config.yaml or create data/train and data/test."
        )


def load_datasets(
    train_dir: str,
    test_dir: str,
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
):
    _check_dir(train_dir)
    _check_dir(test_dir)

    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
        verbose=True,
    )

    test_ds = keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        batch_size=batch_size,
        image_size=img_size,
        shuffle=False,
        verbose=True,
    )

    class_names = train_ds.class_names

    def preprocess(image, label):
        return preprocess_input(image), label

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.map(preprocess).prefetch(autotune)
    test_ds = test_ds.map(preprocess).prefetch(autotune)

    return train_ds, test_ds, class_names
