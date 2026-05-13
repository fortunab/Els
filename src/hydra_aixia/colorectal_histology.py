from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow import keras
from tensorflow.keras import layers


@dataclass
class ColorectalConfig:
    dataset_name: str = "colorectal_histology"
    image_size: int = 64
    batch_size: int = 32
    train_fraction: float = 0.8
    epochs: int = 5
    model_path: str = "models/cnn_colorectal_histology.h5"


def load_dataset(config: ColorectalConfig):
    ds_full, ds_info = tfds.load(config.dataset_name, split="train", as_supervised=True, with_info=True)
    train_count = int(config.train_fraction * ds_info.splits["train"].num_examples)

    def preprocess(image, label):
        image = tf.image.resize(image, (config.image_size, config.image_size)) / 255.0
        return image, label

    ds_train = ds_full.take(train_count).map(preprocess).batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
    ds_test = ds_full.skip(train_count).map(preprocess).batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
    return ds_train, ds_test, ds_info


def build_cnn(input_shape: tuple[int, int, int], num_classes: int) -> keras.Model:
    return keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (5, 5), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (5, 5), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.1),
        layers.Dense(num_classes, activation="softmax"),
    ])


def train_and_save(config: ColorectalConfig = ColorectalConfig()) -> keras.Model:
    ds_train, ds_test, ds_info = load_dataset(config)
    model = build_cnn((config.image_size, config.image_size, 3), ds_info.features["label"].num_classes)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(ds_train, epochs=config.epochs, validation_data=ds_test)
    os.makedirs(os.path.dirname(config.model_path), exist_ok=True)
    model.save(config.model_path)
    return model


def summarize_medical_terms(texts: dict[str, str], num_topics: int = 2, num_words: int = 20) -> list[str]:
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(texts.values())
    nmf = NMF(n_components=num_topics, random_state=42).fit(tfidf)
    names = vectorizer.get_feature_names_out()
    return [" ".join(names[i] for i in topic.argsort()[:-num_words - 1:-1]) for topic in nmf.components_]


def predict_batch(model: keras.Model, images: np.ndarray, ds_info) -> tuple[str, float]:
    predictions = model.predict(images)
    predicted_class = int(np.argmax(predictions[0]))
    predicted_class = max(0, min(predicted_class, ds_info.features["label"].num_classes - 1))
    return ds_info.features["label"].int2str(predicted_class), float(np.max(predictions[0]) * 100)
