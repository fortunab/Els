"""aaSAG model definitions."""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_mlp():
    return models.Sequential(
        [
            layers.Flatten(input_shape=(28, 28, 1)),
            layers.Dense(64, activation="relu"),
            layers.Dense(10),
        ],
        name="aaSAG_MLP",
    )


def build_cnn():
    return models.Sequential(
        [
            layers.Input((28, 28, 1)),
            layers.Conv2D(16, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, activation="relu"),
            layers.Flatten(),
            layers.Dense(10),
        ],
        name="aaSAG_CNN",
    )


def build_convmixer_lite():
    inputs = layers.Input((28, 28, 1))
    x = layers.Conv2D(32, 4, strides=4, padding="same", activation="gelu")(inputs)
    for _ in range(2):
        residual = x
        x = layers.DepthwiseConv2D(3, padding="same", activation="gelu")(x)
        x = layers.Add()([x, residual])
        x = layers.Conv2D(32, 1, activation="gelu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(10)(x)
    return models.Model(inputs, outputs, name="aaSAG_ConvMixerLite")
