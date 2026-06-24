from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import Model, regularizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    Flatten,
    Input,
    RandomFlip,
    RandomRotation,
)


def build_model(
    img_size: Tuple[int, int] = (224, 224),
    num_classes: int = 4,
    fine_tune_from: str = "block5a_expand_activation",
) -> Model:
    inputs = Input(shape=(*img_size, 3))
    x = RandomFlip("horizontal")(inputs)
    x = RandomRotation(0.1)(x)

    backbone = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
    )

    backbone.trainable = True
    set_trainable = False
    for layer in backbone.layers:
        if layer.name == fine_tune_from:
            set_trainable = True
        layer.trainable = set_trainable

    x = backbone.output
    x = Flatten()(x)
    x = Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    return Model(inputs=inputs, outputs=outputs)
