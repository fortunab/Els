"""Zero-cost NAS candidate model generator."""

import random
from tensorflow import keras
from tensorflow.keras import layers


VALID_FILTERS = [8, 16, 24, 32, 48, 64]
VALID_KERNEL_SIZES = [3, 5, 7]
VALID_POOL_SIZES = [2, 3]


def generate_candidate(input_shape=(28, 28, 1), num_classes=10, max_blocks=3):
    inputs = layers.Input(input_shape)
    x = inputs
    details = []

    n_blocks = random.randint(1, max_blocks)
    for _ in range(n_blocks):
        filters = random.choice(VALID_FILTERS)
        kernel = random.choice(VALID_KERNEL_SIZES)
        use_depthwise = random.choice([True, False])
        use_pool = random.choice([True, False])

        if use_depthwise:
            x = layers.DepthwiseConv2D(kernel, padding="same", activation="relu")(x)
            x = layers.Conv2D(filters, 1, activation="relu")(x)
        else:
            x = layers.Conv2D(filters, kernel, padding="same", activation="relu")(x)

        x = layers.BatchNormalization()(x)

        if use_pool:
            x = layers.MaxPooling2D(pool_size=random.choice(VALID_POOL_SIZES), padding="same")(x)

        details.append({"filters": filters, "kernel": kernel, "depthwise": use_depthwise, "pool": use_pool})

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="ZeroNAS_Candidate"), details
