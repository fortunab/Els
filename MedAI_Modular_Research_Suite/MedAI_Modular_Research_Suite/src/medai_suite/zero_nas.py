import random
import numpy as np
import tensorflow as tf
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
    for i in range(n_blocks):
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
            pool = random.choice(VALID_POOL_SIZES)
            x = layers.MaxPooling2D(pool_size=pool, padding="same")(x)

        details.append(
            {"filters": filters, "kernel": kernel, "depthwise": use_depthwise, "pool": use_pool}
        )

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    return model, details


def compute_activation_diversity(model, sample_images):
    conv_layers = [layer.output for layer in model.layers if isinstance(layer, (layers.Conv2D, layers.DepthwiseConv2D))]
    if not conv_layers:
        return 0.0

    activation_model = keras.Model(inputs=model.input, outputs=conv_layers)
    activations = activation_model.predict(sample_images, verbose=0)
    if not isinstance(activations, list):
        activations = [activations]

    scores = []
    for act in activations:
        flat = act.reshape((act.shape[0], -1))
        scores.append(float(np.mean(np.std(flat, axis=0))))

    return float(np.mean(scores))


def zero_cost_search(sample_images, n_architectures=20, input_shape=(28, 28, 1), num_classes=10):
    best = None
    records = []

    for i in range(n_architectures):
        model, details = generate_candidate(input_shape=input_shape, num_classes=num_classes)
        params = model.count_params()
        diversity = compute_activation_diversity(model, sample_images)
        score = diversity / np.log(params + 10)

        record = {
            "index": i,
            "params": int(params),
            "diversity": float(diversity),
            "score": float(score),
            "details": details,
        }
        records.append(record)

        if best is None or score > best["score"]:
            best = record
            best["model"] = model

    return best, records
