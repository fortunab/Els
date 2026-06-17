import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist


def load_mnist(batch_size=128):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train[..., None].astype("float32") / 255.0
    x_test = x_test[..., None].astype("float32") / 255.0
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds, test_ds, (x_test, y_test)


def build_mlp():
    return models.Sequential(
        [
            layers.Flatten(input_shape=(28, 28, 1)),
            layers.Dense(64, activation="relu"),
            layers.Dense(10),
        ],
        name="MLP",
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
        name="CNN",
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
    return models.Model(inputs, outputs, name="ConvMixerLite")


def train_model(model, train_ds, test_ds, epochs=2):
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    history = model.fit(train_ds, validation_data=test_ds, epochs=epochs)
    return history


def get_saliency(model, image, label):
    image = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(image)
        logits = model(image, training=False)
        score = logits[:, int(label)]
    grads = tape.gradient(score, image)
    sal = tf.reduce_mean(tf.abs(grads), axis=-1).numpy()[0]
    sal = sal / (sal.max() + 1e-8)
    return sal


def sag_aggregate(model_list, image, label):
    """Saliency-Aligned Aggregation.

    Models with saliency maps that align better with the ensemble saliency receive higher weights.
    """
    logits = []
    saliencies = []
    for model in model_list:
        logits.append(model(image[None, ...], training=False).numpy()[0])
        saliencies.append(get_saliency(model, image, label))

    mean_sal = np.mean(saliencies, axis=0)
    weights = []
    for sal in saliencies:
        numerator = np.sum(sal * mean_sal)
        denominator = np.sqrt(np.sum(sal ** 2)) * np.sqrt(np.sum(mean_sal ** 2)) + 1e-8
        weights.append(numerator / denominator)

    weights = np.asarray(weights)
    weights = weights / (weights.sum() + 1e-8)

    agg_logits = np.sum([w * l for w, l in zip(weights, logits)], axis=0)
    return agg_logits, weights, saliencies
