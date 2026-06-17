import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers


def preprocess_crc(example, img_size=96, training=False):
    image = tf.image.resize(example["image"], [img_size, img_size])
    image = tf.cast(image, tf.float32) / 255.0
    if training:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.08)
    return image, example["label"]


def load_crc_tfds(img_size=96, batch_size=32, seed=42):
    ds = tfds.load("colorectal_histology", split="train", as_supervised=False)
    ds = ds.shuffle(5000, seed=seed, reshuffle_each_iteration=True)
    train = ds.take(4000).map(lambda x: preprocess_crc(x, img_size, True), num_parallel_calls=tf.data.AUTOTUNE)
    val = ds.skip(4000).take(500).map(lambda x: preprocess_crc(x, img_size, False), num_parallel_calls=tf.data.AUTOTUNE)
    test = ds.skip(4500).take(500).map(lambda x: preprocess_crc(x, img_size, False), num_parallel_calls=tf.data.AUTOTUNE)
    return (
        train.batch(batch_size).prefetch(tf.data.AUTOTUNE),
        val.batch(batch_size).prefetch(tf.data.AUTOTUNE),
        test.batch(batch_size).prefetch(tf.data.AUTOTUNE),
        8,
    )


def build_backbone(img_size=96, feature_dim=256):
    inputs = layers.Input((img_size, img_size, 3))
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.DepthwiseConv2D(3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 1, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.DepthwiseConv2D(3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, 1, activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(feature_dim, activation="relu")(x)
    return keras.Model(inputs, outputs, name="crc_light_backbone")


def make_expert(num_classes, hidden=256, name="expert"):
    return keras.Sequential(
        [
            layers.Dense(hidden, activation="relu"),
            layers.Dropout(0.15),
            layers.Dense(num_classes),
        ],
        name=name,
    )


def expert_gradnorm_score(expert, feats, labels):
    with tf.GradientTape() as tape:
        logits = expert(feats, training=True)
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True))
    grads = tape.gradient(loss, expert.trainable_variables)
    score = tf.add_n([tf.reduce_sum(tf.abs(g)) for g in grads if g is not None])
    return float(score.numpy())


def score_experts_zero_cost(backbone, experts, train_ds, batches=1):
    scores = np.zeros(len(experts), dtype=np.float64)
    for b, (x, y) in enumerate(train_ds.take(batches)):
        feats = backbone(x, training=False)
        for i, expert in enumerate(experts):
            scores[i] += expert_gradnorm_score(expert, feats, y)
    return scores.tolist()


class SparseMoELogits(layers.Layer):
    def __init__(self, experts, k=2, load_balance_weight=1e-2, **kwargs):
        super().__init__(**kwargs)
        self.experts = experts
        self.k = k
        self.load_balance_weight = load_balance_weight
        self.router = None

    def build(self, input_shape):
        self.router = layers.Dense(len(self.experts), name="router")
        super().build(input_shape)

    def call(self, feats, training=None):
        router_logits = self.router(feats)
        router_probs = tf.nn.softmax(router_logits, axis=-1)
        top_values, top_indices = tf.math.top_k(router_probs, k=min(self.k, len(self.experts)))

        expert_outputs = tf.stack([expert(feats, training=training) for expert in self.experts], axis=1)
        one_hot = tf.one_hot(top_indices, depth=len(self.experts))
        weights = one_hot * tf.expand_dims(top_values, -1)
        weights = weights / (tf.reduce_sum(weights, axis=1, keepdims=True) + 1e-8)
        mixed_weights = tf.reduce_sum(weights, axis=1)

        logits = tf.reduce_sum(expert_outputs * tf.expand_dims(mixed_weights, -1), axis=1)

        # Simple load-balance penalty.
        importance = tf.reduce_mean(router_probs, axis=0)
        lb_loss = tf.reduce_sum(importance * importance) * self.load_balance_weight
        self.add_loss(lb_loss)
        return logits


def build_pruned_moe_model(num_classes, selected_experts, img_size=96, k_route=2):
    inputs = layers.Input((img_size, img_size, 3))
    backbone = build_backbone(img_size=img_size)
    feats = backbone(inputs)
    logits = SparseMoELogits(selected_experts, k=k_route)(feats)
    outputs = layers.Activation("softmax")(logits)
    return keras.Model(inputs, outputs, name="CRC_ZeroCost_Pruned_MoE")


def build_crc_moe_experiment(
    img_size=96,
    num_experts_pool=8,
    topk_experts=4,
    topk_route=2,
    score_batches=1,
    seed=42,
):
    train_ds, val_ds, test_ds, num_classes = load_crc_tfds(img_size=img_size, seed=seed)
    backbone = build_backbone(img_size=img_size)
    experts_pool = [make_expert(num_classes, name=f"expert_{i}") for i in range(num_experts_pool)]

    # Build variables.
    for x, _ in train_ds.take(1):
        feats = backbone(x)
        for expert in experts_pool:
            _ = expert(feats)

    scores = score_experts_zero_cost(backbone, experts_pool, train_ds, batches=score_batches)
    selected_idx = np.argsort(scores)[-topk_experts:][::-1].tolist()
    selected_experts = [experts_pool[i] for i in selected_idx]

    model = build_pruned_moe_model(num_classes, selected_experts, img_size=img_size, k_route=topk_route)
    return model, train_ds, val_ds, test_ds, scores, selected_idx
