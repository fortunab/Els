"""CRC Mixture-of-Experts model definition."""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


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


class SparseMoELogits(layers.Layer):
    def __init__(self, experts, k=2, load_balance_weight=1e-2, **kwargs):
        super().__init__(**kwargs)
        self.experts = experts
        self.k = k
        self.load_balance_weight = load_balance_weight

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

        importance = tf.reduce_mean(router_probs, axis=0)
        self.add_loss(tf.reduce_sum(importance * importance) * self.load_balance_weight)
        return logits


def build_crc_moe_model(num_classes=8, img_size=96, num_experts=4, k_route=2):
    inputs = layers.Input((img_size, img_size, 3))
    backbone = build_backbone(img_size=img_size)
    feats = backbone(inputs)
    experts = [make_expert(num_classes, name=f"expert_{i}") for i in range(num_experts)]
    logits = SparseMoELogits(experts, k=k_route)(feats)
    outputs = layers.Activation("softmax")(logits)
    return keras.Model(inputs, outputs, name="CRC_ZeroCost_Pruned_MoE")
