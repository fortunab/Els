# ==========================================================
# Alzheimer CNN Training Pipeline
# ==========================================================

import os
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ==========================================================
# DATASET PATH
# ==========================================================

DATASET_PATH = "/kaggle/input/dataset-alzheimer/Alzheimer_s Dataset"

train_dir = os.path.join(DATASET_PATH, "train")
test_dir = os.path.join(DATASET_PATH, "test")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ==========================================================
# DATA GENERATORS
# ==========================================================

train_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = train_gen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ==========================================================
# CNN MODEL
# ==========================================================

inputs = layers.Input(shape=(224, 224, 3))

x = layers.Conv2D(64, 3, activation='relu',
                  kernel_regularizer=regularizers.l2(0.001))(inputs)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(128, 3, activation='relu',
                  kernel_regularizer=regularizers.l2(0.001))(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(256, 3, activation='relu',
                  kernel_regularizer=regularizers.l2(0.001))(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(512, 3, activation='relu',
                  kernel_regularizer=regularizers.l2(0.001))(x)
x = layers.MaxPooling2D()(x)

x = layers.Flatten()(x)

x = layers.Dense(
    512,
    activation='relu',
    kernel_regularizer=regularizers.l2(0.001)
)(x)

outputs = layers.Dense(4, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

# ==========================================================
# COMPILE MODEL
# ==========================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==========================================================
# CALLBACKS
# ==========================================================

callbacks = [
    ModelCheckpoint(
        "best_model.keras",
        save_best_only=True,
        monitor="val_accuracy"
    ),
    EarlyStopping(
        patience=5,
        restore_best_weights=True
    )
]

# ==========================================================
# TRAINING
# ==========================================================

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=25,
    callbacks=callbacks
)

# ==========================================================
# EVALUATION
# ==========================================================

test_loss, test_acc = model.evaluate(test_data)

print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")
