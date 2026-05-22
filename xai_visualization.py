# ==========================================================
# Explainable AI Pipeline (GradCAM / GradCAM++ / ScoreCAM)
# ==========================================================

import os
import glob
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.utils.scores import CategoricalScore

# ==========================================================
# DATASET PATH
# ==========================================================

DATASET_PATH = "/kaggle/input/dataset-alzheimer/Alzheimer_s Dataset"

test_dir = os.path.join(DATASET_PATH, "test")

# ==========================================================
# LOAD TRAINED MODEL
# ==========================================================

model = tf.keras.models.load_model("best_model.keras")

# ==========================================================
# LOAD RANDOM IMAGE
# ==========================================================

all_images = glob.glob(test_dir + "/*/*.jpg")

img_path = random.choice(all_images)

true_class = os.path.basename(os.path.dirname(img_path))

print("Selected image:", img_path)
print("True class:", true_class)

# ==========================================================
# IMAGE PREPROCESSING
# ==========================================================

def load_img(path):
    img = tf.keras.utils.load_img(path, target_size=(224,224))
    arr = tf.keras.utils.img_to_array(img) / 255.0
    return np.expand_dims(arr,0), arr

x, orig = load_img(img_path)

# ==========================================================
# PREDICTION
# ==========================================================

pred = model.predict(x)
pred_class = np.argmax(pred)

print("Predicted class:", pred_class)

# ==========================================================
# KERAS 3 FIX
# ==========================================================

def replace_to_linear_keras3(model_instance):
    model_instance.layers[-1].activation = tf.keras.activations.linear

score = CategoricalScore(pred_class)

# ==========================================================
# CAM INITIALIZATION
# ==========================================================

gradcam = Gradcam(
    model,
    model_modifier=replace_to_linear_keras3,
    clone=False
)

gradcam_pp = GradcamPlusPlus(
    model,
    model_modifier=replace_to_linear_keras3,
    clone=False
)

scorecam = Scorecam(
    model,
    model_modifier=replace_to_linear_keras3,
    clone=False
)

# ==========================================================
# GENERATE HEATMAPS
# ==========================================================

g = gradcam(score, x)[0]
gp = gradcam_pp(score, x)[0]
sc = scorecam(score, x)[0]

# ==========================================================
# NORMALIZATION
# ==========================================================

def norm(c):
    c = np.maximum(c, 0)
    denom = c.max()
    return c / (denom + 1e-8) if denom > 0 else c

g, gp, sc = norm(g), norm(gp), norm(sc)

# ==========================================================
# OVERLAY FUNCTION
# ==========================================================

def overlay(cam, img):

    img_uint8 = np.uint8(img * 255)

    cam = cv2.resize(cam, (224,224))
    cam = np.uint8(255 * cam)

    heat = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(
        img_uint8,
        0.6,
        heat,
        0.4,
        0
    )

    gray_img = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

    _, mask = cv2.threshold(
        gray_img,
        1,
        255,
        cv2.THRESH_BINARY
    )

    fixed_overlay = cv2.bitwise_and(
        blended,
        blended,
        mask=mask
    )

    return fixed_overlay

g_img = overlay(g, orig)
gp_img = overlay(gp, orig)
sc_img = overlay(sc, orig)

# ==========================================================
# VISUALIZATION
# ==========================================================

plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(orig)
plt.title(f"Original MRI\\n(True: {true_class})")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(g_img)
plt.title("GradCAM")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(gp_img)
plt.title("GradCAM++")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(sc_img)
plt.title("ScoreCAM")
plt.axis("off")

plt.tight_layout()
plt.show()
