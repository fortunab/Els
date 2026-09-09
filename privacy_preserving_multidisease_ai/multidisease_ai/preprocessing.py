"""
Preprocessing and Enhancement Pipelines
Section 3.3 of the paper:
- 3.3.1 Image Normalization and Standardization (Eq. 4)
- 3.3.2 Noise Reduction and Denoising ZS-N2N (Eq. 5)
- 3.3.3 Autoencoder-Based Feature Enhancement VAE/MAE (Eq. 6-7)
- 3.3.4 Domain Adaptation Using CycleGAN (Eq. 8)
"""

import numpy as np

def min_max_normalize(image: np.ndarray) -> np.ndarray:
    """
    Min-Max Scaling (Eq. 4):
    x_norm = (x - min(x)) / (max(x) - min(x))
    """
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val - min_val == 0:
        return np.zeros_like(image, dtype=np.float32)
    return (image - min_val) / (max_val - min_val)

def calculate_zs_n2n_loss(denoised_output: np.ndarray, noisy_realization: np.ndarray) -> float:
    """
    Zero-Shot Noise-to-Noise Denoising Objective (Eq. 5):
    L_N2N(theta) = E [ || D_theta(x) - x' ||_2^2 ]
    """
    diff = denoised_output - noisy_realization
    return float(np.mean(diff ** 2))

def calculate_autoencoder_reconstruction_loss(original_image: np.ndarray, reconstructed_image: np.ndarray) -> float:
    """
    VAE/MAE Reconstruction Objective (Eq. 6-7):
    L_AE = || x - x_hat ||_2^2
    """
    diff = original_image - reconstructed_image
    return float(np.mean(diff ** 2))

def calculate_cyclegan_consistency_loss(
    source_x: np.ndarray,
    reconstructed_x: np.ndarray,
    target_y: np.ndarray,
    reconstructed_y: np.ndarray
) -> float:
    """
    CycleGAN Cycle-Consistency Objective (Eq. 8):
    L_cyc(G, F) = E_{x~A}[ || F(G(x)) - x ||_1 ] + E_{y~B}[ || G(F(y)) - y ||_1 ]
    """
    l1_x = np.mean(np.abs(reconstructed_x - source_x))
    l1_y = np.mean(np.abs(reconstructed_y - target_y))
    return float(l1_x + l1_y)
