"""
Explainability & Attention Visualization Engine
Section 3.7 & Section 4.5 (Table 4 & Figure 6):
- Grad-CAM importance weight formula (Eq. 12)
- Grad-CAM++, Score-CAM, and Transformer Self-Attention
- Localization accuracy, IoU, and Clinician Agreement metrics across disease domains
- Alzheimer MRI stage visual explanation matrix simulation (Figure 6)
"""

import numpy as np
from typing import Dict, Any

def compute_gradcam_weights(feature_map_gradients: np.ndarray) -> np.ndarray:
    """
    Grad-CAM importance weight alpha_k^c (Eq. 12):
    alpha_k^c = (1 / Z) * sum_i sum_j ( d y^c / d A_{ij}^k )
    """
    Z = feature_map_gradients.shape[0] * feature_map_gradients.shape[1]
    return np.sum(feature_map_gradients, axis=(0, 1)) / Z

def generate_synthetic_heatmap(grid_size: int = 64, mode: str = "gradcam", stage: str = "Mild") -> np.ndarray:
    """
    Generates synthetic activation heatmaps simulating Alzheimer MRI stage classification (Figure 6).
    Modes: 'gradcam', 'gradcam++', 'scorecam'
    Stages: 'Non demented', 'Moderate', 'Very mild', 'Mild'
    """
    x = np.linspace(-3, 3, grid_size)
    y = np.linspace(-3, 3, grid_size)
    xx, yy = np.meshgrid(x, y)
    
    # Base brain ventricular/cortical atrophy profile
    r = np.sqrt(xx**2 + yy**2)
    
    if stage == "Non demented":
        center_intensity = 0.3
        sigma = 1.2
    elif stage == "Moderate":
        center_intensity = 0.95
        sigma = 0.7
    elif stage == "Very mild":
        center_intensity = 0.55
        sigma = 1.0
    else:  # Mild
        center_intensity = 0.75
        sigma = 0.85
        
    heatmap = center_intensity * np.exp(-(r**2) / (2 * sigma**2))
    
    if mode == "gradcam++":
        # Sharper activations
        heatmap = np.power(heatmap, 1.8)
    elif mode == "scorecam":
        # Smoother contextual activations
        heatmap = 0.85 * heatmap + 0.15 * np.exp(-((xx-1)**2 + (yy-1)**2)/2)

    # Normalize to [0, 1]
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
    return heatmap

EXPLAINABILITY_TABLE_DATA = {
    "Colorectal polyps": {
        "localization_accuracy": 0.9318,
        "iou": 0.8642,
        "clinician_agreement": 0.9027
    },
    "Cervical cytology": {
        "localization_accuracy": 0.9145,
        "iou": 0.8426,
        "clinician_agreement": 0.8874
    },
    "Alzheimer’s disease": {
        "localization_accuracy": 0.8961,
        "iou": 0.8175,
        "clinician_agreement": 0.8738
    },
    "Diabetic retinopathy": {
        "localization_accuracy": 0.9273,
        "iou": 0.8519,
        "clinician_agreement": 0.8962
    },
    "Skin lesions": {
        "localization_accuracy": 0.9084,
        "iou": 0.8361,
        "clinician_agreement": 0.8829
    }
}
