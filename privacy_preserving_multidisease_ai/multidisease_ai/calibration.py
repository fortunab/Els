"""
Uncertainty Estimation, Calibration & Benchmark Tables
Section 4.2, 4.6, 4.8, 4.9 (Tables 1, 5, 8, 9, 10, 11):
- Expected Calibration Error (ECE)
- Brier Score & Predictive Entropy
- Comprehensive multi-disease benchmarking and ablation datasets
"""

import numpy as np
from typing import Dict, List, Tuple

def calculate_ece(confidences: np.ndarray, predictions: np.ndarray, labels: np.ndarray, num_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE):
    ECE = sum_{b=1}^B (|B_b| / N) * | acc(B_b) - conf(B_b) |
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    n = len(confidences)
    
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(predictions[in_bin] == labels[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
            
    return float(ece)

def calculate_brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """
    Brier Score:
    BS = (1 / N) * sum (p_i - y_i)^2
    """
    return float(np.mean((probabilities - labels) ** 2))

def calculate_entropy(probabilities: np.ndarray) -> float:
    """
    Predictive Entropy:
    H = - sum p_i * log(p_i)
    """
    eps = 1e-12
    p = np.clip(probabilities, eps, 1 - eps)
    return float(-np.mean(p * np.log(p) + (1 - p) * np.log(1 - p)))

# Table 1: Classification performance of framework across disease domains
TABLE_1_DATA = {
    "Colorectal polyps": {"accuracy": 0.9324, "precision": 0.9142, "sensitivity": 0.9287, "f1_score": 0.9214, "auc": 0.9518},
    "Cervical cytology": {"accuracy": 0.9106, "precision": 0.8931, "sensitivity": 0.9075, "f1_score": 0.9002, "auc": 0.9247},
    "Alzheimer’s disease": {"accuracy": 0.8893, "precision": 0.8728, "sensitivity": 0.8916, "f1_score": 0.8821, "auc": 0.9035},
    "Diabetic retinopathy": {"accuracy": 0.9217, "precision": 0.9054, "sensitivity": 0.9189, "f1_score": 0.9121, "auc": 0.9384},
    "Skin lesions": {"accuracy": 0.9012, "precision": 0.8876, "sensitivity": 0.8993, "f1_score": 0.8934, "auc": 0.9142}
}

# Table 5: Uncertainty estimation and calibration performance across disease domains
TABLE_5_DATA = {
    "Colorectal polyps": {"ece": 0.0217, "brier_score": 0.0734, "entropy": 0.1842},
    "Cervical cytology": {"ece": 0.0276, "brier_score": 0.0815, "entropy": 0.1967},
    "Alzheimer’s disease": {"ece": 0.0341, "brier_score": 0.0928, "entropy": 0.2149},
    "Diabetic retinopathy": {"ece": 0.0234, "brier_score": 0.0761, "entropy": 0.1885},
    "Skin lesions": {"ece": 0.0298, "brier_score": 0.0854, "entropy": 0.2013}
}

# Table 8: Extended evaluation metrics across disease domains
TABLE_8_DATA = {
    "Colorectal polyps": {"pr_auc": 0.9442, "froc": 0.9126, "mcc": 0.8873},
    "Cervical cytology": {"pr_auc": 0.9138, "froc": 0.8861, "mcc": 0.8421},
    "Alzheimer’s disease": {"pr_auc": 0.8912, "froc": 0.8547, "mcc": 0.8067},
    "Diabetic retinopathy": {"pr_auc": 0.9296, "froc": 0.9018, "mcc": 0.8649},
    "Skin lesions": {"pr_auc": 0.9027, "froc": 0.8734, "mcc": 0.8283}
}

# Table 9: Extended comparative analysis using PR-AUC, FROC, MCC
TABLE_9_DATA = {
    "ResNet50": {"pr_auc": 0.8914, "froc": 0.8647, "mcc": 0.7816},
    "EfficientNet-B4": {"pr_auc": 0.9078, "froc": 0.8819, "mcc": 0.8127},
    "Vision transformer": {"pr_auc": 0.9216, "froc": 0.8964, "mcc": 0.8368},
    "FedAvg-CNN": {"pr_auc": 0.9297, "froc": 0.9043, "mcc": 0.8512},
    "Hybrid CNN-ViT": {"pr_auc": 0.9378, "froc": 0.9135, "mcc": 0.8729},
    "Architecture MH": {"pr_auc": 0.9442, "froc": 0.9126, "mcc": 0.8873}
}

# Table 10: Comparative analysis between proposed framework and SOTA multi-disease medical AI systems
TABLE_10_DATA = {
    "MedViT MTL [56]": {"architecture": "Transformer Multi-Task", "federated": "No", "expl": "Partial", "accuracy": 0.9018, "auc": 0.9187, "ece": 0.0415},
    "Hybrid-MedNet [15]": {"architecture": "CNN-Based Multi-Disease", "federated": "No", "expl": "No", "accuracy": 0.8946, "auc": 0.9102, "ece": 0.0461},
    "FedHealth [21]": {"architecture": "Federated CNN", "federated": "Yes", "expl": "No", "accuracy": 0.9127, "auc": 0.9278, "ece": 0.0368},
    "MedFuse-Transformer [30]": {"architecture": "Hybrid CNN-ViT", "federated": "No", "expl": "Partial", "accuracy": 0.9185, "auc": 0.9346, "ece": 0.0324},
    "Federated MedViT [28]": {"architecture": "Federated Transformer", "federated": "Yes", "expl": "Partial", "accuracy": 0.9241, "auc": 0.9417, "ece": 0.0283},
    "Architecture MH": {"architecture": "Hybrid Federated NAS", "federated": "Yes", "expl": "Yes", "accuracy": 0.9342, "auc": 0.9516, "ece": 0.0217}
}

# Table 11: Ablation study of the proposed MH framework
TABLE_11_DATA = {
    "Full MH framework": {"accuracy": 0.9342, "f1_score": 0.9238, "auc": 0.9516, "ece": 0.0217},
    "W/o federated learning": {"accuracy": 0.9181, "f1_score": 0.9074, "auc": 0.9365, "ece": 0.0326},
    "W/o preprocessing": {"accuracy": 0.9097, "f1_score": 0.8983, "auc": 0.9271, "ece": 0.0369},
    "W/o NAS optimization": {"accuracy": 0.9214, "f1_score": 0.9112, "auc": 0.9398, "ece": 0.0297},
    "W/o explainability module": {"accuracy": 0.9258, "f1_score": 0.9149, "auc": 0.9431, "ece": 0.0284},
    "W/o calibration mechanism": {"accuracy": 0.9286, "f1_score": 0.9172, "auc": 0.9453, "ece": 0.0418}
}

# Figure 1: Heatmap Mapping of ML Approaches across Health Conditions
# Rows: ML Methods (CNN, SVM, ResNet, Attn, MLP, RF, Tr/ViT, DenseNet, DT, KNN, LSTM, AlexNet)
# Columns: Disease Groups (Alzheimer, Respiratory, Pneumonia, Dementia + MCI, Brain tumor, Cardiovascular, Eye disease & Glaucoma, Skin cancer & lesions, Diabetic retinopathy, Parkinson, Breast cancer)
FIGURE_1_CATEGORIES = [
    "Alzheimer", "Respiratory", "Pneumonia", "Dementia + MCI", "Brain tumor", 
    "Cardiovascular", "Eye disease & Glaucoma", "Skin cancer & lesions", 
    "Diabetic retinopathy", "Parkinson", "Breast cancer"
]

FIGURE_1_METHODS = ["CNN", "SVM", "ResNet", "Attn", "MLP", "RF", "Tr/ViT", "DenseNet", "DT", "KNN", "LSTM", "AlexNet"]

FIGURE_1_MATRIX = [
    [61, 52, 51, 46, 38, 34, 49, 30, 26, 17, 16], # CNN
    [29, 16,  9, 33, 25, 28, 14, 10,  8, 16,  8], # SVM
    [17, 24, 28, 16, 11, 13, 21, 13, 11,  1,  6], # ResNet
    [21, 17, 11, 16, 15,  7, 14, 10,  7,  4,  4], # Attn
    [16,  8, 11, 12, 10, 15,  9,  6,  6,  5,  7], # MLP
    [14,  4,  1, 11, 11, 16,  5,  2,  3,  4,  1], # RF
    [ 6,  6,  7,  4,  8,  1,  9, 10,  5,  3,  4], # Tr/ViT
    [ 7,  8,  9,  6,  7,  5, 11,  9,  6,  0,  3], # DenseNet
    [ 6,  3,  1,  6,  5, 11,  1,  1,  0,  3,  3], # DT
    [ 8,  4,  1,  5,  4,  8,  2,  0,  2,  5,  3], # KNN
    [11,  3,  0,  8,  4,  4,  1,  2,  1,  8,  1], # LSTM
    [ 2,  1,  3,  4,  4,  3,  1,  5,  0,  0,  1]  # AlexNet
]
