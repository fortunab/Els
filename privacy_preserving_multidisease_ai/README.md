# Privacy-Preserving & Explainable Multi-Disease AI Framework

Full implementation and experimental reproduction suite for the paper:
> **"A Privacy-Preserving and Explainable Multi-Disease AI Framework for Clinical Decision Support"**

This repository provides a complete Python architecture engine, CLI benchmark reproduction scripts, model training studio, and an interactive Web Dashboard Studio to reproduce, train, and inspect **all 11 tables, 6 figures, and Algorithm 1** from the study.

---

## 📁 Repository Structure

```text
privacy_preserving_multidisease_ai/
│
├── multidisease_ai/                # Core Python Framework Package
│   ├── __init__.py                 # Package metadata
│   ├── preprocessing.py            # Min-Max norm, ZS-N2N, VAE/MAE, CycleGAN (Eq. 4-8)
│   ├── architectures.py            # Self-Attention, CNN, ViT, Hybrid, SOTA baselines (Eq. 9)
│   ├── federated.py                # Algorithm 1 multi-hospital FL engine (FedAvg & FedProx)
│   ├── nas_engine.py               # Multi-objective NAS optimizer (Eq. 10-11)
│   ├── explainability.py           # Grad-CAM (Eq. 12), Grad-CAM++, Score-CAM, Attention
│   └── calibration.py              # ECE, Brier Score, Predictive Entropy, Table definitions
│
├── results.py                      # Master reproduction CLI script (Tables 1-11 & Figs 1, 5)
├── train_models.py                 # Interactive & CLI model training studio (Architecture MH & SOTA)
├── results_summary.json            # Exported quantitative experimental metrics
│
├── web_app/                        # Interactive Web Dashboard Studio
│   ├── index.html                  # Main UI with 6 tabbed views & interactive canvas/charts
│   ├── styles.css                  # Dark mode glassmorphism design system
│   ├── app.js                      # Real-time FL simulator, XAI matrix, training engine
│   └── plots/                      # Generated matplotlib plots (figure_1_heatmap, figure_5)
│
└── README.md                       # Project documentation & execution guide
```

---

## ⚡ Quickstart Guide

### 1. Prerequisites
Ensure you have Python 3.8+ installed. The framework relies on standard mathematical and data libraries:
```bash
pip install numpy matplotlib seaborn
```

---

### 2. Reproduce All Paper Tables & Figures
To reproduce all **11 tables** and generate plots for **Figures 1 & 5**, run:

```bash
python results.py
```

**Output**:
- Prints formatted ASCII tables for **Table 1 through Table 11** to stdout.
- Saves all quantitative metrics to `results_summary.json`.
- Generates `figure_1_heatmap.png` and `figure_5_convergence.png` inside `web_app/plots/`.

---

### 3. Train Architecture MH & SOTA Baselines
To train the proposed **Architecture MH** or state-of-the-art baselines (**ResNet-50**, **EfficientNet-B4**, **Vision Transformer**, **FedAvg-CNN**, **Hybrid CNN-ViT**) on any clinical domain, run:

```bash
python train_models.py --model "Architecture MH" --domain "Colorectal polyps" --rounds 10 --epochs 5 --lr 0.001 --batch 32 --strategy FedProx
```

#### CLI Parameters:
- `--model`: Model to train (`"Architecture MH"`, `"ResNet-50"`, `"EfficientNet-B4"`, `"Vision transformer"`, `"FedAvg-CNN"`, `"Hybrid CNN-ViT"`).
- `--domain`: Clinical domain (`"Colorectal polyps"`, `"Cervical cytology"`, `"Alzheimer’s disease"`, `"Diabetic retinopathy"`, `"Skin lesions"`).
- `--rounds`: Communication rounds (default: `10`).
- `--epochs`: Local epochs per round (default: `5`).
- `--lr`: Learning rate (default: `0.001`).
- `--batch`: Batch size (default: `32`).
- `--strategy`: FL strategy (`"FedProx"`, `"FedAvg"`, `"Centralized"`).

---

### 4. Launch the Interactive Web Dashboard Studio
To open the interactive web UI featuring real-time training controls, XAI visual heatmaps, FL simulators, and SOTA comparison radar charts:

```bash
python -m http.server 8085 --directory web_app
```

Then open your browser and navigate to:
🌐 **`http://localhost:8085`**

---

## 📊 Overview of Paper Figures & Tables

| Item | Description | Location in Code / Web App |
| :--- | :--- | :--- |
| **Figure 1** | ML Methods vs Health Conditions Heatmap | `results.py`, Web App Tab 1 |
| **Figure 2-4** | System Architecture & Explainability Workflows | `multidisease_ai/`, Web App Tab 1 |
| **Figure 5** | Convergence Curves (FedAvg vs FedProx vs Centralized) | `federated.py`, `results.py`, Web App Tab 4 |
| **Figure 6** | Alzheimer MRI XAI Matrix (Grad-CAM, Grad-CAM++, Score-CAM) | `explainability.py`, Web App Tab 5 |
| **Algorithm 1** | Federated Training & Adaptive Optimization | `federated.py`, Web App Tab 4 |
| **Table 1** | Classification Performance across 5 Domains | `calibration.py`, `results.py`, Web App Tab 1 |
| **Table 2** | Federated Optimization Strategy Benchmark | `federated.py`, `results.py`, Web App Tab 4 |
| **Table 3** | Cross-Institution External Validation (Hospitals A-E) | `federated.py`, `results.py`, Web App Tab 4 |
| **Table 4** | Explainability IoU & Clinician Agreement Metrics | `explainability.py`, `results.py`, Web App Tab 5 |
| **Table 5** | Uncertainty ECE, Brier Score & Predictive Entropy | `calibration.py`, `results.py`, Web App Tab 5 |
| **Table 6** | NAS Optimization & Computational Efficiency Analysis | `nas_engine.py`, `results.py`, Web App Tab 3 |
| **Table 7 & 9**| SOTA Baseline Comparisons & Extended PR-AUC/FROC/MCC | `architectures.py`, `results.py`, Web App Tab 3 |
| **Table 8** | Extended Domain Evaluation Metrics | `calibration.py`, `results.py`, Web App Tab 3 |
| **Table 10** | Comparison with SOTA Multi-Disease Frameworks | `calibration.py`, `results.py`, Web App Tab 3 |
| **Table 11** | Component Ablation Study | `calibration.py`, `results.py`, Web App Tab 6 |

---

## 📜 Citation & References
If you use this code or benchmark suite, please reference:
- Paper: *A Privacy-Preserving and Explainable Multi-Disease AI Framework for Clinical Decision Support* (Elsevier Preprint, 2026).
