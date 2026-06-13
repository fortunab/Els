# Federated Multi-Disease Medical AI Framework

This repository contains a reproducible PyTorch implementation scaffold for the **MultiH-EU** framework described in the manuscript: a federated, explainable, and deployment-aware multi-disease medical AI system for colorectal polyps, cervical cytology, Alzheimer MRI, diabetic retinopathy, and skin lesion classification.

The paper reports a framework that combines disease-specific CNN/ViT/hybrid architectures, preprocessing, federated learning, NAS-guided optimization, explainability, uncertainty, calibration, and cross-institution validation. This codebase provides the implementation components needed to reproduce and extend those experiments.

> The repository is dataset-agnostic. Public datasets such as Kvasir-SEG, CVC-ClinicDB, ADNI, EyePACS/APTOS, ISIC, HAM10000, Herlev, and SIPaKMeD must be downloaded separately according to their licenses.

## Main Features

- Disease-specific architectures
- Federated training simulation with FedAvg and FedProx.
- Multi-disease dataset organization with institution-aware splits.
- Metrics: Accuracy, Precision, Recall/Sensitivity, Specificity, F1, ROC-AUC, PR-AUC, MCC, ECE, Brier Score.
- Explainability: Grad-CAM for CNN/hybrid models.
- Uncertainty: Monte Carlo Dropout and predictive entropy.
- NAS-lite random search for accuracy-latency trade-off.
- Edge/deployment utilities: latency benchmarking, parameter count, FLOPs placeholder hooks.

## Repository Structure

```text
root/
├── configs/                  # YAML experiment configs
├── data/                     # dataset placeholders; no medical data included yet
├── docs/                     # paper-related notes
├── notebooks/                # example notebook placeholder
├── scripts/                  # CLI entry points
├── src/multih/               # Python package
│   ├── models/               # CNN, ViT, hybrid disease models
│   ├── training/             # centralized training and evaluation
│   ├── fl/                   # federated training strategies
│   ├── metrics/              # medical metrics and calibration
│   ├── xai/                  # Grad-CAM and interpretability
│   ├── nas/                  # NAS-lite search
│   └── utils/                # config, seed, logging, benchmark helpers
└── tests/                    # smoke tests
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Citation

If used, cite the manuscript and the original datasets/frameworks according to their licenses.
