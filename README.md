# Federated Multi-Disease Medical AI Framework

This repository contains a reproducible PyTorch implementation scaffold for the **MultiH-EU** framework described in the manuscript: a federated, explainable, and deployment-aware multi-disease medical AI system for colorectal polyps, cervical cytology, Alzheimer MRI, diabetic retinopathy, and skin lesion classification.

The paper reports a framework that combines disease-specific CNN/ViT/hybrid architectures, preprocessing, federated learning, NAS-guided optimization, explainability, uncertainty, calibration, and cross-institution validation. This codebase provides the implementation components needed to reproduce and extend those experiments.

> The repository is dataset-agnostic. Public datasets such as Kvasir-SEG, CVC-ClinicDB, ADNI, EyePACS/APTOS, ISIC, HAM10000, Herlev, and SIPaKMeD must be downloaded separately according to their licenses.

## Main Features

- Disease-specific architectures: CNN, lightweight ViT, hybrid CNN-ViT, Bionnica-style CNN, EDoViT-style ViT, EI-ViT-style retinal transformer, BOVNet/BFIS-style cytology CNN.
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

## Quick Start with Synthetic Data

Run a centralized smoke experiment:

```bash
python scripts/train_centralized.py --config configs/centralized_synthetic.yaml
```

Run federated simulation:

```bash
python scripts/train_federated.py --config configs/federated_synthetic.yaml
```

Run NAS-lite search:

```bash
python scripts/run_nas.py --config configs/nas_synthetic.yaml
```

Run evaluation metrics from saved predictions:

```bash
python scripts/evaluate_predictions.py --predictions outputs/predictions.csv
```

## Dataset Format

Expected image classification format (not yet available):

```text
data/
└── processed/
    ├── colorectal/
    │   ├── institution_a/
    │   │   ├── class_0/*.png
    │   │   └── class_1/*.png
    │   └── institution_b/...
    ├── cervical/
    ├── alzheimer/
    ├── retinopathy/
    └── skin/
```

Each disease domain can have a different number of classes. This will be set in the YAML config.

## Important Reproducibility Note

This repository includes executable training/evaluation infrastructure, but not private clinical datasets. Reported manuscript values should only be claimed if reproduced with the actual datasets, splits, and experimental protocol.

## Citation

If used, cite the manuscript and the original datasets/frameworks according to their licenses.
