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
- Edge/deployment utilities: latency benchmarking, parameter count, FLOPs.


# Els

This repository contains experimental scripts for evaluating classification performance, federated learning behavior, cross-institution generalization, explainability, calibration, NAS efficiency, state-of-the-art comparisons, framework comparisons, and ablation studies.

The project can be executed either script by script or all at once using `run_all.py`.

## Repository

```bash
git clone https://github.com/fortunab/Els.git
cd Els
```

## Requirements

Make sure you have the following installed:

* Python 3.9 or newer
* Git
* pip

## Installation with Virtual Environment

Using a virtual environment is recommended because it isolates the dependencies of this project from other Python projects on your system.

### Windows

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install --upgrade pip
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
```

### Linux/macOS

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

## Run All Experiments

To execute all available experiments using `run_all.py`, run:

### Windows

```powershell
.\\.venv\\Scripts\\python.exe src/run_all.py
```

### Linux/macOS

```bash
./.venv/bin/python src/run_all.py
```

## Run Individual Scripts

Each experiment can also be executed independently.

### Windows

```powershell
.\\.venv\\Scripts\\python.exe src/classification_performance.py
.\\.venv\\Scripts\\python.exe src/federated_performance.py
.\\.venv\\Scripts\\python.exe src/cross_institution_generalization.py
.\\.venv\\Scripts\\python.exe src/explainability_analysis.py
.\\.venv\\Scripts\\python.exe src/calibration_analysis.py
.\\.venv\\Scripts\\python.exe src/nas_efficiency.py
.\\.venv\\Scripts\\python.exe src/sota_comparison.py
.\\.venv\\Scripts\\python.exe src/extended_metrics_domains.py
.\\.venv\\Scripts\\python.exe src/extended_sota_metrics.py
.\\.venv\\Scripts\\python.exe src/framework_comparison.py
.\\.venv\\Scripts\\python.exe src/ablation_study.py
```

### Linux/macOS

```bash
./.venv/bin/python src/classification_performance.py
./.venv/bin/python src/federated_performance.py
./.venv/bin/python src/cross_institution_generalization.py
./.venv/bin/python src/explainability_analysis.py
./.venv/bin/python src/calibration_analysis.py
./.venv/bin/python src/nas_efficiency.py
./.venv/bin/python src/sota_comparison.py
./.venv/bin/python src/extended_metrics_domains.py
./.venv/bin/python src/extended_sota_metrics.py
./.venv/bin/python src/framework_comparison.py
./.venv/bin/python src/ablation_study.py
```

## Optional: Activate the Virtual Environment

Instead of calling the virtual environment Python directly every time, you can activate it.

### Windows CMD

```cmd
.venv\\Scripts\\activate.bat
```

### Windows PowerShell

```powershell
.venv\\Scripts\\Activate.ps1
```

If PowerShell blocks script execution, you can still use the direct commands shown above, such as:

```powershell
.\\.venv\\Scripts\\python.exe src/classification_performance.py
```

### Linux/macOS

```bash
source .venv/bin/activate
```

After activation, scripts can be executed simply with:

```bash
python src/classification_performance.py
python src/run_all.py
```

### Git is not recognized on Windows

Install Git for Windows, then open a new PowerShell or CMD window and verify:

```powershell
git --version
```

### Missing Python package

If you see an error such as `ModuleNotFoundError`, reinstall dependencies inside the virtual environment:

### Windows

```powershell
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
```

### Linux/macOS

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

### PowerShell script execution is disabled

You do not need to activate the environment. Use the direct virtual environment Python command instead:

```powershell
.\\.venv\\Scripts\\python.exe run_all.py
```

