# MedAI Modular Research Suite

This repository converts several experimental notebooks into a structured Python project for GitHub.

The project contains runnable components for:

- CRC Mixture-of-Experts with zero-cost expert pruning
- aaSAG: Saliency-Aligned Aggregation
- Zero-cost NAS / activation-diversity search
- Semantic monitorable histopathology framework
- R-ULIx barebone latent routing model

The original notebooks are preserved in the `notebooks/` folder.

---

## Repository structure

```text
MedAI_Modular_Research_Suite/
├── README.md
├── requirements.txt
├── environment.yml
├── LICENSE
├── CITATION.cff
├── configs/
│   └── default.yaml
├── src/
│   └── medai_suite/
│       ├── __init__.py
│       ├── aasag.py
│       ├── crc_moe.py
│       ├── rulix.py
│       ├── semantic_framework.py
│       ├── utils.py
│       └── zero_nas.py
├── scripts/
│   ├── run_aasag.py
│   ├── run_crc_moe.py
│   ├── run_zero_nas.py
│   ├── run_semantic_demo.py
│   ├── run_rulix_demo.py
│   └── run_all.py
├── notebooks/
├── docs/
└── outputs/
```

---

## Required Python version

Recommended:

```text
Python 3.10
```

Why Python 3.10?

TensorFlow and TensorFlow Datasets are sensitive to Python and protobuf versions. Python 3.10 provides a stable environment for the TensorFlow stack used in these experiments.

---

## Option A: Virtual environment with venv

### 1. Create a virtual environment

```bash
python -m venv .venv
```

This creates a project-specific environment and avoids interacting with the current system Python installation.

### 2. Activate the virtual environment

Windows PowerShell:

```bash
.venv/Scripts/Activate.ps1
```

Windows CMD:

```bash
.venv/Scripts/activate.bat
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Add/install packages

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

This installs all packages needed by the scripts.

### 4. Execute the required commands

Run the CRC MoE zero-cost pruning experiment:

```bash
python scripts/run_crc_moe.py --epochs 3
```

Run aaSAG:

```bash
python scripts/run_aasag.py --epochs 1
```

Run Zero-NAS:

```bash
python scripts/run_zero_nas.py --n-architectures 20
```

Run semantic framework demo:

```bash
python scripts/run_semantic_demo.py
```

Run R-ULIx barebone demo:

```bash
python scripts/run_rulix_demo.py
```

Run a light combined demo:

```bash
python scripts/run_all.py
```

### 5. Deactivate the environment

```bash
deactivate
```

This returns the terminal to the global Python environment.

---

## Option B: Conda environment

The mechanism is similar: create a separate environment, activate it, install packages, run commands, then deactivate it.

### 1. Create the environment

```bash
conda create -n medai-suite python=3.10 -y
```

or use:

```bash
conda env create -f environment.yml
```

### 2. Activate the environment

```bash
conda activate medai-suite
```

### 3. Install packages

```bash
pip install -r requirements.txt
```

### 4. Execute commands

Example:

```bash
python scripts/run_crc_moe.py --epochs 3
```

### 5. Deactivate

```bash
conda deactivate
```

---

## Components

### 1. CRC MoE with zero-cost pruning

Script:

```bash
python scripts/run_crc_moe.py --epochs 3
```

What it does:

- loads `colorectal_histology` from TensorFlow Datasets
- builds a pool of candidate experts
- scores experts using a zero-cost gradient-norm proxy
- keeps the top experts
- trains a sparse MoE classifier

Output:

```text
outputs/models/crc_moe.keras
outputs/results/crc_moe_results.json
```

---

### 2. aaSAG: Saliency-Aligned Aggregation

Script:

```bash
python scripts/run_aasag.py --epochs 1
```

What it does:

- trains small MLP/CNN/ConvMixer-lite models on MNIST
- computes gradient saliency maps
- aligns model predictions using saliency agreement
- produces saliency-based aggregation weights

Output:

```text
outputs/results/aasag_results.json
```

---

### 3. Zero-cost NAS

Script:

```bash
python scripts/run_zero_nas.py --n-architectures 20
```

What it does:

- samples candidate CNN architectures
- computes activation diversity without full training
- ranks candidates using a zero-cost score
- saves the best architecture

Output:

```text
outputs/models/zero_nas_best.keras
outputs/results/zero_nas_results.json
```

---

### 4. Semantic monitorable framework

Script:

```bash
python scripts/run_semantic_demo.py
```

What it does:

- creates architecture-decision records
- ranks architecture candidates
- saves a monitorable semantic JSON file

Output:

```text
outputs/results/semantic_demo.json
```

---

### 5. R-ULIx barebone

Script:

```bash
python scripts/run_rulix_demo.py
```

What it does:

- builds a lightweight latent routing model
- tests forward inference on dummy tokens
- verifies model shape and execution

---

## Minimal reproducibility workflow

Windows PowerShell:

```bash
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python scripts/run_zero_nas.py --n-architectures 20
python scripts/run_semantic_demo.py
python scripts/run_rulix_demo.py
deactivate
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python scripts/run_zero_nas.py --n-architectures 20
python scripts/run_semantic_demo.py
python scripts/run_rulix_demo.py
deactivate
```

---

## Notes

The TensorFlow scripts may download datasets automatically through TensorFlow Datasets. Internet access is required the first time.

For GPU acceleration, install the TensorFlow/PyTorch versions that match your CUDA installation.
