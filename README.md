# Hydra AIXIA Extended

Hydra AIXIA extended is a research-oriented collection of machine-learning experiments for medical image analysis, federated-learning prototypes, multimodal ML/LM fusion, and resource-aware neural-network inference scheduling.

The repository was reorganized from standalone Colab-style Python scripts into a cleaner GitHub-ready layout with reusable modules, examples, documentation, and safe configuration practices.

## Repository structure

```text
hydra_aixia/
├── src/hydra_aixia/          # Reusable Python package
│   ├── alzheimer_vit.py      # Alzheimer MRI ViT prototype utilities
│   ├── colorectal_histology.py
│   ├── medfusion_lmml.py     # ML/LM fusion and Streamlit-style helpers
│   ├── mnist_resnet_cv.py    # ResNet50 MNIST cross-validation experiment
│   ├── scheduler_dvfs.py     # Hybrid EDF + DVFS scheduler prototype
│   ├── masked_autoencoder.py
│   ├── nas_search.py
│   └── zs_n2n.py
├── examples/                 # Runnable example entry points
├── docs/                     # Notes and migration guide
├── tests/                    # Basic import tests
├── requirements.txt
├── pyproject.toml
├── .env.example
└── .gitignore
```

## Installation

```bash
git clone https://github.com/<your-user>/hydra_aixia.git
cd hydra_aixia
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Configuration

Never commit API keys. Copy the example environment file and set local values only on your machine:

```bash
cp .env.example .env
```

## Example usage

Run the MNIST ResNet cross-validation experiment:

```bash
python examples/run_mnist_resnet.py
```

Run the colorectal histology example:

```bash
python examples/run_colorectal_histology.py
```

Run the EDF + DVFS scheduler demo:

```bash
python examples/run_scheduler_demo.py
```

## Original scripts mapped to the new structure

| Original file | New location |
|---|---|
| `alzheimer_mri_8x8_vit_ver2_(1).py` | `src/hydra_aixia/alzheimer_vit.py` |
| `colorectal (1).py` | `src/hydra_aixia/colorectal_histology.py` |
| `experiment.py` | `src/hydra_aixia/mnist_resnet_cv.py` |
| `experiment_lmml_(1).py` | `src/hydra_aixia/medfusion_lmml.py` |
| `implementation_hybridedfschedulerwdvfs_onfcn.py` | `src/hydra_aixia/scheduler_dvfs.py` |
| `ml_pipeline (1).py` | `src/hydra_aixia/medfusion_lmml.py` |
| `themaskedautoencoders (2).py` | `src/hydra_aixia/masked_autoencoder.py` |
| `v_nas_this.py` | `src/hydra_aixia/nas_search.py` |
| `zs_n2n.py` | `src/hydra_aixia/zs_n2n.py` |

## Safety and reproducibility notes

- Hard-coded credentials were removed.
- Local data, trained models, and generated artifacts are ignored by Git.
- Medical predictions produced by these scripts are experimental and must not be used as clinical advice.

## License

Add the license that fits your publication or project requirements.
