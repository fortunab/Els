# Manuscript-to-Code Mapping

This repository implements the main computational modules described in the MultiH-EU manuscript.

| Manuscript component | Code location |
|---|---|
| Disease-specific CNN/ViT/hybrid models | `src/multih/models/` |
| Federated optimization | `src/multih/fl/strategies.py` |
| Evaluation metrics: ROC-AUC, PR-AUC, MCC, ECE | `src/multih/metrics/classification.py` |
| FROC utility | `src/multih/metrics/froc.py` |
| Explainability / Grad-CAM | `src/multih/xai/gradcam.py` |
| Uncertainty / MC Dropout | `src/multih/training/uncertainty.py` |
| NAS-lite search | `src/multih/nas/search.py` |
| Centralized training | `scripts/train_centralized.py` |
| Federated training simulation | `scripts/train_federated.py` |

## Notes

The manuscript includes multiple public and institutional datasets. Because data licenses differ and private clinical data cannot be redistributed, this repository provides synthetic data configs for smoke testing and expects users to place real datasets under `data/processed/`.
