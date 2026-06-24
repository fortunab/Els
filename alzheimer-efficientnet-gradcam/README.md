# Alzheimer's MRI Classification with EfficientNetB0 and Eigen-CAM/Grad-CAM Style Visualization

A GitHub-ready deep learning project for four-class Alzheimer's MRI classification using EfficientNetB0, plus visual explanation maps using an Eigen-CAM attention approach from the final convolutional feature maps.

> **Dataset used:** [Best Alzheimer's MRI Dataset 99% Accuracy](https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy)

## Project structure

```text
alzheimer-efficientnet-gradcam/
├── configs/
│   └── config.yaml
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explain.py
│   └── utils.py
├── scripts/
│   └── download_dataset.py
├── notebooks/
│   └── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## 1. Create environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## 2. Download the dataset

### Option A: KaggleHub

```bash
python scripts/download_dataset.py
```

This downloads the Kaggle dataset and copies/symlinks it into the project `data/` directory when possible.

### Option B: Kaggle website

1. Open the dataset page: <https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy>
2. Download and unzip it.
3. Put the dataset into this structure:

```text
data/
├── train/
│   ├── Mild Impairment/
│   ├── Moderate Impairment/
│   ├── No Impairment/
│   └── Very Mild Impairment/
└── test/
    ├── Mild Impairment/
    ├── Moderate Impairment/
    ├── No Impairment/
    └── Very Mild Impairment/
```

If your extracted folder uses different class folder names, keep them consistent between `train/` and `test/`. The code infers class names automatically.

## 3. Configure paths

Edit `configs/config.yaml` if needed:

```yaml
data_dir: data
train_dir: data/train
test_dir: data/test
model_path: outputs/models/alzheimers_efficientnet_model.keras
```

No Google Drive or Colab path is hardcoded. For Colab, mount Drive and set these paths to your Drive folder.

## 4. Train the classifier

```bash
python -m src.train --config configs/config.yaml
```

The model is EfficientNetB0 with ImageNet weights, partial fine-tuning from `block5a_expand_activation`, dense classification layers, dropout, batch normalization, early stopping, and learning-rate reduction.

## 5. Evaluate the model

```bash
python -m src.evaluate --config configs/config.yaml
```

Outputs include:

- classification report
- confusion matrix image
- ROC-AUC curve image
- metrics JSON

Saved under `outputs/reports/`.

## 6. Generate Eigen-CAM / attention overlays

```bash
python -m src.explain --config configs/config.yaml --images-per-class 3
```

This creates original image, heatmap, and overlay panels for sampled test images under:

```text
outputs/gradcam/eigen_cam_grid.png
```

## Notes

- This repository is for research and education, not clinical diagnosis.
- For reproducible experiments, use the same dataset split, seed, TensorFlow version, and hardware settings.
- If you publish results, cite the dataset source and report the exact split and preprocessing used.
