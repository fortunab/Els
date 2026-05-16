from __future__ import annotations

from torch import nn
from .cnn import MedicalCNN, BionnicaCNN
from .vit import MedicalViT, EDoViTAlz, EIViTNet
from .hybrid import HybridCNNViT


def build_model(name: str, num_classes: int, image_size: int = 224, in_ch: int = 3) -> nn.Module:
    name = name.lower()
    if name in {"cnn", "medicalcnn", "bfis", "bovnet"}:
        return MedicalCNN(num_classes=num_classes, in_ch=in_ch)
    if name in {"bionnica"}:
        return BionnicaCNN(num_classes=num_classes, in_ch=in_ch)
    if name in {"vit", "vision_transformer"}:
        return MedicalViT(num_classes=num_classes, image_size=image_size, in_ch=in_ch)
    if name in {"edovit", "edovit_alz", "edovit-alz"}:
        return EDoViTAlz(num_classes=num_classes, image_size=image_size, in_ch=in_ch)
    if name in {"eivit", "ei-vit-net", "eivitnet"}:
        return EIViTNet(num_classes=num_classes, image_size=image_size, in_ch=in_ch)
    if name in {"hybrid", "hybrid_cnn_vit", "cnn_vit"}:
        return HybridCNNViT(num_classes=num_classes, image_size=image_size, in_ch=in_ch)
    raise ValueError(f"Unknown model name: {name}")
