from __future__ import annotations

import torch
from torch import nn
from .cnn import MedicalCNN
from .vit import MedicalViT


class HybridCNNViT(nn.Module):
    """Hybrid model combining local CNN features and global ViT features."""

    def __init__(self, num_classes: int, image_size: int = 224, in_ch: int = 3, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.cnn = MedicalCNN(num_classes=128, in_ch=in_ch, width=24, dropout=0.15)
        self.vit = MedicalViT(num_classes=128, image_size=image_size, in_ch=in_ch, embed_dim=128, depth=3, heads=4)
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_cnn = self.cnn(x)
        f_vit = self.vit(x)
        fused = self.alpha * f_cnn + (1.0 - self.alpha) * f_vit
        return self.head(fused)
