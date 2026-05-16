from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MedicalCNN(nn.Module):
    """Compact CNN baseline for multi-disease classification."""

    def __init__(self, num_classes: int, in_ch: int = 3, width: int = 32, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_ch, width, dropout),
            ConvBlock(width, width * 2, dropout),
            ConvBlock(width * 2, width * 4, dropout),
            ConvBlock(width * 4, width * 8, dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 8, width * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(width * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class BionnicaCNN(MedicalCNN):
    """Bionnica-style texture-sensitive CNN for colorectal/histopathology experiments."""

    def __init__(self, num_classes: int, in_ch: int = 3):
        super().__init__(num_classes=num_classes, in_ch=in_ch, width=40, dropout=0.25)
