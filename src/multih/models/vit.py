from __future__ import annotations

import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(self, in_ch: int = 3, embed_dim: int = 128, patch_size: int = 16):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class MedicalViT(nn.Module):
    """Lightweight Vision Transformer suitable for medical image classification."""

    def __init__(self, num_classes: int, image_size: int = 224, patch_size: int = 16, in_ch: int = 3, embed_dim: int = 128, depth: int = 4, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = PatchEmbedding(in_ch, embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim * 4, dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : x.shape[1]]
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)


class EDoViTAlz(MedicalViT):
    """EDoViT-Alz style model for MRI-based Alzheimer classification."""

    def __init__(self, num_classes: int, image_size: int = 224, in_ch: int = 3):
        super().__init__(num_classes=num_classes, image_size=image_size, patch_size=16, in_ch=in_ch, embed_dim=160, depth=6, heads=5, dropout=0.15)


class EIViTNet(MedicalViT):
    """EI-ViT-Net style retinal transformer."""

    def __init__(self, num_classes: int, image_size: int = 224, in_ch: int = 3):
        super().__init__(num_classes=num_classes, image_size=image_size, patch_size=16, in_ch=in_ch, embed_dim=192, depth=5, heads=6, dropout=0.1)
