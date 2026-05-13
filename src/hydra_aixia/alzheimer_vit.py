from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ViTConfig:
    image_size: int = 8
    patch_size: int = 2
    channels: int = 1
    num_classes: int = 4
    dim: int = 64
    depth: int = 4
    heads: int = 4
    mlp_dim: int = 128


class TinyVisionTransformer(nn.Module):
    """Compact ViT-style classifier for small MRI image prototypes."""

    def __init__(self, config: ViTConfig = ViTConfig()) -> None:
        super().__init__()
        assert config.image_size % config.patch_size == 0
        num_patches = (config.image_size // config.patch_size) ** 2
        patch_dim = config.channels * config.patch_size * config.patch_size
        self.patch_size = config.patch_size
        self.patch_embed = nn.Linear(patch_dim, config.dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.dim, nhead=config.heads, dim_feedforward=config.mlp_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.depth)
        self.classifier = nn.Linear(config.dim, config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        p = self.patch_size
        patches = x.unfold(2, p, p).unfold(3, p, p).contiguous().view(b, c, -1, p, p)
        patches = patches.permute(0, 2, 1, 3, 4).flatten(2)
        tokens = self.patch_embed(patches)
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed
        encoded = self.encoder(tokens)
        return self.classifier(encoded[:, 0])
