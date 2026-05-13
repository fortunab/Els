from __future__ import annotations

import torch
from torch import nn


class SimpleMaskedAutoencoder(nn.Module):
    """Minimal masked autoencoder prototype for image-like tensors."""

    def __init__(self, input_dim: int, latent_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, latent_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(latent_dim, input_dim), nn.Sigmoid())

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.25) -> torch.Tensor:
        mask = torch.rand_like(x) > mask_ratio
        z = self.encoder(x * mask)
        return self.decoder(z)
