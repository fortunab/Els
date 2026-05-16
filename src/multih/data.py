from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms


@dataclass
class DomainSpec:
    name: str
    num_classes: int
    image_size: int = 224
    channels: int = 3


class SyntheticMedicalDataset(Dataset):
    """Small synthetic dataset for smoke tests when real datasets are unavailable."""

    def __init__(self, n: int, num_classes: int, image_size: int = 64, channels: int = 3, seed: int = 42):
        generator = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, channels, image_size, image_size, generator=generator)
        self.y = torch.randint(0, num_classes, (n,), generator=generator)
        # Add weak class-specific signal to make learning possible in smoke tests.
        for c in range(num_classes):
            self.x[self.y == c, :, : image_size // 4, : image_size // 4] += c / max(1, num_classes - 1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def build_transforms(image_size: int, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def load_image_folder(path: str | Path, image_size: int, train: bool = True) -> Dataset:
    return datasets.ImageFolder(str(path), transform=build_transforms(image_size=image_size, train=train))


def make_loaders(dataset: Dataset, batch_size: int, val_fraction: float = 0.15, test_fraction: float = 0.15, seed: int = 42):
    n = len(dataset)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    n_train = n - n_val - n_test
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(seed))
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2),
        DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2),
    )
