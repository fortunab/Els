from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet50


class ResNet50MNIST(nn.Module):
    """ResNet-50 adapted for single-channel MNIST classification."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.resnet = resnet50(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)


def mnist_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])


def train_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: str = "cpu") -> None:
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()


def evaluate(model: nn.Module, loader: DataLoader, device: str = "cpu") -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predicted = torch.argmax(model(images), dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total if total else 0.0


def run_cross_validation(data_root: str = "./data", folds: int = 10, epochs: int = 5, batch_size: int = 32, device: str | None = None) -> float:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=mnist_transform())
    kfold = KFold(n_splits=folds, shuffle=True, random_state=42)
    scores: list[float] = []

    for train_idx, test_idx in kfold.split(dataset):
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)
        model = ResNet50MNIST().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(epochs):
            train_epoch(model, train_loader, criterion, optimizer, device)
        scores.append(evaluate(model, test_loader, device))

    return float(np.mean(scores))
