from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from multih.metrics.classification import compute_metrics


@dataclass
class TrainResult:
    history: list[dict[str, float]]
    best_metrics: dict[str, float]


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: str) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
    return total_loss / max(1, total)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, probs = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        p = torch.softmax(logits, dim=1).cpu().numpy()
        probs.append(p)
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)


def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int, lr: float, device: str, patience: int = 5) -> TrainResult:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_f1 = -1.0
    best_state = None
    history: list[dict[str, float]] = []
    stale = 0
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        y_true, probs = predict(model, val_loader, device)
        metrics = compute_metrics(y_true, probs)
        metrics["loss"] = loss
        metrics["epoch"] = epoch
        history.append(metrics)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    y_true, probs = predict(model, val_loader, device)
    return TrainResult(history=history, best_metrics=compute_metrics(y_true, probs))
