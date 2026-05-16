from __future__ import annotations

from dataclasses import dataclass
import random
import torch
from multih.models.cnn import MedicalCNN
from multih.models.hybrid import HybridCNNViT
from multih.models.vit import MedicalViT
from multih.training.engine import fit
from multih.utils.benchmark import benchmark_latency, count_parameters


@dataclass
class Candidate:
    model_name: str
    width: int | None = None
    embed_dim: int | None = None
    depth: int | None = None
    heads: int | None = None


def sample_candidate() -> Candidate:
    model_name = random.choice(["cnn", "vit", "hybrid"])
    if model_name == "cnn":
        return Candidate(model_name="cnn", width=random.choice([16, 24, 32, 40]))
    if model_name == "vit":
        return Candidate(model_name="vit", embed_dim=random.choice([96, 128, 160]), depth=random.choice([2, 3, 4]), heads=random.choice([3, 4, 5]))
    return Candidate(model_name="hybrid")


def build_candidate(c: Candidate, num_classes: int, image_size: int):
    if c.model_name == "cnn":
        return MedicalCNN(num_classes=num_classes, width=c.width or 32)
    if c.model_name == "vit":
        return MedicalViT(num_classes=num_classes, image_size=image_size, embed_dim=c.embed_dim or 128, depth=c.depth or 3, heads=c.heads or 4)
    return HybridCNNViT(num_classes=num_classes, image_size=image_size)


def random_search(train_loader, val_loader, num_classes: int, image_size: int, trials: int, epochs: int, lr: float, device: str, latency_weight: float = 0.001):
    results = []
    best = None
    best_score = float("inf")
    for _ in range(trials):
        cand = sample_candidate()
        model = build_candidate(cand, num_classes=num_classes, image_size=image_size)
        train_result = fit(model, train_loader, val_loader, epochs=epochs, lr=lr, device=device, patience=2)
        latency = benchmark_latency(model, (1, 3, image_size, image_size), device=device, runs=10)
        params = count_parameters(model)
        score = (1.0 - train_result.best_metrics["accuracy"]) + latency_weight * latency
        row = {"candidate": cand, "score": score, "latency_ms": latency, "params": params, **train_result.best_metrics}
        results.append(row)
        if score < best_score:
            best_score = score
            best = model
    return best, results
