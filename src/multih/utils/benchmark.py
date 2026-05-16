from __future__ import annotations

import time
import torch
from torch import nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def benchmark_latency(model: nn.Module, input_shape: tuple[int, ...], device: str = "cpu", warmup: int = 10, runs: int = 50) -> float:
    model = model.to(device).eval()
    x = torch.randn(*input_shape, device=device)
    for _ in range(warmup):
        _ = model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        _ = model(x)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (elapsed / runs) * 1000.0
