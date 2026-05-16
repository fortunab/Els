from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def enable_dropout(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
            module.train()


@torch.no_grad()
def mc_dropout_predict(model: nn.Module, loader: DataLoader, device: str, passes: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.to(device)
    all_y = []
    all_probs_passes = []
    for _ in range(passes):
        model.eval()
        enable_dropout(model)
        probs, ys = [], []
        for x, y in loader:
            x = x.to(device)
            p = torch.softmax(model(x), dim=1).cpu().numpy()
            probs.append(p)
            ys.append(y.numpy())
        all_probs_passes.append(np.concatenate(probs))
        if not all_y:
            all_y = list(np.concatenate(ys))
    stacked = np.stack(all_probs_passes, axis=0)
    mean_probs = stacked.mean(axis=0)
    var_probs = stacked.var(axis=0).mean(axis=1)
    return np.asarray(all_y), mean_probs, var_probs
