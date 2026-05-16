from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import torch
from torch import nn
from torch.utils.data import DataLoader
from multih.training.engine import train_one_epoch, predict
from multih.metrics.classification import compute_metrics


StateDict = OrderedDict[str, torch.Tensor]


@dataclass
class Client:
    name: str
    train_loader: DataLoader
    val_loader: DataLoader
    num_samples: int


def get_state(model: nn.Module) -> StateDict:
    return OrderedDict((k, v.detach().cpu().clone()) for k, v in model.state_dict().items())


def set_state(model: nn.Module, state: StateDict) -> None:
    model.load_state_dict(state, strict=True)


def weighted_average(states: list[StateDict], weights: list[float]) -> StateDict:
    out = OrderedDict()
    total = sum(weights)
    for key in states[0].keys():
        out[key] = sum(state[key] * (w / total) for state, w in zip(states, weights))
    return out


def local_train(model: nn.Module, global_state: StateDict, client: Client, epochs: int, lr: float, device: str, mu: float = 0.0) -> StateDict:
    set_state(model, global_state)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    global_params = [p.detach().clone() for p in model.parameters()]
    for _ in range(epochs):
        if mu <= 0:
            train_one_epoch(model, client.train_loader, optimizer, criterion, device)
        else:
            model.train()
            for x, y in client.train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                prox = 0.0
                for p, gp in zip(model.parameters(), global_params):
                    prox = prox + torch.sum((p - gp) ** 2)
                loss = loss + (mu / 2.0) * prox
                loss.backward()
                optimizer.step()
    return get_state(model)


def run_federated(model_fn, clients: list[Client], rounds: int, local_epochs: int, lr: float, device: str, strategy: str = "fedavg", mu: float = 0.01):
    global_model = model_fn()
    global_state = get_state(global_model)
    history = []
    for r in range(1, rounds + 1):
        states, weights = [], []
        for client in clients:
            local_model = model_fn()
            state = local_train(local_model, global_state, client, local_epochs, lr, device, mu=mu if strategy.lower() == "fedprox" else 0.0)
            states.append(state)
            weights.append(float(client.num_samples))
        global_state = weighted_average(states, weights)
        set_state(global_model, global_state)
        # Validation over all clients.
        metrics_per_client = []
        for client in clients:
            y_true, probs = predict(global_model.to(device), client.val_loader, device)
            metrics_per_client.append(compute_metrics(y_true, probs))
        mean_metrics = {k: float(sum(m[k] for m in metrics_per_client) / len(metrics_per_client)) for k in metrics_per_client[0]}
        mean_metrics["round"] = r
        history.append(mean_metrics)
    return global_model, history
