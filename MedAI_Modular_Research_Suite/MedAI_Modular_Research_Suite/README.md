"""R-ULIx barebone PyTorch model."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectNormalize(nn.Module):
    def __init__(self, in_dim, shared_dim, use_norm=True):
        super().__init__()
        self.proj = nn.Linear(in_dim, shared_dim)
        self.norm = nn.LayerNorm(shared_dim) if use_norm else nn.Identity()

    def forward(self, x):
        return self.norm(self.proj(x))


class Router(nn.Module):
    def __init__(self, dim, experts, topk=2):
        super().__init__()
        self.gate = nn.Linear(dim, experts)
        self.topk = topk

    def forward(self, x):
        probs = F.softmax(self.gate(x), dim=-1)
        weights, indices = torch.topk(probs, k=min(self.topk, probs.shape[-1]), dim=-1)
        return weights, indices


class LatentExperts(nn.Module):
    def __init__(self, num_experts, dim, heads=4):
        super().__init__()
        self.experts = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True) for _ in range(num_experts)]
        )

    def forward(self, tokens, topk_idx):
        outputs = [expert(tokens) for expert in self.experts]
        return torch.stack(outputs, dim=0).mean(dim=0)


class Reasoner(nn.Module):
    def __init__(self, dim, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_classes),
        )

    def forward(self, x):
        return self.net(x.mean(dim=1))


class R_ULIx(nn.Module):
    def __init__(self, in_dim=128, shared_dim=256, num_experts=4, topk=2, num_classes=2):
        super().__init__()
        self.project = ProjectNormalize(in_dim, shared_dim)
        self.router = Router(shared_dim, num_experts, topk=topk)
        self.experts = LatentExperts(num_experts, shared_dim)
        self.reasoner = Reasoner(shared_dim, num_classes=num_classes)

    def forward(self, x):
        tokens = self.project(x)
        _, idx = self.router(tokens)
        fused = self.experts(tokens, idx)
        return self.reasoner(fused)


def build_rulix_demo():
    return R_ULIx(in_dim=128, shared_dim=256, num_experts=4, topk=2, num_classes=2)
