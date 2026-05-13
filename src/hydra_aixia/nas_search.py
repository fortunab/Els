from __future__ import annotations

from dataclasses import dataclass
from itertools import product


@dataclass
class ArchitectureCandidate:
    layers: int
    width: int
    dropout: float


def generate_search_space(layers=(2, 3, 4), widths=(32, 64, 128), dropouts=(0.0, 0.1, 0.2)) -> list[ArchitectureCandidate]:
    return [ArchitectureCandidate(l, w, d) for l, w, d in product(layers, widths, dropouts)]


def select_best_candidate(results: dict[ArchitectureCandidate, float]) -> ArchitectureCandidate:
    return max(results, key=results.get)
