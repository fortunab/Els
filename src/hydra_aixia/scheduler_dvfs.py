from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(order=True)
class Task:
    deadline: float
    task_id: str = field(compare=False)
    workload: float = field(compare=False)
    energy_cost: float = field(default=1.0, compare=False)


@dataclass
class ScheduleDecision:
    task_id: str
    frequency: float
    estimated_runtime: float
    estimated_energy: float


def choose_frequency(task: Task, current_time: float, min_freq: float = 0.5, max_freq: float = 1.0) -> float:
    slack = max(task.deadline - current_time, 1e-6)
    required = task.workload / slack
    return min(max(required, min_freq), max_freq)


def hybrid_edf_dvfs_schedule(tasks: Iterable[Task], current_time: float = 0.0) -> list[ScheduleDecision]:
    ordered = sorted(tasks)
    decisions: list[ScheduleDecision] = []
    t = current_time
    for task in ordered:
        freq = choose_frequency(task, t)
        runtime = task.workload / freq
        energy = task.energy_cost * (freq ** 2) * runtime
        decisions.append(ScheduleDecision(task.task_id, freq, runtime, energy))
        t += runtime
    return decisions
