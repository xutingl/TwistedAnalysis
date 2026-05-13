from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol

from twisted_analysis.model.flow import Flow, AllToAll


@dataclass(frozen=True)
class Injection:
    """One unit of a flow is injected onto its first hop at `start_step`."""
    flow: Flow
    start_step: int
    priority: int = 0  # lower = higher priority at link contention; default FIFO
    hop_schedule: tuple[int, ...] = ()  # LP-derived fire step per hop; empty = use priority only


@dataclass(frozen=True)
class ScheduleResult:
    name: str
    makespan: int
    lower_bound: int
    per_step_busy: tuple[int, ...] = ()
    idle_steps_on_bottleneck: dict[tuple, int] = field(default_factory=dict)

    @property
    def ratio(self) -> float:
        return self.makespan / self.lower_bound


class Schedule(Protocol):
    name: str
    def emit(self, workload: AllToAll) -> list[Injection]: ...
