"""Ragged (per-pair-sized) AllToAll workload over flat node IDs.

Unlike `twisted_analysis.model.flow.AllToAll` (coordinate-based, uniform
msg_size), a ragged workload arrives as flat-ID (src, dst) -> size pairs
loaded from JSON, and its paths come from an already-loaded routing table
(`table[src][dst] = [flat-id, ...]`).

Sizes are in workload units (bytes for the shipped fixtures). The quantum
is the gcd of all sizes; schedule time is measured in quanta. See
docs/superpowers/specs/2026-07-14-ragged-a2a-scheduling-design.md.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from math import gcd
from types import MappingProxyType
from typing import Mapping

Edge = tuple[int, int]


@dataclass(frozen=True)
class RaggedWorkload:
    demand: Mapping[tuple[int, int], int]  # (src, dst) -> size > 0

    def __post_init__(self) -> None:
        if not self.demand:
            raise ValueError("workload must contain at least one flow")
        for (s, d), size in self.demand.items():
            if s == d:
                raise ValueError(f"self-pair ({s}, {d}) not allowed")
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                raise ValueError(
                    f"flow ({s}, {d}): size must be positive int, got {size!r}"
                )
        object.__setattr__(self, "demand", MappingProxyType(dict(self.demand)))

    @cached_property
    def quantum(self) -> int:
        q = 0
        for size in self.demand.values():
            q = gcd(q, size)
        return q

    def link_load(self, table: list[list[list[int]]]) -> dict[Edge, int]:
        """Size-weighted directed-edge load (same convention as AllToAll)."""
        c: Counter[Edge] = Counter()
        for (s, d), size in self.demand.items():
            path = table[s][d]
            for u, v in zip(path, path[1:]):
                c[(u, v)] += size
        return dict(c)

    def lower_bound(self, table: list[list[list[int]]]) -> int:
        """Max size-weighted edge load: hard makespan LB in workload units."""
        return max(self.link_load(table).values())

    def bottleneck_edges(self, table: list[list[list[int]]]) -> list[Edge]:
        loads = self.link_load(table)
        lb = max(loads.values())
        return [e for e, load in loads.items() if load == lb]
