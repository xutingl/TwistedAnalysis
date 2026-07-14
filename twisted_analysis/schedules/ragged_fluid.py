"""Closed-form water-filling schedule for ragged workloads.

Static rates rate_f = size_f / LB (LB = max size-weighted edge load) give
every flow one chunk active over the whole horizon: every edge carries
exactly load(e)/LB <= 1, and every flow finishes at exactly LB quanta plus
its (L-1)-quantum pipeline fill. This is makespan-optimal (no schedule can
move load(bottleneck) across a unit-capacity edge faster than LB) and
entry-count-optimal (one entry per flow) in the fluid model — the LP
relaxation of the integral one-flow-per-edge-per-step problem, which fixed
paths make solvable in closed form. See
docs/superpowers/specs/2026-07-14-ragged-a2a-scheduling-design.md.

rate_f <= 1 always holds: LB >= load on the flow's own first edge >= size_f.
"""
from __future__ import annotations

from twisted_analysis.model.ragged import RaggedWorkload


def ragged_fluid(
    table: list[list[list[int]]],
    workload: RaggedWorkload,
) -> list[dict]:
    lb = workload.lower_bound(table)
    entries: list[dict] = []
    for (s, d), size in workload.demand.items():
        entries.append({
            "round": 0,
            "src": s,
            "dst": d,
            "path": list(table[s][d]),
            "rate": size / lb,
            "size": size,
        })
    entries.sort(key=lambda e: (e["round"], e["src"], e["dst"]))
    return entries
