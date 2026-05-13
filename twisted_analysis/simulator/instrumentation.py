from __future__ import annotations
import csv
from pathlib import Path

from twisted_analysis.topology import DirectedLink
from twisted_analysis.simulator.engine import Simulator


def collect_idle_trace(
    sim: Simulator, bottleneck_edges: list[DirectedLink]
) -> dict[DirectedLink, int]:
    """For each bottleneck edge, count steps where the edge was idle but
    other links were still working (i.e. work was not finished)."""
    if not sim.record_history:
        raise ValueError("Simulator must be constructed with record_history=True")
    result: dict[DirectedLink, int] = {}
    total_steps = len(sim.busy_per_step)
    for e in bottleneck_edges:
        busy = sim.link_busy.get(e, [False] * total_steps)
        idle = sum(1 for b in busy if not b)
        result[e] = idle
    return result


def gantt_log(sim: Simulator) -> list[tuple[int, DirectedLink, str, int]]:
    """Return Gantt rows: (step, link, flow_repr, hop_index)."""
    return [(t, link, repr(flow), hop) for t, link, flow, hop in sim.history]


def write_gantt_csv(sim: Simulator, path: Path) -> None:
    rows = gantt_log(sim)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "link", "flow", "hop_index"])
        for r in rows:
            w.writerow(r)
