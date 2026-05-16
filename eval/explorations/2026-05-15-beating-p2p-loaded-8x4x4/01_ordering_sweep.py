"""Phase 1: deterministic orderings on orbit_greedy_full and literal_greedy.

Runs every valid (scheduler, order) combination on loaded 8x4x4 routing and
writes one JSON row per combination."""
from __future__ import annotations
import json
import time
from pathlib import Path

from twisted_analysis.topology import Topology
from twisted_analysis.io.routing_table import load_routing_table
from twisted_analysis.io.schedule import (
    schedule_from_orbit_greedy_full,
    schedule_from_literal_greedy,
)
from twisted_analysis.schedules.verify import schedule_makespan, verify_capacity

ROUTING = "fixtures/routing_table_8x4x4_twist.json"
SLICE = (8, 4, 4)
OUT = Path(__file__).with_suffix("").parent / "01_ordering_sweep_results.json"


def run():
    topology = Topology(slice=SLICE)
    table = load_routing_table(ROUTING)
    rows = []
    for order in ("lpt_tail_asc", "lpt", "spt", "tail_asc"):
        t0 = time.time()
        sch = schedule_from_orbit_greedy_full(topology, table, order=order)
        dt = time.time() - t0
        v = verify_capacity(sch)
        rows.append({
            "scheduler": "orbit_greedy_full",
            "order": order,
            "makespan": schedule_makespan(sch),
            "violations": len(v),
            "runtime_s": round(dt, 3),
            "n_flows": len(sch),
        })
    for order in ("lpt", "spt", "natural"):
        t0 = time.time()
        sch = schedule_from_literal_greedy(topology, table, order=order)
        dt = time.time() - t0
        v = verify_capacity(sch)
        rows.append({
            "scheduler": "literal_greedy",
            "order": order,
            "makespan": schedule_makespan(sch),
            "violations": len(v),
            "runtime_s": round(dt, 3),
            "n_flows": len(sch),
        })
    rows.sort(key=lambda r: r["makespan"])
    OUT.write_text(json.dumps(rows, indent=2))
    for r in rows:
        print(f"  {r['scheduler']:20s} {r['order']:14s}"
              f"  makespan={r['makespan']:3d}  viol={r['violations']:3d}"
              f"  t={r['runtime_s']:5.1f}s")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    run()
