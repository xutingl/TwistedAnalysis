"""Generate orbit_greedy_full schedules for the routcache twisted-torus cells.

For each `fixtures/routing/routcache_torus_<coords>_twisted.json` loaded routing, run
the `orbit_greedy_full` scheduler (full physical-edge accounting) and emit:

  1. fixtures/nonragged/schedule_<slice>_loaded_orbit_greedy_full_<order>.json
  2. fixtures/nonragged/cns_schedules/schedule_orbitfull_<coords>_twisted.json  (renamed copy)

Each schedule is verified for physical-edge capacity violations and its makespan
/ routing LB reported. The flatten `slice` for each cell is the torus coords with
largest dim first (verified by single-hop topology consistency, not assumed).

Reproducible: `python scripts/generate_routcache_orbitfull_schedules.py`
"""
from __future__ import annotations
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
# Avoid submodule shadowing when run from project root.
if "" in sys.path:
    sys.path.remove("")

from twisted_analysis.io.routing_table import load_routing_table
from twisted_analysis.io.schedule import (
    save_schedule,
    schedule_from_orbit_greedy_full,
)
from twisted_analysis.schedules.verify import schedule_makespan, verify_capacity
from twisted_analysis.topology import Topology

ORDER = "lpt_tail_asc"

# (routcache filename, flatten slice, cns torus-coord label)
CELLS = [
    ("routcache_torus_4x8_twisted.json", (8, 4), "4x8"),
    ("routcache_torus_8x16_twisted.json", (16, 8), "8x16"),
    ("routcache_torus_4x8x8_twisted.json", (8, 8, 4), "4x8x8"),
]

FIX = _HERE.parent / "fixtures"
ROUTING = FIX / "routing"
NONRAGGED = FIX / "nonragged"
CNS = NONRAGGED / "cns_schedules"


def routing_lb(table: list[list[list[int]]]) -> int:
    """Max load over directed physical edges = bandwidth lower bound."""
    c: Counter = Counter()
    for row in table:
        for path in row:
            for a, b in zip(path[:-1], path[1:]):
                c[(a, b)] += 1
    return max(c.values()) if c else 0


def main() -> int:
    print(f"{'cell':>10} {'N':>4} {'slice':>10} {'makespan':>9} {'LB':>4} "
          f"{'violations':>11}")
    print("-" * 56)
    for fname, slice_, coords in CELLS:
        table = load_routing_table(ROUTING / fname)
        topo = Topology(slice=slice_)
        if len(table) != topo.n_nodes:
            raise SystemExit(
                f"{fname}: table N={len(table)} != slice {slice_} "
                f"n_nodes={topo.n_nodes}"
            )

        entries = schedule_from_orbit_greedy_full(topo, table, order=ORDER)
        mk = schedule_makespan(entries)
        viol = verify_capacity(entries)
        lb = routing_lb(table)

        slice_str = "x".join(str(s) for s in slice_)
        fix_out = NONRAGGED / f"schedule_{slice_str}_loaded_orbit_greedy_full_{ORDER}.json"
        cns_out = CNS / f"schedule_orbitfull_{coords}_twisted.json"
        save_schedule(entries, fix_out)
        save_schedule(entries, cns_out)

        print(f"{coords:>10} {len(table):>4} {str(slice_):>10} {mk:>9} {lb:>4} "
              f"{len(viol):>11}")
        print(f"           -> {fix_out.relative_to(_HERE.parent)} "
              f"({fix_out.stat().st_size:,} B, {len(entries):,} entries)")
        print(f"           -> {cns_out.relative_to(_HERE.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
