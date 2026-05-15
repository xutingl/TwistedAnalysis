"""Phase 1b: random orbit-orderings on orbit_greedy_full.

Calls the internal `compute_hop0_firing_times_full` style logic but with a
shuffled orbit order. Records the best ordering found and its makespan."""
from __future__ import annotations
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from twisted_analysis.topology import Topology
from twisted_analysis.io.routing_table import load_routing_table
from twisted_analysis.io.coords import flatten
from twisted_analysis.lp.orbit import compute_orbits
from twisted_analysis.schedules.orbit_greedy_full import _orbit_hop_edge_sets
from twisted_analysis.schedules.verify import verify_capacity, schedule_makespan
from twisted_analysis.io.schedule import save_schedule

ROUTING = "fixtures/routing_table_8x4x4_twist.json"
SLICE = (8, 4, 4)
N_SEEDS = 1000
OUT = Path(__file__).parent / "02_random_orbit_shuffle_results.json"
BEST = Path(__file__).parent / "02_best_random_shuffle_schedule.json"


def fire_with_ordering(per_orbit, ordering):
    """Greedy-fire orbits in the given order; return per-orbit hop-0 time."""
    edge_busy: dict[tuple[int, int], set[int]] = defaultdict(set)
    t_hop0: dict = {}
    for orbit_id in ordering:
        hops = per_orbit[orbit_id]
        start = 0
        while True:
            ok = True
            for i, edges in enumerate(hops):
                t = start + i
                for e in edges:
                    if t in edge_busy[e]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                break
            start += 1
        for i, edges in enumerate(hops):
            for e in edges:
                edge_busy[e].add(start + i)
        t_hop0[orbit_id] = start
    return t_hop0


def assemble_schedule(t_hop0, orbits, table, slice_):
    entries = []
    for orbit_id, members in orbits.items():
        r = int(t_hop0[orbit_id])
        for (src, dst) in members:
            sf = flatten(src, slice_)
            df = flatten(dst, slice_)
            entries.append({"round": r, "src": sf, "dst": df,
                            "path": list(table[sf][df])})
    entries.sort(key=lambda e: (e["round"], e["src"]))
    return entries


def run():
    topology = Topology(slice=SLICE)
    table = load_routing_table(ROUTING)
    per_orbit = _orbit_hop_edge_sets(topology, table)
    orbit_ids = list(per_orbit.keys())
    orbits = compute_orbits(topology)

    best_makespan = None
    best_seed = None
    best_ordering = None
    history = []

    t_start = time.time()
    rng = random.Random(0)
    for seed in range(N_SEEDS):
        ordering = orbit_ids[:]
        rng.shuffle(ordering)
        t_hop0 = fire_with_ordering(per_orbit, ordering)
        sch = assemble_schedule(t_hop0, orbits, table, topology.slice)
        m = schedule_makespan(sch)
        if best_makespan is None or m < best_makespan:
            best_makespan = m
            best_seed = seed
            best_ordering = list(ordering)
            save_schedule(sch, BEST)
            print(f"  seed={seed} new best makespan={m}")
        history.append({"seed": seed, "makespan": m})

    result = {
        "best_makespan": best_makespan,
        "best_seed": best_seed,
        "n_seeds": N_SEEDS,
        "runtime_s": round(time.time() - t_start, 1),
        "histogram": _hist(history),
        "best_schedule_file": str(BEST.relative_to(BEST.parents[3])),
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(f"\nbest makespan {best_makespan} (seed {best_seed})")
    print(f"wrote {OUT}")


def _hist(history):
    counts = defaultdict(int)
    for h in history:
        counts[h["makespan"]] += 1
    return [{"makespan": m, "count": counts[m]} for m in sorted(counts)]


if __name__ == "__main__":
    run()
