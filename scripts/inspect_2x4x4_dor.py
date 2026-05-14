"""Inspect the 2x4x4 DOR edge case where OrbitGreedy LPT gives makespan=17 vs LB=16."""
from __future__ import annotations
from collections import Counter, defaultdict

from twisted_analysis.lp.orbit import compute_orbits
from twisted_analysis.model import AllToAll
from twisted_analysis.schedules.orbit_greedy import (
    OrbitGreedySchedule, PipelinedOrbitSchedule, _emit_orbit_greedy, _canonical_paths,
)
from twisted_analysis.simulator import Simulator
from twisted_analysis.topology import DORRouter, Topology


def main() -> None:
    t = Topology(slice=(2, 4, 4))
    r = DORRouter(t)
    w = AllToAll(t, r, msg_size=1)
    print(f"slice=(2,4,4) router=DOR  N={t.n_nodes} LB={w.lower_bound}")
    print()

    canon = _canonical_paths(t, r)
    orbits = compute_orbits(t)

    # Edge-orbit load (max should be LB)
    edge_orbit_load: Counter = Counter()
    for orbit_id, path in canon.items():
        for _, _, dim, dir in path:
            edge_orbit_load[(dim, dir)] += 1
    print(f"Edge-orbit loads: {dict(edge_orbit_load)}")
    bottleneck_edges = [k for k, v in edge_orbit_load.items() if v == max(edge_orbit_load.values())]
    print(f"Bottleneck edge orbits (load={max(edge_orbit_load.values())}): {bottleneck_edges}")
    print()

    # Path length distribution
    by_len = Counter(len(p) for p in canon.values())
    print(f"Path length distribution: {dict(by_len)}")
    print()

    # Run greedy and find which step is the makespan-deciding hop
    assignment = _emit_orbit_greedy(t, r, "lpt")
    # Last fire step
    max_t = max(t for (_, _, t) in assignment.keys())
    print(f"OrbitGreedy LPT makespan = max fire time + 1 = {max_t + 1}")
    # Which orbit fires last and on what hop/edge?
    last_fires = [(o, i, ti) for (o, i, ti) in assignment if ti == max_t]
    for o, i, ti in last_fires:
        path = canon[o]
        edge = (path[i][2], path[i][3])
        print(f"  Last fire: orbit={o} hop={i}/{len(path)-1} edge_orbit={edge} time={ti}")

    # Full per-edge schedule
    print("Per-edge-orbit usage (steps in use):")
    edge_steps: dict = defaultdict(list)
    for (o, i, ti), val in assignment.items():
        if val > 0.5:
            edge = (canon[o][i][2], canon[o][i][3])
            edge_steps[edge].append((ti, o, i))
    for edge, steps in sorted(edge_steps.items()):
        steps_sorted = sorted(s[0] for s in steps)
        load = edge_orbit_load[edge]
        print(f"  edge {edge} (load={load}): steps={steps_sorted}")
    print()
    # For each bottleneck edge orbit, print the schedule
    print("Per-bottleneck-edge schedule:")
    for be in bottleneck_edges:
        slots = []
        for (o, i, ti), val in assignment.items():
            if val > 0.5 and (canon[o][i][2], canon[o][i][3]) == be:
                slots.append((ti, o, i))
        slots.sort()
        print(f"  edge {be}: occupied steps = {[s[0] for s in slots]}  (max={max(s[0] for s in slots) if slots else None})")
        # Look for gaps
        steps = sorted(s[0] for s in slots)
        gaps = [steps[i+1] - steps[i] - 1 for i in range(len(steps)-1)]
        if any(g > 0 for g in gaps):
            print(f"    GAPS exist: {gaps}")
    print()

    # For each orbit, print path + schedule
    print("Orbit schedules sorted by completion time:")
    by_orbit_hop: dict = defaultdict(dict)
    for (o, i, ti), val in assignment.items():
        if val > 0.5:
            by_orbit_hop[o][i] = ti
    rows = []
    for o, hops in by_orbit_hop.items():
        path = canon[o]
        sched = [hops[i] for i in sorted(hops.keys())]
        complete = sched[-1] + 1
        path_str = "".join(f"({d},{'+' if dr>0 else '-'})" for _, _, d, dr in path)
        rows.append((complete, o, path_str, sched))
    rows.sort(key=lambda r: -r[0])
    for complete, o, path_str, sched in rows[:8]:
        print(f"  complete={complete}  orbit={o}  path={path_str}  sched={sched}")
    print()


if __name__ == "__main__":
    main()
