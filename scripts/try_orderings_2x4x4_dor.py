"""Try alternative orderings of orbits for the 2x4x4 DOR greedy.

Goal: find any orbit ordering / scheduling strategy that achieves makespan = LB = 16.
"""
from __future__ import annotations
import random
from collections import defaultdict, Counter

from twisted_analysis.lp.orbit import compute_orbits
from twisted_analysis.model import AllToAll
from twisted_analysis.schedules.lp_symmetric import symmetric_assignment_to_injections
from twisted_analysis.schedules.orbit_greedy import _canonical_paths
from twisted_analysis.simulator import Simulator
from twisted_analysis.topology import DORRouter, Topology


def emit_greedy_with_order(t, r, ordered_orbit_ids, canon) -> dict:
    edge_busy: dict = defaultdict(set)
    assignment: dict = {}
    for orbit_id in ordered_orbit_ids:
        path = canon[orbit_id]
        prev_t = -1
        for i, (_, _, dim, dir) in enumerate(path):
            tt = prev_t + 1
            while tt in edge_busy[(dim, dir)]:
                tt += 1
            assignment[(orbit_id, i, tt)] = 1.0
            edge_busy[(dim, dir)].add(tt)
            prev_t = tt
    return assignment


def makespan(assignment) -> int:
    if not assignment: return 0
    return max(tt for (_, _, tt) in assignment) + 1


def measure(t, r, w, ordered, canon):
    assignment = emit_greedy_with_order(t, r, ordered, canon)
    m = makespan(assignment)
    # Verify via simulator
    injs = symmetric_assignment_to_injections(t, r, list(w.flows), assignment)
    sim = Simulator(t, r, list(w.flows))
    for inj in injs:
        sim.inject(inj)
    real_m = sim.run()
    return m, real_m


def main() -> None:
    t = Topology(slice=(2, 4, 4))
    r = DORRouter(t)
    w = AllToAll(t, r, msg_size=1)
    lb = w.lower_bound
    canon = _canonical_paths(t, r)
    orbits = list(canon.keys())
    edge_load = Counter()
    for o, p in canon.items():
        for _, _, d, dr in p:
            edge_load[(d, dr)] += 1
    print(f"LB = {lb}")

    # Try various deterministic orderings
    strategies = []

    # LPT (current default)
    strategies.append(("lpt", sorted(orbits, key=lambda o: (-len(canon[o]), o))))
    # SPT
    strategies.append(("spt", sorted(orbits, key=lambda o: (len(canon[o]), o))))
    # By total edge-load along path, desc (heaviest-path first)
    def path_load(o): return sum(edge_load[(p[2], p[3])] for p in canon[o])
    strategies.append(("path_load_desc", sorted(orbits, key=lambda o: (-path_load(o), o))))
    strategies.append(("path_load_asc", sorted(orbits, key=lambda o: (path_load(o), o))))
    # By max edge-load along path (bottleneck-first), desc
    def max_edge_load(o): return max(edge_load[(p[2], p[3])] for p in canon[o])
    strategies.append(("max_edge_desc", sorted(orbits, key=lambda o: (-max_edge_load(o), -len(canon[o]), o))))
    # By tail edge load, asc (most-constrained tail first)
    def tail_load(o): return edge_load[(canon[o][-1][2], canon[o][-1][3])]
    strategies.append(("tail_load_asc", sorted(orbits, key=lambda o: (tail_load(o), -len(canon[o]), o))))
    # By tail edge load, desc
    strategies.append(("tail_load_desc", sorted(orbits, key=lambda o: (-tail_load(o), -len(canon[o]), o))))
    # LPT, but break ties by tail_load asc
    strategies.append(("lpt+tail_asc", sorted(orbits, key=lambda o: (-len(canon[o]), tail_load(o), o))))
    # LPT, but break ties by tail_load desc
    strategies.append(("lpt+tail_desc", sorted(orbits, key=lambda o: (-len(canon[o]), -tail_load(o), o))))

    print(f"\nDeterministic orderings:")
    best = ("", lb + 100)
    for name, ordered in strategies:
        m, real = measure(t, r, w, ordered, canon)
        marker = " ✓" if real == lb else ""
        print(f"  {name:<22} greedy_makespan={m:>3}  simulator={real:>3}{marker}")
        if real < best[1]:
            best = (name, real)

    print(f"\n>>> Best deterministic: {best[0]} → makespan={best[1]} (LB={lb})")

    # Random shuffles
    print(f"\nRandom shuffle (1000 seeds):")
    rng = random.Random(0)
    best_rand = lb + 100
    best_seed = None
    for seed in range(1000):
        r0 = random.Random(seed)
        ordered = list(orbits)
        r0.shuffle(ordered)
        m, real = measure(t, r, w, ordered, canon)
        if real < best_rand:
            best_rand = real
            best_seed = seed
            if real == lb:
                print(f"  seed={seed}: makespan={real}  ✓ LB-tight")
                break
    print(f"\n>>> Best random: seed={best_seed} → makespan={best_rand} (LB={lb})")


if __name__ == "__main__":
    main()
