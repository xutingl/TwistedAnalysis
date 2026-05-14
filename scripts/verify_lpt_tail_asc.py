"""Verify that the `lpt+tail_asc` ordering achieves LB on every cell."""
from __future__ import annotations
from collections import defaultdict, Counter

from twisted_analysis.model import AllToAll
from twisted_analysis.schedules.lp_symmetric import symmetric_assignment_to_injections
from twisted_analysis.schedules.orbit_greedy import _canonical_paths
from twisted_analysis.simulator import Simulator
from twisted_analysis.topology import DORRouter, ILPRouter, Topology


def emit_greedy_with_order(t, r, ordered, canon):
    edge_busy: dict = defaultdict(set)
    a: dict = {}
    for orbit_id in ordered:
        path = canon[orbit_id]
        prev_t = -1
        for i, (_, _, dim, dir) in enumerate(path):
            tt = prev_t + 1
            while tt in edge_busy[(dim, dir)]:
                tt += 1
            a[(orbit_id, i, tt)] = 1.0
            edge_busy[(dim, dir)].add(tt)
            prev_t = tt
    return a


def main() -> None:
    for slice_ in [(2,4), (2,2,4), (2,4,4), (4,8), (4,4,8)]:
        t = Topology(slice=slice_)
        for r_name in ['dor', 'ilp']:
            r = DORRouter(t) if r_name == 'dor' else ILPRouter(t)
            w = AllToAll(t, r, msg_size=1)
            lb = w.lower_bound
            canon = _canonical_paths(t, r)
            edge_load = Counter()
            for o, p in canon.items():
                for _, _, d, dr in p:
                    edge_load[(d, dr)] += 1

            def tail_load(o): return edge_load[(canon[o][-1][2], canon[o][-1][3])]
            ordered = sorted(canon.keys(), key=lambda o: (-len(canon[o]), tail_load(o), o))

            a = emit_greedy_with_order(t, r, ordered, canon)
            injs = symmetric_assignment_to_injections(t, r, list(w.flows), a)
            sim = Simulator(t, r, list(w.flows))
            for inj in injs:
                sim.inject(inj)
            real_m = sim.run()
            mark = "✓" if real_m == lb else "✗"
            print(f"  {slice_!s:<10} {r_name:<3} LB={lb:>3} lpt+tail_asc → {real_m:>3} ratio={real_m/lb:.3f} {mark}")


if __name__ == "__main__":
    main()
