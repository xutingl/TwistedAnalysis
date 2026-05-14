"""Mechanically verify Smith's deadline-feasibility (‡) on every (topology, router)
cell by enumerating canonical paths.

For each edge orbit class e, list the demands (orbit, hop) on e with their
deadlines `LB - L(orbit) + hop`. Sort. Verify `d_k >= k - 1` for every k.

This is the concrete proof technique of §4.3.16 in docs/orbit_greedy_optimality.md,
applied at scale. If this script reports OK on every cell tested in the experiment
matrix, then Theorem 4.3.6 (makespan = LB) is proven on those cells *with no further
combinatorial argument needed*.
"""
from __future__ import annotations
from collections import defaultdict

from twisted_analysis.model import AllToAll
from twisted_analysis.schedules.orbit_greedy import _canonical_paths
from twisted_analysis.topology import DORRouter, ILPRouter, Topology


def verify_smith(slice_, router_name, verbose=False) -> bool:
    t = Topology(slice=slice_)
    r = DORRouter(t) if router_name == "dor" else ILPRouter(t)
    w = AllToAll(t, r, msg_size=1)
    LB = w.lower_bound
    canon = _canonical_paths(t, r)

    demands_by_edge: dict = defaultdict(list)  # e -> [(deadline, orbit, hop)]
    for o, path in canon.items():
        for i, (_, _, dim, dir_) in enumerate(path):
            deadline = LB - len(path) + i
            demands_by_edge[(dim, dir_)].append((deadline, o, i))

    overall_ok = True
    for e, demands in sorted(demands_by_edge.items()):
        demands.sort()  # by deadline
        for k, (d_k, o, i) in enumerate(demands, start=1):
            if d_k < k - 1:
                print(f"  SMITH FAIL: edge {e}, demand #{k}: d_{k}={d_k} < {k-1} (orbit={o}, hop={i})")
                overall_ok = False
        if verbose:
            ds = [d for d, _, _ in demands]
            print(f"  edge {e:<14} load={len(ds):>3} deadlines={ds}")
    return overall_ok


def main() -> None:
    print(f"{'topology':<10} {'router':<6} {'LB':>3}  Smith (‡)")
    print("-" * 40)
    all_ok = True
    for slice_ in [(2, 4), (2, 2, 4), (2, 4, 4), (4, 8), (4, 4, 8)]:
        for rn in ["dor", "ilp"]:
            t = Topology(slice=slice_)
            r = DORRouter(t) if rn == "dor" else ILPRouter(t)
            w = AllToAll(t, r, msg_size=1)
            LB = w.lower_bound
            ok = verify_smith(slice_, rn)
            mark = "✓ PROVED" if ok else "✗ FAILED"
            print(f"{str(slice_):<10} {rn:<6} {LB:>3}  {mark}")
            if not ok:
                all_ok = False

    print()
    if all_ok:
        print("All cells satisfy Smith's deadline-feasibility (‡).")
        print("By Theorem 4.3.6, makespan = LB is achievable on every cell.")
    else:
        print("FAILED — some cell violates Smith's condition.")


if __name__ == "__main__":
    main()
