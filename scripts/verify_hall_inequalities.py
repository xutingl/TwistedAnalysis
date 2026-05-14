"""Verify the König/Hall inequalities (★), (†), and (‡) from
docs/orbit_greedy_optimality.md hold on every (topology, router) cell.

(★) Σ_i load_i(e) = load(e) ≤ LB                  — conservation identity
(†) #{j : e_i^j = e, L_j = L} + |B_i(e) ∩ [0, LB-L+i]| ≤ LB - L + i + 1
    — deadline-aware Hall at level i, edge e, length L
(‡) D(e, T) := |{(j,i) : e_i^j = e, LB - L_j + i ≤ T}| ≤ T + 1   ∀ T
    — Smith's EDF feasibility on edge e

(‡) is the *aggregate* form of (†) per Reduction Lemma 4.3.7.
"""
from __future__ import annotations
from collections import Counter, defaultdict

from twisted_analysis.model import AllToAll
from twisted_analysis.schedules.orbit_greedy import (
    _canonical_paths, _emit_orbit_greedy, _edge_orbit_load,
)
from twisted_analysis.topology import DORRouter, ILPRouter, Topology


def check(slice_, router_name) -> None:
    t = Topology(slice=slice_)
    r = DORRouter(t) if router_name == "dor" else ILPRouter(t)
    w = AllToAll(t, r, msg_size=1)
    lb = w.lower_bound
    canon = _canonical_paths(t, r)
    edge_load = _edge_orbit_load(canon)
    d = max(len(p) for p in canon.values())

    # (★) Σ_i load_i(e) = load(e), trivially true; also confirm ≤ LB.
    load_at_level: dict = defaultdict(Counter)  # level i -> {edge -> count}
    for o, path in canon.items():
        for i, (_, _, dim, dir) in enumerate(path):
            load_at_level[i][(dim, dir)] += 1
    star_ok = True
    for e, total in edge_load.items():
        s = sum(load_at_level[i][e] for i in range(d))
        if s != total or total > lb:
            star_ok = False
            print(f"  (★) FAIL: edge {e}: Σ_i load_i = {s}, load = {total}, LB = {lb}")
    assert star_ok

    # (†): use the greedy's actual schedule to instantiate B_i(e).
    assignment = _emit_orbit_greedy(t, r, "lpt_tail_asc")
    # B_i(e) := { t : (orbit, hop, t) with the hop's edge = e AND that orbit's hop index < i }
    # But for our purpose, B_i(e) is just the slots booked on e by ANY orbit's
    # hops scheduled at any prior level — i.e., all (orbit, hop_idx, t) with
    # hop_idx < i and the orbit's hop_idx hop's edge = e.
    dagger_violations = 0
    dagger_tight = 0
    for e in edge_load:
        for i in range(d):
            # B_i(e) = slots on e booked by hops at levels < i
            booked_in_window: dict = defaultdict(set)  # L -> set of booked slots in [0, LB-L+i]
            for (o, hop_idx, time), val in assignment.items():
                if val < 0.5 or hop_idx >= i:
                    continue
                path = canon[o]
                if (path[hop_idx][2], path[hop_idx][3]) != e:
                    continue
                # we don't know L_j here; track per-L
                for L in range(1, d + 1):
                    if time <= lb - L + i:
                        booked_in_window[L].add(time)
            # Count orbits with hop i on e AND length L
            orbits_by_L: Counter = Counter()
            for o, path in canon.items():
                if len(path) <= i:
                    continue
                if (path[i][2], path[i][3]) != e:
                    continue
                orbits_by_L[len(path)] += 1
            for L, count in orbits_by_L.items():
                budget = lb - L + i + 1
                used = len(booked_in_window[L])
                lhs = count + used
                if lhs > budget:
                    dagger_violations += 1
                    print(f"  (†) VIOLATION: edge {e} level i={i} L={L}: "
                          f"{count} + {used} = {lhs} > budget {budget}")
                elif lhs == budget:
                    dagger_tight += 1

    # (‡) — Smith's cumulative form, the cleaner aggregate of (†).
    # D(e, T) := |{(j, i) : e_i^j = e, LB - L_j + i ≤ T}|
    ddag_violations = 0
    ddag_min_slack = lb + 1
    ddag_min_at = None
    for e in edge_load:
        # Collect (orbit, hop) demands on e, with deadlines
        demands: list = []
        for o, path in canon.items():
            for i, (_, _, dim, dir) in enumerate(path):
                if (dim, dir) == e:
                    demands.append(lb - len(path) + i)  # deadline
        demands.sort()
        for k, T in enumerate(sorted(set(demands))):
            d_e_T = sum(1 for d_ in demands if d_ <= T)
            slack = T + 1 - d_e_T
            if slack < 0:
                ddag_violations += 1
                print(f"  (‡) VIOLATION: edge {e} T={T}: D={d_e_T} > {T+1}")
            if slack < ddag_min_slack:
                ddag_min_slack = slack
                ddag_min_at = (e, T)

    # (♦) — Long-tail bound, the cleanest sufficient condition for (‡)
    # |{(j, i) : e_i^j = e, L_j - i >= R}| <= load(e) - R + 1 for R >= 2
    diamond_violations = 0
    diamond_min_slack = lb + 1
    for e in edge_load:
        for R in range(2, d + 1):
            long_tail_count = 0
            for o, path in canon.items():
                Lj = len(path)
                for i in range(Lj):
                    if (path[i][2], path[i][3]) != e:
                        continue
                    if Lj - i >= R:
                        long_tail_count += 1
            budget = edge_load[e] - R + 1
            slack = budget - long_tail_count
            if slack < 0:
                diamond_violations += 1
                print(f"  (♦) VIOLATION: edge {e} R={R}: {long_tail_count} > {budget}")
            if slack < diamond_min_slack:
                diamond_min_slack = slack

    status = "OK" if dagger_violations == 0 else f"FAIL ({dagger_violations})"
    ddag_status = "OK" if ddag_violations == 0 else f"FAIL ({ddag_violations})"
    dia_status = "OK" if diamond_violations == 0 else f"FAIL ({diamond_violations})"
    print(f"  {str(slice_):<10} {router_name:<3} LB={lb:>3} d={d}  "
          f"(★) OK  (†) {status}  (‡) {ddag_status}  (♦) {dia_status}  "
          f"min-slack(♦)={diamond_min_slack}")


def main() -> None:
    for slice_ in [(2, 4), (2, 2, 4), (2, 4, 4), (4, 8), (4, 4, 8)]:
        for rn in ["dor", "ilp"]:
            check(slice_, rn)


if __name__ == "__main__":
    main()
