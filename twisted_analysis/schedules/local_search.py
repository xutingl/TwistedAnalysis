"""Local-search repair scheduler.

Given a seed schedule (any feasible list of {round, src, dst, path} entries),
repeatedly try the cheapest improving move:
  - For each flow f currently finishing at time M (the makespan):
      try shifting f's round by -k for k = 1, 2, ... until either the shift
      is feasible (no edge-time conflict) or we exhaust the search.
  - If a shift reduces M, accept and continue.

When no shift reduces M, terminate. Polynomial per iteration: O(F * M * D)
where F = flows, M = current makespan, D = max path length.

This is a refinement step — call after a base scheduler produces a feasible
schedule (orbit_greedy_full, lp_rounding, etc.)."""
from __future__ import annotations
from collections import defaultdict
from copy import deepcopy

from twisted_analysis.topology import Topology


def _occupancy(schedule):
    """Return {(edge, time): flow_idx or sentinel}; assumes valid schedule."""
    occ: dict[tuple[tuple[int, int], int], int] = {}
    for f_idx, e in enumerate(schedule):
        r = e["round"]
        path = e["path"]
        for h in range(len(path) - 1):
            occ[((path[h], path[h + 1]), r + h)] = f_idx
    return occ


def _makespan(schedule):
    m = 0
    for e in schedule:
        L = len(e["path"]) - 1
        f = e["round"] + L
        if f > m:
            m = f
    return m


def _try_shift(schedule, occ, f_idx, new_round):
    """Return new occ if shifting flow f_idx to new_round is feasible, else None."""
    if new_round < 0:
        return None
    e = schedule[f_idx]
    old_round = e["round"]
    if new_round == old_round:
        return occ
    path = e["path"]
    # Remove old slots from occ.
    candidate = dict(occ)
    for h in range(len(path) - 1):
        key = ((path[h], path[h + 1]), old_round + h)
        if candidate.get(key) == f_idx:
            del candidate[key]
    # Check new slots free.
    for h in range(len(path) - 1):
        key = ((path[h], path[h + 1]), new_round + h)
        if key in candidate:
            return None
    # Apply.
    for h in range(len(path) - 1):
        key = ((path[h], path[h + 1]), new_round + h)
        candidate[key] = f_idx
    return candidate


def local_search_repair(
    topology: Topology,
    table: list[list[list[int]]],  # accepted for signature symmetry, unused
    seed_schedule: list[dict],
    *,
    max_iters: int = 1000,
) -> list[dict]:
    """Repeatedly apply earliest-shift moves to reduce makespan.

    Returns a feasible schedule with makespan <= seed_schedule's.
    """
    schedule = [dict(e) for e in seed_schedule]
    occ = _occupancy(schedule)
    M = _makespan(schedule)

    for _ in range(max_iters):
        improved = False
        # Identify flows finishing at M.
        late_flows = []
        for f_idx, e in enumerate(schedule):
            L = len(e["path"]) - 1
            if e["round"] + L == M:
                late_flows.append(f_idx)
        # Try to shift each by -1, -2, ...
        for f_idx in late_flows:
            old_r = schedule[f_idx]["round"]
            best_shift = None
            for new_r in range(old_r - 1, -1, -1):
                cand_occ = _try_shift(schedule, occ, f_idx, new_r)
                if cand_occ is not None:
                    best_shift = (new_r, cand_occ)
                else:
                    break  # earlier slots usually only become harder; bail
            if best_shift is not None:
                new_r, new_occ = best_shift
                schedule[f_idx]["round"] = new_r
                occ = new_occ
                improved = True
                break
        if not improved:
            break
        M = _makespan(schedule)

    schedule.sort(key=lambda e: (e["round"], e["src"]))
    return schedule
