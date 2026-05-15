"""Greedy constructive schedules on translation orbits.

These do NOT call an ILP. They build an orbit-level (orbit_id, hop_i, t)
assignment by listing orbits in some order and packing each orbit's hops into
the earliest feasible (dim, dir, t) slots, then expand to per-unit Injections
via `symmetric_assignment_to_injections`.

Two variants:

* `OrbitGreedySchedule(order=<spec>)` — schedule each orbit's hops at the
  earliest time greater than the previous hop's time, skipping (dim, dir, t)
  slots already taken by another orbit. Path-internal gaps allowed.

* `PipelinedOrbitSchedule(order=<spec>)` — force each orbit to fire its hops
  pipelined: `t_i = start + i`. Pick the smallest `start` such that no
  (dim, dir, t) slot is already taken.

Both algorithms run in O(N_orbits * diameter * makespan) time. By the
Leighton-Maggs-Rao bound, makespan <= LB + diameter is achievable with a
careful construction; with the default `lpt_tail_asc` ordering, these
heuristics achieve `makespan = LB` on every (topology, router) cell tested
including 2x4x4 DOR (which the plain `lpt` ordering misses by 1 step).
See `docs/results.md`.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass

from twisted_analysis.lp.orbit import compute_orbits
from twisted_analysis.model.flow import AllToAll
from twisted_analysis.schedules.base import Injection
from twisted_analysis.schedules.lp_symmetric import symmetric_assignment_to_injections
from twisted_analysis.topology import Router, Topology


def _canonical_paths(topology: Topology, router: Router) -> dict:
    origin = tuple([0] * topology.ndim)
    orbits = compute_orbits(topology)
    canon: dict = {}
    for orbit_id, members in orbits.items():
        c = next(((s, d) for (s, d) in members if s == origin), None)
        assert c is not None
        canon[orbit_id] = router.path(c[0], c[1])
    return canon


def _edge_orbit_load(canon: dict) -> Counter:
    """Total hits on each (dim, dir) class across all canonical paths."""
    c: Counter = Counter()
    for path in canon.values():
        for _, _, dim, dir in path:
            c[(dim, dir)] += 1
    return c


_VALID_ORDERS = {"lpt", "spt", "lpt_tail_asc", "tail_asc"}


def _ordered_orbits(canon: dict, edge_load: Counter, order: str) -> list:
    """Return orbit ids sorted by the requested heuristic.

    - "lpt": longest path first (tiebreak by orbit_id).
    - "spt": shortest path first.
    - "lpt_tail_asc": longest path first; ties broken by ascending load of the
      orbit's tail edge orbit class. Empirically achieves LB on every cell
      tested (including 2x4x4 DOR where plain LPT misses by 1 step). DEFAULT.
    - "tail_asc": pure tail-load-ascending ordering, with longest-path-first
      as a secondary tiebreak.
    """
    if order == "lpt":
        return sorted(canon.keys(), key=lambda o: (-len(canon[o]), o))
    if order == "spt":
        return sorted(canon.keys(), key=lambda o: (len(canon[o]), o))
    if order == "lpt_tail_asc":
        def tail(o):
            p = canon[o]
            return edge_load[(p[-1][2], p[-1][3])] if p else 0
        return sorted(canon.keys(), key=lambda o: (-len(canon[o]), tail(o), o))
    if order == "tail_asc":
        def tail(o):
            p = canon[o]
            return edge_load[(p[-1][2], p[-1][3])] if p else 0
        return sorted(canon.keys(), key=lambda o: (tail(o), -len(canon[o]), o))
    raise ValueError(f"unknown order: {order}; valid={sorted(_VALID_ORDERS)}")


def _emit_orbit_greedy(
    topology: Topology, router: Router, order: str,
) -> dict[tuple, float]:
    """Return assignment dict (orbit_id, hop_i, t) -> 1.0 for fired slots."""
    canon = _canonical_paths(topology, router)
    edge_load = _edge_orbit_load(canon)
    edge_busy: dict[tuple[int, int], set[int]] = defaultdict(set)
    assignment: dict[tuple, float] = {}
    for orbit_id in _ordered_orbits(canon, edge_load, order):
        path = canon[orbit_id]
        prev_t = -1
        for i, (_, _, dim, dir) in enumerate(path):
            t = prev_t + 1
            while t in edge_busy[(dim, dir)]:
                t += 1
            assignment[(orbit_id, i, t)] = 1.0
            edge_busy[(dim, dir)].add(t)
            prev_t = t
    return assignment


def _emit_pipelined_orbit(
    topology: Topology, router: Router, order: str,
) -> dict[tuple, float]:
    """Pipelined: each orbit fires hops at start, start+1, ..., start+L-1."""
    canon = _canonical_paths(topology, router)
    edge_load = _edge_orbit_load(canon)
    edge_busy: dict[tuple[int, int], set[int]] = defaultdict(set)
    assignment: dict[tuple, float] = {}
    for orbit_id in _ordered_orbits(canon, edge_load, order):
        path = canon[orbit_id]
        start = 0
        while True:
            ok = True
            for i, (_, _, dim, dir) in enumerate(path):
                if (start + i) in edge_busy[(dim, dir)]:
                    ok = False
                    break
            if ok:
                break
            start += 1
        for i, (_, _, dim, dir) in enumerate(path):
            t = start + i
            assignment[(orbit_id, i, t)] = 1.0
            edge_busy[(dim, dir)].add(t)
    return assignment


def compute_hop0_firing_times(
    topology: Topology,
    router: Router,
    order: str = "lpt_tail_asc",
) -> dict:
    """Return per-orbit hop-0 firing time `t_0^O` under OrbitGreedy.

    For each orbit O, returns the OrbitGreedy step at which O fires its
    first hop. Used to derive the `round` field in `schedule.json` (see
    twisted_analysis.io.schedule.schedule_from_orbit_greedy).
    """
    assignment = _emit_orbit_greedy(topology, router, order)
    return {oid: t for (oid, hop_i, t) in assignment if hop_i == 0}


@dataclass
class OrbitGreedySchedule:
    """Greedy chain-respecting orbit packing. Polynomial time, no ILP.

    `order`: ordering of orbits before greedy packing.
      - "lpt_tail_asc" (default): longest-path-first, tiebreak tail-load-asc.
        Achieves `makespan = LB` on every (topology, router) cell tested.
      - "lpt": longest-path-first only. Misses LB on 2x4x4 DOR by 1 step.
      - "spt": shortest-first. 5-16% gap vs LPT.
      - "tail_asc": tail-load-asc only.
    """
    order: str = "lpt_tail_asc"
    name: str = "orbit_greedy"

    def __post_init__(self) -> None:
        if self.order not in _VALID_ORDERS:
            raise ValueError(f"order must be one of {sorted(_VALID_ORDERS)}; got {self.order}")

    def emit(self, workload: AllToAll) -> list[Injection]:
        assignment = _emit_orbit_greedy(workload.topology, workload.router, self.order)
        return symmetric_assignment_to_injections(
            workload.topology, workload.router, list(workload.flows), assignment,
        )


@dataclass
class PipelinedOrbitSchedule:
    """Each orbit fires its hops pipelined (t_i = start + i). Polynomial time."""
    order: str = "lpt_tail_asc"
    name: str = "pipelined_orbit"

    def __post_init__(self) -> None:
        if self.order not in _VALID_ORDERS:
            raise ValueError(f"order must be one of {sorted(_VALID_ORDERS)}; got {self.order}")

    def emit(self, workload: AllToAll) -> list[Injection]:
        assignment = _emit_pipelined_orbit(workload.topology, workload.router, self.order)
        return symmetric_assignment_to_injections(
            workload.topology, workload.router, list(workload.flows), assignment,
        )
