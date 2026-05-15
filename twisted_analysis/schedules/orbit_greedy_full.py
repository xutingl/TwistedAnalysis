"""Orbit-greedy with FULL physical-edge accounting.

Differs from twisted_analysis.schedules.orbit_greedy in one place: the
busy table is keyed on the literal set of physical edges an orbit's `N`
flows traverse at each hop, rather than on the `(dim, dir)` class label.

When the routing is translation-equivariant under the (dim, dir) action
(DOR, ILP), the two formulations are equivalent: each (dim, dir) class is
saturated by exactly N distinct physical edges, so checking the class is
the same as checking the edge set. When the routing is NOT equivariant
under (dim, dir) — e.g., a TPU OCS-derived "loaded" routing where
twist-wrap edges and standard edges are intermixed — the two diverge,
and only the full-edge formulation respects physical-edge capacity.
"""
from __future__ import annotations
from collections import Counter, defaultdict

from twisted_analysis.io.coords import flatten
from twisted_analysis.lp.orbit import compute_orbits
from twisted_analysis.topology import Topology

_VALID_ORDERS = {"lpt", "spt", "lpt_tail_asc", "tail_asc"}


def _orbit_hop_edge_sets(
    topology: Topology,
    table: list[list[list[int]]],
) -> dict:
    """For each orbit, return [set_of_edges_at_hop_0, set_at_hop_1, ...]."""
    orbits = compute_orbits(topology)
    slice_ = topology.slice
    per_orbit: dict = {}
    for orbit_id, members in orbits.items():
        hop_sets: list[set[tuple[int, int]]] = []
        for src, dst in members:
            src_flat = flatten(src, slice_)
            dst_flat = flatten(dst, slice_)
            path = table[src_flat][dst_flat]
            for i in range(len(path) - 1):
                if i >= len(hop_sets):
                    hop_sets.append(set())
                hop_sets[i].add((path[i], path[i + 1]))
        per_orbit[orbit_id] = hop_sets
    return per_orbit


def _edge_orbit_load_full(per_orbit: dict) -> Counter:
    """Per-physical-edge total demand across the workload."""
    c: Counter = Counter()
    for hop_sets in per_orbit.values():
        for edges in hop_sets:
            for e in edges:
                c[e] += 1
    return c


def _ordered_orbits_full(per_orbit: dict, edge_load: Counter, order: str) -> list:
    if order == "lpt":
        return sorted(per_orbit, key=lambda o: (-len(per_orbit[o]), o))
    if order == "spt":
        return sorted(per_orbit, key=lambda o: (len(per_orbit[o]), o))
    if order == "lpt_tail_asc":
        def tail_load(o):
            hops = per_orbit[o]
            if not hops:
                return 0
            return max(edge_load[e] for e in hops[-1])
        return sorted(per_orbit, key=lambda o: (-len(per_orbit[o]), tail_load(o), o))
    if order == "tail_asc":
        def tail_load(o):
            hops = per_orbit[o]
            if not hops:
                return 0
            return max(edge_load[e] for e in hops[-1])
        return sorted(per_orbit, key=lambda o: (tail_load(o), -len(per_orbit[o]), o))
    raise ValueError(f"unknown order: {order}; valid={sorted(_VALID_ORDERS)}")


def compute_hop0_firing_times_full(
    topology: Topology,
    table: list[list[list[int]]],
    *,
    order: str = "lpt_tail_asc",
) -> dict:
    """Return per-orbit hop-0 firing time under orbit_greedy_full.

    Greedy: process orbits in `order`; for each orbit, find the earliest
    `t_0` such that for every hop i, ALL physical edges in the orbit's
    hop-i set are free at time t_0 + i. Mark them busy after firing.
    """
    if order not in _VALID_ORDERS:
        raise ValueError(f"order must be one of {sorted(_VALID_ORDERS)}; got {order!r}")
    per_orbit = _orbit_hop_edge_sets(topology, table)
    edge_load = _edge_orbit_load_full(per_orbit)
    edge_busy: dict[tuple[int, int], set[int]] = defaultdict(set)
    t_hop0: dict = {}
    for orbit_id in _ordered_orbits_full(per_orbit, edge_load, order):
        hops = per_orbit[orbit_id]
        # Find smallest start such that for every i, hops[i] is fully free at start+i.
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
