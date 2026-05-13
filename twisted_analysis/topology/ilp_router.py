"""ILP-based router: picks one minimal path per (src, dst) to minimize max
channel load. Ported from btowles' solve_minimal_routes with PuLP/CBC.

Uses translational symmetry: variables are created only for paths from a
canonical origin; other sources reuse the same variables under translation.
This reduces variable count by N.
"""
from __future__ import annotations
import collections
import itertools
from dataclasses import dataclass
from functools import cached_property
from typing import Iterable

import pulp

from twisted_analysis.topology.lattice import Topology, Node, DirectedLink

Path = tuple[DirectedLink, ...]


def _walk(topology: Topology, src: Node, delta: tuple[int, ...]) -> Node:
    node = src
    for dim, count in enumerate(delta):
        direction = 1 if count >= 0 else -1
        for _ in range(abs(count)):
            node = topology.neighbor(node, dim, direction)
    return node


def _delta_to_path(topology: Topology, src: Node, delta: tuple[int, ...]) -> Path:
    node = src
    hops: list[DirectedLink] = []
    for dim, count in enumerate(delta):
        direction = 1 if count >= 0 else -1
        for _ in range(abs(count)):
            nxt = topology.neighbor(node, dim, direction)
            hops.append((node, nxt, dim, direction))
            node = nxt
    return tuple(hops)


def _minimal_path_deltas(topology: Topology) -> dict[Node, list[tuple[int, ...]]]:
    """For each destination from origin, returns all minimal-hop delta tuples."""
    origin = tuple([0] * topology.ndim)
    endpoint_to_deltas: dict[Node, list[tuple[int, ...]]] = {}
    kr = [range(-topology.slice[d], topology.slice[d] + 1)
          for d in range(topology.ndim)]
    for delta in itertools.product(*kr):
        endpoint = _walk(topology, origin, delta)
        endpoint_to_deltas.setdefault(endpoint, []).append(delta)
    result: dict[Node, list[tuple[int, ...]]] = {}
    for dst, deltas in endpoint_to_deltas.items():
        min_hops = min(sum(abs(x) for x in delta) for delta in deltas)
        result[dst] = [d for d in deltas if sum(abs(x) for x in d) == min_hops]
    return result


def _coord_diff(topology: Topology, src: Node, dst: Node) -> tuple[int, ...]:
    """Return delta vector representing dst-from-origin under translation
    inverse-of-src. Computed via BFS from src to dst then replay from origin.
    """
    from collections import deque
    origin = tuple([0] * topology.ndim)
    parent: dict[Node, tuple[Node, int, int] | None] = {src: None}
    q: deque[Node] = deque([src])
    while q:
        u = q.popleft()
        if u == dst:
            break
        for dim in range(topology.ndim):
            for dir in (-1, 1):
                v = topology.neighbor(u, dim, dir)
                if v not in parent:
                    parent[v] = (u, dim, dir)
                    q.append(v)
    hops: list[tuple[int, int]] = []
    cur = dst
    while parent[cur] is not None:
        p, dim, dir = parent[cur]
        hops.append((dim, dir))
        cur = p
    hops.reverse()
    node = origin
    delta = [0] * topology.ndim
    for dim, dir in hops:
        node = topology.neighbor(node, dim, dir)
        delta[dim] += dir
    return tuple(delta)


@dataclass(frozen=True)
class ILPRouter:
    """Load-balanced minimal router. Implements the Router Protocol.

    On first .path() call, solves an ILP to pick one minimal path per
    (origin, dst) such that max channel load over the directed-edge orbit
    classes is minimized. Subsequent .path() calls use the cached table.
    """
    topology: Topology
    ilp_timeout_seconds: float = 60.0

    @cached_property
    def _origin(self) -> Node:
        return tuple([0] * self.topology.ndim)

    @cached_property
    def _chosen_delta(self) -> dict[Node, tuple[int, ...]]:
        t = self.topology
        minimal = _minimal_path_deltas(t)
        prob = pulp.LpProblem("ilp_router", pulp.LpMinimize)

        x: dict[tuple[Node, int], pulp.LpVariable] = {}
        for dst, deltas in minimal.items():
            if dst == self._origin or len(deltas) <= 1:
                continue
            choice_sum = []
            for k, _ in enumerate(deltas):
                v = pulp.LpVariable(f"x_{dst}_{k}", cat=pulp.LpBinary)
                x[(dst, k)] = v
                choice_sum.append(v)
            prob += pulp.lpSum(choice_sum) == 1

        edge_orbit_fixed: collections.Counter = collections.Counter()
        edge_orbit_var_contrib: dict[tuple[int, int], list[tuple[pulp.LpVariable, int]]] = \
            collections.defaultdict(list)
        for dst, deltas in minimal.items():
            if dst == self._origin:
                continue
            if len(deltas) == 1:
                delta = deltas[0]
                for dim, count in enumerate(delta):
                    if count > 0:
                        edge_orbit_fixed[(dim, 1)] += count
                    elif count < 0:
                        edge_orbit_fixed[(dim, -1)] += -count
            else:
                for k, delta in enumerate(deltas):
                    v = x[(dst, k)]
                    for dim, count in enumerate(delta):
                        if count > 0:
                            edge_orbit_var_contrib[(dim, 1)].append((v, count))
                        elif count < 0:
                            edge_orbit_var_contrib[(dim, -1)].append((v, -count))

        max_bound = sum(edge_orbit_fixed.values()) + sum(
            sum(c for _, c in vs) for vs in edge_orbit_var_contrib.values()
        )
        M = pulp.LpVariable("M", lowBound=0, upBound=max_bound or 1)
        prob += M

        all_orbits = set(edge_orbit_fixed.keys()) | set(edge_orbit_var_contrib.keys())
        for orbit in all_orbits:
            fixed = edge_orbit_fixed.get(orbit, 0)
            contribs = edge_orbit_var_contrib.get(orbit, [])
            prob += (pulp.lpSum(v * c for v, c in contribs) + fixed) <= M

        solver = pulp.getSolver(
            "PULP_CBC_CMD", msg=False, timeLimit=int(self.ilp_timeout_seconds)
        )
        prob.solve(solver)
        if pulp.LpStatus[prob.status] not in ("Optimal", "Not Solved"):
            raise RuntimeError(f"ILP router failed: {pulp.LpStatus[prob.status]}")

        chosen: dict[Node, tuple[int, ...]] = {}
        for dst, deltas in minimal.items():
            if dst == self._origin:
                chosen[dst] = tuple([0] * t.ndim)
                continue
            if len(deltas) == 1:
                chosen[dst] = deltas[0]
            else:
                picked = None
                for k, delta in enumerate(deltas):
                    v = x[(dst, k)]
                    val = pulp.value(v)
                    if val is not None and val > 0.5:
                        if picked is None:
                            picked = delta
                if picked is None:
                    picked = deltas[0]
                chosen[dst] = picked
        return chosen

    def path(self, src: Node, dst: Node) -> Path:
        if src == dst:
            return ()
        t = self.topology
        canonical_dst = _walk(t, self._origin, _coord_diff(t, src, dst))
        delta = self._chosen_delta[canonical_dst]
        return _delta_to_path(t, src, delta)
