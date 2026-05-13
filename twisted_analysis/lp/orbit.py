from __future__ import annotations
from collections import defaultdict, deque

from twisted_analysis.topology import Topology, Node

OrbitId = Node  # identified by canonical dst-from-origin


def _canonical_dst(topology: Topology, src: Node, dst: Node) -> Node:
    """Return the canonical dst-from-origin for the flow (src, dst).

    Algorithm: BFS from src to dst to get a shortest-hop path; replay that path
    starting from origin to get the canonical endpoint.
    """
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
    for dim, dir in hops:
        node = topology.neighbor(node, dim, dir)
    return node


def compute_orbits(topology: Topology) -> dict[OrbitId, list[tuple[Node, Node]]]:
    """Group AllToAll flows by translation orbit.

    Orbit id = dst-from-origin. Each orbit contains N members (one per src).
    """
    origin = tuple([0] * topology.ndim)
    orbits: dict[OrbitId, list[tuple[Node, Node]]] = defaultdict(list)
    for src in topology.nodes():
        for dst in topology.nodes():
            if src == dst:
                continue
            if src == origin:
                orbit_id = dst
            else:
                orbit_id = _canonical_dst(topology, src, dst)
            orbits[orbit_id].append((src, dst))
    return dict(orbits)
