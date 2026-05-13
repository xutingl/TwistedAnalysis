from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property
from itertools import product
from typing import Protocol, runtime_checkable

from twisted_analysis.topology.lattice import Topology, Node, DirectedLink

# A "Hop" is one directed link: (u, v, dim, dir).
Path = tuple[DirectedLink, ...]


@runtime_checkable
class Router(Protocol):
    """Structural protocol for routers. Any object with .path(src, dst) -> Path
    is a Router. The two concrete implementations are DORRouter (dimension-order)
    and ILPRouter (load-balanced minimal routing).
    """
    topology: "Topology"
    def path(self, src: "Node", dst: "Node") -> "Path": ...


@dataclass(frozen=True)
class DORRouter:
    """Twist-aware DOR router. Resolves dims in order of decreasing slice size.

    For each (src, dst), enumerates candidate wrap choices, picks the minimum
    hop count with deterministic tie-break (no-wrap > wrap, +dir > -dir).
    """
    topology: Topology

    @cached_property
    def _dim_order(self) -> tuple[int, ...]:
        # Resolve largest dim first (configurable later).
        return tuple(sorted(range(self.topology.ndim),
                            key=lambda d: -self.topology.slice[d]))

    @cached_property
    def _neighbor_table(self) -> dict[tuple[Node, int, int], Node]:
        """Precomputed neighbor lookup: (node, dim, dir) -> neighbor node."""
        table: dict[tuple[Node, int, int], Node] = {}
        t = self.topology
        for node in t.nodes():
            for dim in range(t.ndim):
                for dir in (-1, 1):
                    table[(node, dim, dir)] = t.neighbor(node, dim, dir)
        return table

    @cached_property
    def _sorted_candidates(self) -> list[tuple[int, tuple[int, ...]]]:
        """All (total, steps) sorted by (total, wrap_count, tie-break) for reuse."""
        t = self.topology
        # Generate all step vectors and sort by total hops ascending.
        ranges = [range(-t.slice[d], t.slice[d] + 1) for d in range(t.ndim)]
        all_steps = list(product(*ranges))
        # Sort by total hops first; this allows early termination per pair.
        all_steps.sort(key=lambda s: (
            sum(abs(x) for x in s),
            sum(abs(s[d]) > t.slice[d] // 2 for d in range(t.ndim)),
            tuple(-s[d] for d in range(t.ndim)),
        ))
        return [(sum(abs(x) for x in s), s) for s in all_steps]

    def _walk_endpoint_fast(self, src: Node, steps: tuple[int, ...]) -> Node:
        """Walk using precomputed neighbor table."""
        node = src
        nb = self._neighbor_table
        for dim in self._dim_order:
            count = steps[dim]
            if count == 0:
                continue
            direction = 1 if count > 0 else -1
            for _ in range(abs(count)):
                node = nb[(node, dim, direction)]
        return node

    def _best_steps(self, src: Node, dst: Node) -> tuple[int, ...]:
        """Return the best step vector for src→dst.

        Iterates candidates in (total, tie-break) order.  Stops as soon as we
        move past the minimum total found — all subsequent candidates have
        higher total or worse tie-break.
        """
        best: tuple[int, ...] | None = None
        best_total = -1
        for total, steps in self._sorted_candidates:
            if best is not None and total > best_total:
                # All remaining candidates have strictly higher total — done.
                break
            if self._walk_endpoint_fast(src, steps) == dst:
                if best is None:
                    best = steps
                    best_total = total
                # Since candidates are sorted by tie-break too, first match is best.
                break
        if best is None:
            # Fallback: BFS-shortest path reconstruction.
            best = self._bfs_steps(src, dst)
        return best

    def _bfs_steps(self, src: Node, dst: Node) -> tuple[int, ...]:
        """Recover a step vector via BFS — fallback for edge cases."""
        from collections import deque
        adj: dict[Node, list[tuple[Node, int, int]]] = {u: [] for u in self.topology.nodes()}
        for u, v, dim, dir in self.topology.directed_links():
            adj[u].append((v, dim, dir))
        parent: dict[Node, tuple[Node, int, int] | None] = {src: None}
        q: deque[Node] = deque([src])
        while q:
            u = q.popleft()
            if u == dst:
                break
            for v, dim, dir in adj[u]:
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
        steps = [0] * self.topology.ndim
        for dim, dir in hops:
            steps[dim] += dir
        assert self._walk_endpoint_fast(src, tuple(steps)) == dst, (
            f"BFS fallback failed for {src} -> {dst}"
        )
        return tuple(steps)

    def path(self, src: Node, dst: Node) -> Path:
        if src == dst:
            return ()
        steps = self._best_steps(src, dst)
        # Construct the actual link sequence by walking in _dim_order.
        node = src
        nb = self._neighbor_table
        hops: list[DirectedLink] = []
        for dim in self._dim_order:
            count = steps[dim]
            if count == 0:
                continue
            direction = 1 if count > 0 else -1
            for _ in range(abs(count)):
                nxt = nb[(node, dim, direction)]
                hops.append((node, nxt, dim, direction))
                node = nxt
        assert node == dst, f"router walk landed at {node}, expected {dst}"
        return tuple(hops)
