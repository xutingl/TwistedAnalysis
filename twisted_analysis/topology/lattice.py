from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from itertools import product
from typing import Iterator

Node = tuple[int, ...]
DirectedLink = tuple[Node, Node, int, int]  # (u, v, dim, dir)


@dataclass(frozen=True)
class Topology:
    """A twisted-torus topology with shape `slice` (all sizes in {S, 2S})."""
    slice: tuple[int, ...]

    def __post_init__(self) -> None:
        s_min = min(self.slice)
        assert all(s in (s_min, 2 * s_min) for s in self.slice), (
            f"slice {self.slice} violates the {{S, 2S}} family"
        )

    @property
    def n_nodes(self) -> int:
        n = 1
        for s in self.slice:
            n *= s
        return n

    @property
    def ndim(self) -> int:
        return len(self.slice)

    def neighbor(self, node: Node, dim: int, dir: int) -> Node:
        assert len(node) == self.ndim
        assert 0 <= dim < self.ndim
        assert dir in (-1, 1)
        new = list(node)
        new[dim] += dir
        wrapped = new[dim] < 0 or new[dim] >= self.slice[dim]
        if wrapped:
            shift = self.slice[dim]
            new = [(new[i] + shift) % self.slice[i] for i in range(self.ndim)]
        return tuple(new)

    def nodes(self) -> Iterator[Node]:
        yield from product(*(range(s) for s in self.slice))

    def directed_links(self) -> Iterator[DirectedLink]:
        for u in self.nodes():
            for dim in range(self.ndim):
                for dir in (-1, 1):
                    v = self.neighbor(u, dim, dir)
                    yield (u, v, dim, dir)

    def bfs_distances(self) -> dict[Node, dict[Node, int]]:
        adj: dict[Node, list[Node]] = {u: [] for u in self.nodes()}
        for u, v, _, _ in self.directed_links():
            adj[u].append(v)
        result: dict[Node, dict[Node, int]] = {}
        for src in self.nodes():
            dist = {src: 0}
            q: deque[Node] = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        q.append(v)
            result[src] = dist
        return result
