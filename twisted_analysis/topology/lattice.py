from __future__ import annotations
from dataclasses import dataclass

Node = tuple[int, ...]


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

    def neighbor(self, node: Node, dim: int, dir: int) -> Node:
        assert len(node) == len(self.slice)
        assert 0 <= dim < len(self.slice)
        assert dir in (-1, 1)

        new = list(node)
        new[dim] += dir
        wrapped = new[dim] < 0 or new[dim] >= self.slice[dim]
        if wrapped:
            shift = self.slice[dim]
            new = [(new[i] + shift) % self.slice[i] for i in range(len(new))]
        return tuple(new)
