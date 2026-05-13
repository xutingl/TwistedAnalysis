from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from functools import cached_property

from twisted_analysis.topology import Topology, Router, Node, Path, DirectedLink


@dataclass(frozen=True)
class Flow:
    src: Node
    dst: Node
    size: int  # units of payload


@dataclass(frozen=True)
class AllToAll:
    topology: Topology
    router: Router
    msg_size: int = 1

    @cached_property
    def flows(self) -> tuple[Flow, ...]:
        return tuple(
            Flow(s, d, self.msg_size)
            for s in self.topology.nodes()
            for d in self.topology.nodes()
            if s != d
        )

    @cached_property
    def _path_map(self) -> dict[Flow, Path]:
        return {f: self.router.path(f.src, f.dst) for f in self.flows}

    def path(self, flow: Flow) -> Path:
        return self._path_map[flow]

    @cached_property
    def link_load(self) -> dict[DirectedLink, int]:
        c: Counter[DirectedLink] = Counter()
        for f in self.flows:
            for e in self._path_map[f]:
                c[e] += f.size
        return dict(c)

    @cached_property
    def lower_bound(self) -> int:
        if not self.link_load:
            return 0
        return max(self.link_load.values())

    def bottleneck_edges(self) -> list[DirectedLink]:
        lb = self.lower_bound
        return [e for e, load in self.link_load.items() if load == lb]
