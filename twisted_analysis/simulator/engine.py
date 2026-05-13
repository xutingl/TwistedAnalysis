from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from itertools import count

from twisted_analysis.topology import Topology, Router, DirectedLink
from twisted_analysis.model.flow import Flow
from twisted_analysis.schedules.base import Injection


@dataclass
class _Unit:
    """One unit of payload from one flow."""
    flow: Flow
    path: tuple[DirectedLink, ...]
    next_hop_idx: int = 0
    priority: int = 0
    seq: int = 0  # tie-break for deterministic ordering

    @property
    def at_link(self) -> DirectedLink:
        return self.path[self.next_hop_idx]

    @property
    def delivered(self) -> bool:
        return self.next_hop_idx >= len(self.path)


class Simulator:
    """Step-synchronous, store-and-forward, capacity-1 simulator."""

    def __init__(self, topology: Topology, router: Router, flows: list[Flow]):
        self.topology = topology
        self.router = router
        self.flows = flows
        self.path_map = {f: router.path(f.src, f.dst) for f in flows}
        self.link_queue: dict[DirectedLink, deque[_Unit]] = {}
        self.pending_injections: list[Injection] = []
        self.units: list[_Unit] = []
        self._seq = count()
        self.busy_per_step: list[int] = []
        self.delivered_count = 0
        self.total_units = sum(f.size for f in flows)

    def inject(self, injection: Injection) -> None:
        self.pending_injections.append(injection)

    def _enqueue_injections(self, step: int) -> None:
        ready = [i for i in self.pending_injections if i.start_step == step]
        self.pending_injections = [i for i in self.pending_injections if i.start_step != step]
        for inj in ready:
            path = self.path_map[inj.flow]
            if not path:  # self-loop, instantly "delivered"
                self.delivered_count += inj.flow.size
                continue
            for _ in range(inj.flow.size):
                u = _Unit(
                    flow=inj.flow,
                    path=path,
                    priority=inj.priority,
                    seq=next(self._seq),
                )
                self.units.append(u)
                self.link_queue.setdefault(path[0], deque()).append(u)

    def run(self, max_steps: int = 1_000_000) -> int:
        step = 0
        while self.delivered_count < self.total_units and step < max_steps:
            self._enqueue_injections(step)
            busy = self._step()
            self.busy_per_step.append(busy)
            step += 1
        if self.delivered_count < self.total_units:
            raise RuntimeError(f"sim did not drain in {max_steps} steps")
        return step

    def _step(self) -> int:
        """Each link picks one unit; chosen units advance one hop. Return # active links."""
        # Selection: pick (priority, seq) minimum per link.
        chosen: list[tuple[DirectedLink, _Unit]] = []
        for link, q in self.link_queue.items():
            if not q:
                continue
            best = min(q, key=lambda u: (u.priority, u.seq))
            q.remove(best)
            chosen.append((link, best))
        # Advance.
        for link, u in chosen:
            u.next_hop_idx += 1
            if u.delivered:
                self.delivered_count += 1
            else:
                self.link_queue.setdefault(u.at_link, deque()).append(u)
        # Cleanup empty queues to keep iteration sane.
        self.link_queue = {k: v for k, v in self.link_queue.items() if v}
        return len(chosen)
