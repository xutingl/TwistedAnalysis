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
    hop_schedule: tuple[int, ...] = ()  # LP fire step per hop; empty = use static priority

    @property
    def at_link(self) -> DirectedLink:
        return self.path[self.next_hop_idx]

    @property
    def delivered(self) -> bool:
        return self.next_hop_idx >= len(self.path)

    def effective_priority(self) -> int:
        """Return the priority to use for this unit's current hop.

        If a per-hop LP schedule is available, use the LP-assigned fire step for
        the current hop as the priority (lower step = higher priority). Otherwise
        fall back to the static `priority` field.
        """
        if self.hop_schedule and self.next_hop_idx < len(self.hop_schedule):
            return self.hop_schedule[self.next_hop_idx]
        return self.priority


class Simulator:
    """Step-synchronous, store-and-forward, capacity-1 simulator."""

    def __init__(
        self,
        topology: Topology,
        router: Router,
        flows: list[Flow],
        record_history: bool = False,
    ):
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
        self.record_history = record_history
        self.history: list[tuple[int, DirectedLink, Flow, int]] = []
        self.link_busy: dict[DirectedLink, list[bool]] = {}

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
                    hop_schedule=inj.hop_schedule,
                )
                self.units.append(u)
                self.link_queue.setdefault(path[0], deque()).append(u)

    def run(self, max_steps: int = 1_000_000) -> int:
        step = 0
        while self.delivered_count < self.total_units and step < max_steps:
            self._enqueue_injections(step)
            busy_links = self._step(step)
            self.busy_per_step.append(len(busy_links))
            if self.record_history:
                for e in self.topology.directed_links():
                    self.link_busy.setdefault(e, []).append(e in busy_links)
            step += 1
        if self.delivered_count < self.total_units:
            raise RuntimeError(f"sim did not drain in {max_steps} steps")
        return step

    def _step(self, step: int) -> set[DirectedLink]:
        """Each link picks one unit; chosen units advance one hop.

        Returns the set of busy links this step.

        Selection uses (effective_priority, seq) minimum per link.
        effective_priority() uses the LP-assigned fire step for the current hop
        if available, otherwise falls back to the static priority field.
        """
        chosen: list[tuple[DirectedLink, _Unit]] = []
        for link, q in self.link_queue.items():
            if not q:
                continue
            best = min(q, key=lambda u: (u.effective_priority(), u.seq))
            q.remove(best)
            chosen.append((link, best))
        busy_links = {link for link, _ in chosen}
        # Advance.
        for link, u in chosen:
            if self.record_history:
                self.history.append((step, link, u.flow, u.next_hop_idx))
            u.next_hop_idx += 1
            if u.delivered:
                self.delivered_count += 1
            else:
                self.link_queue.setdefault(u.at_link, deque()).append(u)
        # Cleanup empty queues to keep iteration sane.
        self.link_queue = {k: v for k, v in self.link_queue.items() if v}
        return busy_links
