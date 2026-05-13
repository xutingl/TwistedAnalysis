from __future__ import annotations
from dataclasses import dataclass

from twisted_analysis.model.flow import Flow, AllToAll
from twisted_analysis.schedules.base import Injection, Schedule
from twisted_analysis.topology import Topology, Router


@dataclass
class RoundRobinSchedule:
    """Latin-square AllToAll: in phase r, node i sends to (i+r) mod N.

    Phases run back-to-back; each phase starts when the previous one drains
    (computed by a dry-run simulation per phase).
    """
    name: str = "round_robin"

    def emit(self, workload: AllToAll) -> list[Injection]:
        from twisted_analysis.simulator.engine import Simulator
        t = workload.topology
        r = workload.router
        nodes = list(t.nodes())
        N = len(nodes)
        # Flat id for each node based on iteration order
        idx_of = {n: i for i, n in enumerate(nodes)}
        injections: list[Injection] = []
        phase_start = 0
        for phase_r in range(1, N):
            phase_flows: list[Flow] = []
            for src_node in nodes:
                dst_node = nodes[(idx_of[src_node] + phase_r) % N]
                # Find the matching workload flow
                phase_flows.append(Flow(src_node, dst_node, workload.msg_size))
            for f in phase_flows:
                injections.append(Injection(flow=f, start_step=phase_start))
            # Dry-run to compute when this phase drains
            sim = Simulator(t, r, phase_flows)
            for f in phase_flows:
                sim.inject(Injection(flow=f, start_step=0))
            phase_makespan = sim.run()
            phase_start += phase_makespan
        return injections
