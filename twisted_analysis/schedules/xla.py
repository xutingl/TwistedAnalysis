from __future__ import annotations
from dataclasses import dataclass

from twisted_analysis.model.flow import Flow, AllToAll
from twisted_analysis.schedules.base import Injection


@dataclass
class XLASchedule:
    """Replicates XLA's destination-core randomization for AllToAll lowering.

    Phase p in {0..N-2}: permute_idx = ((p*A) + B) mod (N-1) + 1
                         with A = 33617, B = 1299721.
    Phase p sends src -> (src + permute_idx) mod N for every src.

    Across all phases, every (src, dst) pair appears exactly once (because
    gcd(A, N-1) = 1 for our topologies, so the map p -> permute_idx is a
    bijection from {0..N-2} to {1..N-1}). Hence the set of phase workloads is
    a permutation of RoundRobin's.
    """
    name: str = "xla"
    A: int = 33617
    B: int = 1299721

    def emit(self, workload: AllToAll) -> list[Injection]:
        from twisted_analysis.simulator.engine import Simulator
        t = workload.topology
        r = workload.router
        nodes = list(t.nodes())
        N = len(nodes)
        idx_of = {n: i for i, n in enumerate(nodes)}

        injections: list[Injection] = []
        phase_start = 0
        for p in range(N - 1):
            permute_idx = ((p * self.A) + self.B) % (N - 1) + 1
            phase_flows: list[Flow] = []
            for src_node in nodes:
                dst_node = nodes[(idx_of[src_node] + permute_idx) % N]
                phase_flows.append(Flow(src_node, dst_node, workload.msg_size))
            for f in phase_flows:
                injections.append(Injection(flow=f, start_step=phase_start))
            sim = Simulator(t, r, phase_flows)
            for f in phase_flows:
                sim.inject(Injection(flow=f, start_step=0))
            phase_makespan = sim.run()
            phase_start += phase_makespan
        return injections
