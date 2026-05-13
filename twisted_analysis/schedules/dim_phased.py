from __future__ import annotations
from dataclasses import dataclass

from twisted_analysis.model.flow import Flow, AllToAll
from twisted_analysis.schedules.base import Injection, Schedule


@dataclass
class DimPhasedSchedule:
    """Dimension-ordered phased AllToAll: one phase per dim, largest-dim first.

    Phase d's flows: every (src, dst) pair that differs only in dim d.
    Each phase uses only dim-d links; phases don't contend.
    """
    name: str = "dim_phased"

    def emit(self, workload: AllToAll) -> list[Injection]:
        from twisted_analysis.simulator.engine import Simulator
        t = workload.topology
        r = workload.router
        # Phase ordering: largest dim first (default DOR order)
        dim_order = sorted(range(t.ndim), key=lambda d: -t.slice[d])
        injections: list[Injection] = []
        phase_start = 0
        for d in dim_order:
            phase_flows: list[Flow] = []
            for src in t.nodes():
                for dst in t.nodes():
                    if src == dst:
                        continue
                    # Only pairs that differ exactly in dim d
                    if all(src[i] == dst[i] for i in range(t.ndim) if i != d) \
                            and src[d] != dst[d]:
                        phase_flows.append(Flow(src, dst, workload.msg_size))
            for f in phase_flows:
                injections.append(Injection(flow=f, start_step=phase_start))
            # Dry-run for phase makespan
            sim = Simulator(t, r, phase_flows)
            for f in phase_flows:
                sim.inject(Injection(flow=f, start_step=0))
            phase_makespan = sim.run()
            phase_start += phase_makespan
        return injections
