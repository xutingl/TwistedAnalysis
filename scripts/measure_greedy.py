"""Quick measurement: OrbitGreedy vs PipelinedOrbit on every (topology, router)."""
from __future__ import annotations
import time

from twisted_analysis.model import AllToAll
from twisted_analysis.schedules import OrbitGreedySchedule, PipelinedOrbitSchedule
from twisted_analysis.simulator import Simulator
from twisted_analysis.topology import DORRouter, ILPRouter, Topology


CELLS = [
    ((2, 4), "dor"),
    ((2, 4), "ilp"),
    ((4, 8), "dor"),
    ((4, 8), "ilp"),
    ((4, 4, 8), "dor"),
    ((4, 4, 8), "ilp"),
]
SCHEDS = [
    ("orbit_greedy_lpt", OrbitGreedySchedule(order="lpt")),
    ("orbit_greedy_spt", OrbitGreedySchedule(order="spt")),
    ("pipelined_orbit_lpt", PipelinedOrbitSchedule(order="lpt")),
    ("pipelined_orbit_spt", PipelinedOrbitSchedule(order="spt")),
]


def run() -> None:
    print(f"{'slice':<10} {'router':<5} {'sched':<22} {'LB':>5} {'makespan':>10} {'ratio':>7} {'time_s':>7}")
    print("-" * 72)
    for slice_, router_name in CELLS:
        t = Topology(slice=slice_)
        r = ILPRouter(t) if router_name == "ilp" else DORRouter(t)
        w = AllToAll(t, r, msg_size=1)
        lb = w.lower_bound
        for label, sched in SCHEDS:
            t0 = time.perf_counter()
            injs = sched.emit(w)
            sim = Simulator(t, r, list(w.flows))
            for inj in injs:
                sim.inject(inj)
            makespan = sim.run()
            dt = time.perf_counter() - t0
            print(f"{str(slice_):<10} {router_name:<5} {label:<22} "
                  f"{lb:>5} {makespan:>10} {makespan/lb:>7.3f} {dt:>7.2f}")


if __name__ == "__main__":
    run()
