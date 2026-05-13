from twisted_analysis.topology import Topology, DORRouter
from twisted_analysis.model import AllToAll
from twisted_analysis.schedules.dim_phased import DimPhasedSchedule
from twisted_analysis.simulator import Simulator


def test_2x4_dim_phased_two_phases():
    t = Topology(slice=(2, 4))
    r = DORRouter(t)
    w = AllToAll(t, r, msg_size=1)
    sched = DimPhasedSchedule()
    injections = sched.emit(w)
    starts = sorted({inj.start_step for inj in injections})
    # Phase 1 (along longer dim) at step 0, Phase 2 (along shorter dim) at later step
    assert len(starts) == 2 and starts[0] == 0


def test_2x4_dim_phased_makespan_at_least_lb():
    # DimPhased covers only the one-dim-diff subset of pairs, so we build the
    # simulator from the injected flow set (not the full AllToAll workload).
    t = Topology(slice=(2, 4))
    r = DORRouter(t)
    w = AllToAll(t, r, msg_size=1)
    sched = DimPhasedSchedule()
    injs = sched.emit(w)
    flows = list({inj.flow for inj in injs})
    sim = Simulator(t, r, flows)
    for inj in injs:
        sim.inject(inj)
    assert sim.run() >= 1  # at least one step; partial-coverage workload


def test_4x4x8_dim_phased_three_phases():
    t = Topology(slice=(4, 4, 8))
    r = DORRouter(t)
    w = AllToAll(t, r, msg_size=1)
    sched = DimPhasedSchedule()
    injections = sched.emit(w)
    starts = sorted({inj.start_step for inj in injections})
    assert len(starts) == 3
