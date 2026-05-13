from twisted_analysis.topology import Topology, Router
from twisted_analysis.model import AllToAll
from twisted_analysis.schedules.round_robin import RoundRobinSchedule
from twisted_analysis.simulator import Simulator


def test_round_robin_emits_n_minus_1_phases_worth():
    t = Topology(slice=(2, 4))
    r = Router(t)
    w = AllToAll(t, r, msg_size=1)
    sched = RoundRobinSchedule()
    injections = sched.emit(w)
    # N*(N-1) = 56 flows, all injected (each exactly once)
    assert len(injections) == 56


def test_round_robin_makespan_at_least_lb():
    t = Topology(slice=(2, 4))
    r = Router(t)
    w = AllToAll(t, r, msg_size=1)
    sched = RoundRobinSchedule()
    sim = Simulator(t, r, list(w.flows))
    for inj in sched.emit(w):
        sim.inject(inj)
    makespan = sim.run()
    assert makespan >= w.lower_bound


def test_round_robin_phases_dont_overlap():
    # Phase r's flows all share the same start_step.
    t = Topology(slice=(2, 4))
    r = Router(t)
    w = AllToAll(t, r, msg_size=1)
    sched = RoundRobinSchedule()
    injs = sched.emit(w)
    by_start = {}
    for inj in injs:
        by_start.setdefault(inj.start_step, []).append(inj)
    # 7 phases (N-1 = 7), each with N=8 flows
    assert len(by_start) == 7
    assert all(len(v) == 8 for v in by_start.values())
