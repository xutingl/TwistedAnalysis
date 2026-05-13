from twisted_analysis.topology import Topology, DORRouter
from twisted_analysis.model import AllToAll
from twisted_analysis.schedules.round_robin import RoundRobinSchedule
from twisted_analysis.simulator import Simulator
from twisted_analysis.simulator.instrumentation import collect_idle_trace, gantt_log


def test_idle_trace_returns_dict_keyed_by_edge():
    t = Topology(slice=(2, 4))
    r = DORRouter(t)
    w = AllToAll(t, r, msg_size=1)
    sched = RoundRobinSchedule()
    sim = Simulator(t, r, list(w.flows), record_history=True)
    for inj in sched.emit(w):
        sim.inject(inj)
    sim.run()
    trace = collect_idle_trace(sim, bottleneck_edges=w.bottleneck_edges())
    assert all(isinstance(v, int) and v >= 0 for v in trace.values())


def test_gantt_log_has_one_row_per_unit_per_hop():
    t = Topology(slice=(2, 4))
    r = DORRouter(t)
    w = AllToAll(t, r, msg_size=1)
    sched = RoundRobinSchedule()
    sim = Simulator(t, r, list(w.flows), record_history=True)
    for inj in sched.emit(w):
        sim.inject(inj)
    sim.run()
    log = gantt_log(sim)
    expected_rows = sum(len(sim.path_map[f]) for f in w.flows)
    assert len(log) == expected_rows
