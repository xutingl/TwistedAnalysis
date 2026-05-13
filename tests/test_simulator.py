from twisted_analysis.topology import Topology, Router
from twisted_analysis.model import AllToAll, Flow
from twisted_analysis.schedules.base import Injection
from twisted_analysis.simulator.engine import Simulator


def test_single_flow_one_hop():
    t = Topology(slice=(2, 4))
    r = Router(t)
    f = Flow((0, 0), (0, 1), 1)
    sim = Simulator(t, r, [f])
    sim.inject(Injection(flow=f, start_step=0))
    makespan = sim.run()
    assert makespan == 1


def test_single_flow_multi_hop():
    # slice=(4, 8): dim-1 ring of 8; (0,0)->(0,3) is 3 forward hops, no wrap.
    t = Topology(slice=(4, 8))
    r = Router(t)
    f = Flow((0, 0), (0, 3), 1)
    sim = Simulator(t, r, [f])
    sim.inject(Injection(flow=f, start_step=0))
    makespan = sim.run()
    assert makespan == 3  # 3 hops along row


def test_msg_size_2_pipelines():
    # Two units, same path of length 2; pipelined → 3 steps total (S&F).
    t = Topology(slice=(2, 4))
    r = Router(t)
    f = Flow((0, 0), (0, 2), 2)
    sim = Simulator(t, r, [f])
    sim.inject(Injection(flow=f, start_step=0))
    makespan = sim.run()
    assert makespan == 3


def test_two_flows_contend_on_first_hop():
    # Both flows start at (0, 0) and must use the same first link.
    # Second flow waits one step.
    t = Topology(slice=(2, 4))
    r = Router(t)
    f1 = Flow((0, 0), (0, 1), 1)
    f2 = Flow((0, 0), (0, 1), 1)
    sim = Simulator(t, r, [f1, f2])
    sim.inject(Injection(flow=f1, start_step=0, priority=0))
    sim.inject(Injection(flow=f2, start_step=0, priority=1))
    makespan = sim.run()
    assert makespan == 2


def test_makespan_at_least_lower_bound():
    t = Topology(slice=(2, 4))
    r = Router(t)
    w = AllToAll(t, r, msg_size=1)
    sim = Simulator(t, r, list(w.flows))
    for f in w.flows:
        sim.inject(Injection(flow=f, start_step=0))
    makespan = sim.run()
    assert makespan >= w.lower_bound
