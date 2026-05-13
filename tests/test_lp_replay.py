from twisted_analysis.topology import Topology, Router
from twisted_analysis.model import AllToAll
from twisted_analysis.lp.ilp import solve_makespan
from twisted_analysis.schedules.lp_optimal import lp_assignment_to_injections
from twisted_analysis.simulator import Simulator


def test_lp_assignment_replays_to_same_makespan_2x4():
    t = Topology(slice=(2, 4))
    r = Router(t)
    w = AllToAll(t, r, msg_size=1)
    m_opt, assignment = solve_makespan(t, r, list(w.flows), T_upper=w.lower_bound * 4)
    flows_list = list(w.flows)
    injs = lp_assignment_to_injections(flows_list, r, assignment)
    sim = Simulator(t, r, flows_list)
    for inj in injs:
        sim.inject(inj)
    sim_makespan = sim.run()
    # Simulator should reproduce the LP makespan exactly under the LP-derived priorities.
    assert sim_makespan == m_opt
