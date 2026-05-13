from twisted_analysis.topology import Topology, DORRouter
from twisted_analysis.model import AllToAll
from twisted_analysis.lp.relaxation import lp_relax_lower_bound
from twisted_analysis.lp.ilp import solve_makespan


def test_lp_relax_is_ge_link_lb_2x4():
    t = Topology(slice=(2, 4))
    r = DORRouter(t)
    w = AllToAll(t, r, msg_size=1)
    m_lp = lp_relax_lower_bound(t, r, list(w.flows), T_upper=w.lower_bound * 4)
    m_ilp, _ = solve_makespan(t, r, list(w.flows), T_upper=w.lower_bound * 4)
    # LP relaxation gives a lower bound on the ILP optimum
    assert m_lp <= m_ilp + 1e-6
