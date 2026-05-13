from twisted_analysis.topology import Topology, DORRouter
from twisted_analysis.model import AllToAll
from twisted_analysis.lp.symmetric import solve_symmetric_makespan
from twisted_analysis.lp.ilp import solve_makespan


def test_symmetric_matches_asymmetric_2x4():
    """On 2x4, the symmetric ILP optimum should match the asymmetric ILP."""
    t = Topology(slice=(2, 4))
    r = DORRouter(t)
    w = AllToAll(t, r, msg_size=1)
    m_sym, _ = solve_symmetric_makespan(t, r, list(w.flows), T_upper=w.lower_bound * 4)
    m_asym, _ = solve_makespan(t, r, list(w.flows), T_upper=w.lower_bound * 4)
    assert m_sym == m_asym


def test_symmetric_optimum_ge_lb_2x4():
    t = Topology(slice=(2, 4))
    r = DORRouter(t)
    w = AllToAll(t, r, msg_size=1)
    m_sym, _ = solve_symmetric_makespan(t, r, list(w.flows), T_upper=w.lower_bound * 4)
    assert m_sym >= w.lower_bound
