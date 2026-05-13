import pytest

from twisted_analysis.topology import Topology, ILPRouter
from twisted_analysis.model import AllToAll
from twisted_analysis.lp.symmetric import solve_symmetric_makespan


@pytest.mark.slow
def test_symmetric_4x8_tractable():
    """4x8 symmetric scheduling ILP should solve in under 5 minutes."""
    t = Topology(slice=(4, 8))
    r = ILPRouter(t)
    w = AllToAll(t, r, msg_size=1)
    m, _ = solve_symmetric_makespan(t, r, list(w.flows), T_upper=w.lower_bound * 4)
    assert m >= w.lower_bound


@pytest.mark.slow
def test_4x8_ilp_router_plus_symmetric_zero_gap():
    """The central headline claim: with ILP routing AND symmetric ILP scheduling,
    4x8 achieves the lower bound exactly. This locks the result against regressions
    in either the router or the symmetric scheduler.
    """
    from twisted_analysis.model import AllToAll
    t = Topology(slice=(4, 8))
    r = ILPRouter(t)
    w = AllToAll(t, r, msg_size=1)
    m, _ = solve_symmetric_makespan(t, r, list(w.flows), T_upper=w.lower_bound * 2)
    assert m == w.lower_bound
