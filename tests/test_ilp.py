import pytest

from twisted_analysis.topology import Topology, Router
from twisted_analysis.model import AllToAll, Flow
from twisted_analysis.lp.ilp import solve_makespan


def test_single_flow_optimum_equals_path_length():
    t = Topology(slice=(2, 4))
    r = Router(t)
    f = Flow((0, 0), (0, 2), 1)
    workload = AllToAll(t, r, msg_size=1)
    # Override flows to a single one for this test
    workload_single = type(workload)(t, r, msg_size=1)
    # Use a tiny custom solve
    m_opt, _ = solve_makespan(t, r, [f], T_upper=8)
    assert m_opt == 2  # path length


def test_two_contending_flows_optimum_is_2():
    t = Topology(slice=(2, 4))
    r = Router(t)
    f1 = Flow((0, 0), (0, 1), 1)
    f2 = Flow((0, 0), (0, 1), 1)
    m_opt, _ = solve_makespan(t, r, [f1, f2], T_upper=8)
    assert m_opt == 2


def test_optimum_ge_lower_bound_2x4():
    t = Topology(slice=(2, 4))
    r = Router(t)
    w = AllToAll(t, r, msg_size=1)
    # Bound T_upper generously: LB * 4
    m_opt, _ = solve_makespan(t, r, list(w.flows), T_upper=4 * w.lower_bound)
    assert m_opt >= w.lower_bound
