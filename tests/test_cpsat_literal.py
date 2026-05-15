"""CP-SAT literal scheduler: tests on small cells with known optima."""
from __future__ import annotations
from collections import Counter

import pytest

pytest.importorskip("ortools")

from twisted_analysis.topology import Topology, ILPRouter
from twisted_analysis.schedules.cpsat_literal import cpsat_literal
from twisted_analysis.schedules.verify import verify_capacity, schedule_makespan


def _table_from_ilp_router(slice_):
    import tempfile, os
    from twisted_analysis.io.routing_table import save_routing_table, load_routing_table
    t = Topology(slice=slice_)
    r = ILPRouter(t)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name
    try:
        save_routing_table(t, r, tmp_path)
        table = load_routing_table(tmp_path)
    finally:
        os.unlink(tmp_path)
    return t, table


def _physical_edge_lb(table, n):
    c: Counter = Counter()
    for s in range(n):
        for d in range(n):
            if s == d:
                continue
            path = table[s][d]
            for h in range(len(path) - 1):
                c[(path[h], path[h + 1])] += 1
    return max(c.values())


def test_cpsat_literal_2x4_ilp_lb_tight():
    """On (2,4) with ILP routing, LB is achievable at t_upper=LB."""
    t, table = _table_from_ilp_router((2, 4))
    lb = _physical_edge_lb(table, t.n_nodes)
    sch = cpsat_literal(t, table, t_upper=lb, time_limit_s=60)
    assert verify_capacity(sch) == []
    assert schedule_makespan(sch) <= lb


def test_cpsat_literal_infeasible_raises():
    """At t_upper < LB the model must be infeasible."""
    t, table = _table_from_ilp_router((2, 4))
    lb = _physical_edge_lb(table, t.n_nodes)
    with pytest.raises(RuntimeError, match="infeasible|no solution"):
        cpsat_literal(t, table, t_upper=lb - 1, time_limit_s=30)
