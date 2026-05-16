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


def test_cpsat_literal_warm_start_accepts_seed():
    """When warm-started with an optimal seed at the same t_upper, CP-SAT
    must return a feasible schedule (the hint should be respected when
    feasible, and the search at most matches the hint)."""
    t, table = _table_from_ilp_router((2, 4))
    lb = _physical_edge_lb(table, t.n_nodes)
    seed = cpsat_literal(t, table, t_upper=lb, time_limit_s=60)
    # Warm-start the next solve from the seed at the same t_upper.
    sch = cpsat_literal(t, table, t_upper=lb, time_limit_s=60,
                        warm_start_schedule=seed)
    assert verify_capacity(sch) == []
    assert schedule_makespan(sch) <= lb


def test_cpsat_literal_fixed_assignments_pins_flows():
    """fixed_assignments must force named flows to their required round
    (and the rest of the schedule must remain feasible)."""
    t, table = _table_from_ilp_router((2, 4))
    lb = _physical_edge_lb(table, t.n_nodes)
    seed = cpsat_literal(t, table, t_upper=lb, time_limit_s=60)
    # Pin every flow to its seed round; the model should accept and return it.
    fixed = {(e["src"], e["dst"]): e["round"] for e in seed}
    sch = cpsat_literal(t, table, t_upper=lb, time_limit_s=60,
                        fixed_assignments=fixed)
    by_key = {(e["src"], e["dst"]): e["round"] for e in sch}
    for k, r in fixed.items():
        assert by_key[k] == r, f"flow {k}: expected round {r}, got {by_key[k]}"


def test_cpsat_literal_fixed_assignments_infeasible_combination_raises():
    """If pinned flows conflict (two flows on same edge at same time), raise."""
    t, table = _table_from_ilp_router((2, 4))
    lb = _physical_edge_lb(table, t.n_nodes)
    # Find any two flows that share at least one physical edge at any hop.
    from collections import defaultdict
    edge_hops: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for s in range(t.n_nodes):
        for d in range(t.n_nodes):
            if s == d:
                continue
            path = table[s][d]
            for h in range(len(path) - 1):
                edge_hops[(path[h], path[h + 1])].append((s, d, h))
    # Pick a shared edge with at least 2 demands.
    shared = next(((e, demands) for e, demands in edge_hops.items()
                   if len(demands) >= 2), None)
    if shared is None:
        import pytest as _p
        _p.skip("no shared edge on this topology — skip conflict test")
    _edge, demands = shared
    (s1, d1, h1), (s2, d2, h2), *_ = demands
    # Pin both at the same edge-time: round_i = tau - h_i for tau = 0.
    tau = max(h1, h2)
    fixed = {(s1, d1): tau - h1, (s2, d2): tau - h2}
    import pytest as _p
    with _p.raises(RuntimeError, match="infeasible|no solution"):
        cpsat_literal(t, table, t_upper=lb, time_limit_s=30,
                      fixed_assignments=fixed)
