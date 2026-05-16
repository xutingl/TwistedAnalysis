"""LNS CP-SAT repair: tests on a small cell."""
from __future__ import annotations
import tempfile, os
from collections import Counter

import pytest

pytest.importorskip("ortools")

from twisted_analysis.topology import Topology, ILPRouter
from twisted_analysis.io.routing_table import save_routing_table, load_routing_table
from twisted_analysis.schedules.cpsat_literal import cpsat_literal
from twisted_analysis.schedules.lns_cpsat import lns_cpsat_repair
from twisted_analysis.schedules.verify import verify_capacity, schedule_makespan


def _table_from_ilp_router(slice_):
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


def test_lns_returns_feasible_schedule_for_each_strategy():
    """Every destroy strategy must produce a capacity-feasible schedule."""
    t, table = _table_from_ilp_router((2, 4))
    lb = _physical_edge_lb(table, t.n_nodes)
    seed = cpsat_literal(t, table, t_upper=lb + 2, time_limit_s=60)
    for strat in ("time_window", "random_subset", "makespan_flows"):
        sch = lns_cpsat_repair(t, table, seed, n_iters=3,
                               per_subproblem_budget_s=30,
                               destroy_strategies=(strat,))
        assert verify_capacity(sch) == [], f"{strat}: violations"
        assert schedule_makespan(sch) <= schedule_makespan(seed), \
            f"{strat}: makespan increased ({schedule_makespan(sch)} > {schedule_makespan(seed)})"


def test_lns_improve_only_never_increases_makespan():
    """Across many iterations, the makespan must be monotone non-increasing."""
    t, table = _table_from_ilp_router((2, 4))
    lb = _physical_edge_lb(table, t.n_nodes)
    seed = cpsat_literal(t, table, t_upper=lb + 3, time_limit_s=60)
    history: list[int] = []
    def log(it, info):
        history.append(info["current_makespan"])
    sch = lns_cpsat_repair(t, table, seed, n_iters=10,
                           per_subproblem_budget_s=20,
                           rng_seed=42, log_fn=log)
    assert verify_capacity(sch) == []
    for i in range(1, len(history)):
        assert history[i] <= history[i - 1], \
            f"makespan increased at iter {i}: {history[i-1]} -> {history[i]}"


def test_lns_at_lb_stops_immediately():
    """When seeded with an LB-tight schedule, LNS must not increase makespan."""
    t, table = _table_from_ilp_router((2, 4))
    lb = _physical_edge_lb(table, t.n_nodes)
    seed = cpsat_literal(t, table, t_upper=lb, time_limit_s=60)
    assert schedule_makespan(seed) <= lb
    sch = lns_cpsat_repair(t, table, seed, n_iters=5,
                           per_subproblem_budget_s=10)
    assert verify_capacity(sch) == []
    assert schedule_makespan(sch) <= lb
