from __future__ import annotations

import pulp

from twisted_analysis.topology import Topology, Router, DirectedLink
from twisted_analysis.model.flow import Flow
from twisted_analysis.lp.ilp import _unroll_units


def lp_relax_lower_bound(
    topology: Topology,
    router: Router,
    flows: list[Flow],
    T_upper: int,
    solver_name: str = "PULP_CBC_CMD",
) -> float:
    """LP relaxation of the time-indexed makespan ILP. Fractional x in [0, 1].

    Returns the optimal LP objective, which is the minimum *expected* completion
    time across units (i.e. sum_t (t+1) * x[u, last, t]) — guaranteed to be <=
    the ILP optimum, but with fractional x it can be looser than the link-load
    lower bound. Treat this as `LP_relax <= ILP_opt`; do NOT assume it is tighter
    than `AllToAll.lower_bound`. A truly makespan-tight LP relaxation requires a
    different (and more expensive) formulation; out of scope for v1.
    """
    units = _unroll_units(router, flows)
    if not units:
        return 0.0
    T = T_upper
    prob = pulp.LpProblem("twisted_alltoall_relax", pulp.LpMinimize)
    M = pulp.LpVariable("M", lowBound=0)
    x: dict[tuple[int, int, int], pulp.LpVariable] = {}
    for u in units:
        for i in range(len(u.path)):
            for t in range(T):
                x[(u.unit_id, i, t)] = pulp.LpVariable(
                    f"x_{u.unit_id}_{i}_{t}", lowBound=0, upBound=1
                )
    # Per-hop fire-once
    for u in units:
        for i in range(len(u.path)):
            prob += pulp.lpSum(x[(u.unit_id, i, t)] for t in range(T)) == 1
    # Causal order
    for u in units:
        for i in range(len(u.path) - 1):
            for s in range(T):
                prob += (
                    pulp.lpSum(x[(u.unit_id, i + 1, t)] for t in range(s + 1))
                    <= pulp.lpSum(x[(u.unit_id, i, t)] for t in range(s))
                )
    # Link capacity
    edge_to_uses: dict[DirectedLink, list[tuple[int, int]]] = {}
    for u in units:
        for i, e in enumerate(u.path):
            edge_to_uses.setdefault(e, []).append((u.unit_id, i))
    for e, uses in edge_to_uses.items():
        for t in range(T):
            prob += pulp.lpSum(x[(uid, i, t)] for uid, i in uses) <= 1
    # Makespan: last-hop firing time bounded by M
    for u in units:
        last = len(u.path) - 1
        prob += pulp.lpSum((t + 1) * x[(u.unit_id, last, t)] for t in range(T)) <= M
    prob += M
    prob.solve(pulp.getSolver(solver_name, msg=False))
    return float(pulp.value(M))
