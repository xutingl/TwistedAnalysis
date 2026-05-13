from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

import pulp

from twisted_analysis.topology import Topology, Router, DirectedLink
from twisted_analysis.model.flow import Flow


@dataclass
class UnitPath:
    """One unit of payload with its own path. Multiple units may share a flow."""
    unit_id: int
    flow: Flow
    path: tuple[DirectedLink, ...]


def _unroll_units(router: Router, flows: Iterable[Flow]) -> list[UnitPath]:
    units: list[UnitPath] = []
    uid = 0
    for f in flows:
        p = router.path(f.src, f.dst)
        for _ in range(f.size):
            units.append(UnitPath(uid, f, p))
            uid += 1
    return units


def _solve_feasibility(
    units: list[UnitPath], T: int, msg_solver: pulp.LpSolver
) -> dict[tuple[int, int, int], float] | None:
    """Returns assignment x[unit, i, t] -> {0, 1} if feasible, else None."""
    prob = pulp.LpProblem("twisted_alltoall", pulp.LpMinimize)
    x: dict[tuple[int, int, int], pulp.LpVariable] = {}
    for u in units:
        for i in range(len(u.path)):
            for t in range(T):
                x[(u.unit_id, i, t)] = pulp.LpVariable(
                    f"x_{u.unit_id}_{i}_{t}", cat=pulp.LpBinary
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
    # Trivial objective (feasibility)
    prob += 0
    prob.solve(msg_solver)
    if pulp.LpStatus[prob.status] != "Optimal":
        return None
    return {k: pulp.value(v) for k, v in x.items()}


def solve_makespan(
    topology: Topology,
    router: Router,
    flows: list[Flow],
    T_upper: int,
    solver_name: str = "PULP_CBC_CMD",
) -> tuple[int, dict[tuple[int, int, int], float]]:
    """Binary search on makespan T. Returns (optimal_T, optimal_assignment).

    Lower bound for the search = max link load.
    """
    units = _unroll_units(router, flows)
    if not units:
        return 0, {}
    # Compute LB
    from collections import Counter
    c: Counter[DirectedLink] = Counter()
    for u in units:
        for e in u.path:
            c[e] += 1
    lb = max(c.values()) if c else 0
    solver = pulp.getSolver(solver_name, msg=False)
    lo, hi = lb, T_upper
    best_assignment: dict[tuple[int, int, int], float] = {}
    # First confirm hi is feasible (else widen)
    while True:
        a = _solve_feasibility(units, hi, solver)
        if a is not None:
            best_assignment = a
            break
        hi *= 2
        if hi > 1_000_000:
            raise RuntimeError("ILP T_upper grew past 1e6; check formulation")
    # Binary search
    while lo < hi:
        mid = (lo + hi) // 2
        a = _solve_feasibility(units, mid, solver)
        if a is not None:
            hi = mid
            best_assignment = a
        else:
            lo = mid + 1
    return lo, best_assignment
