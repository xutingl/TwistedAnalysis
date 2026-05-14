"""Translation-symmetry-reduced scheduling ILP.

Groups AllToAll flows into translation orbits (N-1 of them, each of size N)
and solves for one schedule per orbit. Variable count: O((N-1) * path_len * T)
instead of O(N(N-1) * path_len * T) — a factor-N reduction.

Symmetry assumption: all N members of an orbit fire their hop i at the same
time. By a Birkhoff-style argument on vertex-transitive workloads, the optimal
makespan under this constraint equals the unconstrained optimum.
"""
from __future__ import annotations
from collections import Counter, defaultdict

import pulp

from twisted_analysis.topology import Topology, Router, DirectedLink
from twisted_analysis.model.flow import Flow
from twisted_analysis.lp.orbit import compute_orbits


def _solve_feasibility_sym(
    topology: Topology,
    router: Router,
    orbits: dict,
    T: int,
    msg_solver: pulp.LpSolver,
) -> dict | None:
    """Returns y[orbit, hop, t] -> 0/1 if feasible, else None."""
    prob = pulp.LpProblem("twisted_alltoall_sym", pulp.LpMinimize)

    # Canonical path per orbit (use the canonical src=origin member's path).
    origin = tuple([0] * topology.ndim)
    canonical_path: dict = {}
    for orbit_id, members in orbits.items():
        canon = next(((s, d) for (s, d) in members if s == origin), None)
        assert canon is not None
        canonical_path[orbit_id] = router.path(canon[0], canon[1])

    y: dict = {}
    for orbit_id, path in canonical_path.items():
        for i in range(len(path)):
            for t in range(T):
                y[(orbit_id, i, t)] = pulp.LpVariable(
                    f"y_{orbit_id}_{i}_{t}", cat=pulp.LpBinary
                )

    # Per-orbit fire-once
    for orbit_id, path in canonical_path.items():
        for i in range(len(path)):
            prob += pulp.lpSum(y[(orbit_id, i, t)] for t in range(T)) == 1

    # Per-orbit causal order: hop i+1 fires by step s <= hop i fired by step s-1
    for orbit_id, path in canonical_path.items():
        for i in range(len(path) - 1):
            for s in range(T):
                prob += (
                    pulp.lpSum(y[(orbit_id, i + 1, t)] for t in range(s + 1))
                    <= pulp.lpSum(y[(orbit_id, i, t)] for t in range(s))
                )

    # Edge-orbit capacity: for each (dim, dir, time), at most 1 orbit firing.
    edge_orbit_hits: dict = defaultdict(list)
    for orbit_id, path in canonical_path.items():
        for i, (_, _, dim, dir) in enumerate(path):
            edge_orbit_hits[(dim, dir)].append((orbit_id, i))
    for (dim, dir), hits in edge_orbit_hits.items():
        for t in range(T):
            prob += pulp.lpSum(
                y[(orbit_id, i, t)] for (orbit_id, i) in hits
            ) <= 1

    # Feasibility (no real objective; minimize 0)
    prob += 0
    prob.solve(msg_solver)
    if pulp.LpStatus[prob.status] != "Optimal":
        return None
    return {k: pulp.value(v) for k, v in y.items()}


def solve_symmetric_makespan(
    topology: Topology,
    router: Router,
    flows: list[Flow],
    T_upper: int,
    solver_name: str = "PULP_CBC_CMD",
    time_limit_seconds: int | None = None,
) -> tuple[int, dict]:
    """Binary-search makespan T for the symmetric scheduling ILP.

    `time_limit_seconds`: per-feasibility-check CBC wall-clock cap. None = no cap.
    """
    orbits = compute_orbits(topology)

    # Compute LB over edge orbits: how many times does each (dim, dir) class
    # appear across all orbits' canonical paths.
    origin = tuple([0] * topology.ndim)
    edge_orbit_load: Counter = Counter()
    for orbit_id, members in orbits.items():
        canon = next(((s, d) for (s, d) in members if s == origin), None)
        if canon is None:
            continue
        path = router.path(*canon)
        for _, _, dim, dir in path:
            edge_orbit_load[(dim, dir)] += 1
    lb = max(edge_orbit_load.values()) if edge_orbit_load else 0

    solver_kwargs = {"msg": False}
    if time_limit_seconds is not None:
        solver_kwargs["timeLimit"] = int(time_limit_seconds)
    solver = pulp.getSolver(solver_name, **solver_kwargs)
    lo, hi = lb, T_upper
    best_assignment: dict = {}
    while True:
        a = _solve_feasibility_sym(topology, router, orbits, hi, solver)
        if a is not None:
            best_assignment = a
            break
        hi *= 2
        if hi > 1_000_000:
            raise RuntimeError("Symmetric ILP T_upper grew past 1e6")
    while lo < hi:
        mid = (lo + hi) // 2
        a = _solve_feasibility_sym(topology, router, orbits, mid, solver)
        if a is not None:
            hi = mid
            best_assignment = a
        else:
            lo = mid + 1
    return lo, best_assignment
