"""CP-SAT feasibility / minimization on the literal flow set.

Variables (Boolean):
  y[f, s]  = 1 iff flow f is injected at time s.
              Domain of s: 0 <= s <= t_upper - L_f.

Constraints:
  - sum_s y[f, s] == 1                                    (exactly-one start)
  - for each (edge e, time tau): at-most-1 over (f, s)    (capacity)
       such that path(f) contains e at hop h with s + h == tau.

Objective (when minimize=True):
  - minimize the makespan M, where M >= s + L_f for the chosen y[f, s].

CP-SAT handles pseudo-Boolean and at-most-one constraints natively, which
is much more efficient than CBC on this structure. Use this when literal_ilp
(pulp + CBC) times out.
"""
from __future__ import annotations
from collections import defaultdict


def _flow_set(table: list[list[list[int]]], n: int):
    flows = []
    for s in range(n):
        for d in range(n):
            if s == d:
                continue
            flows.append((s, d, list(table[s][d])))
    return flows


def cpsat_literal(
    topology,
    table: list[list[list[int]]],
    *,
    t_upper: int,
    time_limit_s: int = 600,
    solver_msg: bool = False,
    n_workers: int = 8,
    minimize: bool = True,
) -> list[dict]:
    """Solve / feasibility-probe the literal scheduling problem with CP-SAT.

    Raises ImportError if ortools is not installed.
    Raises RuntimeError if the problem is infeasible at the given `t_upper`.
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise ImportError(
            "cpsat_literal requires ortools. Install: `uv pip install ortools`."
        ) from exc

    n = topology.n_nodes
    flows = _flow_set(table, n)

    model = cp_model.CpModel()

    y: dict[tuple[int, int], cp_model.IntVar] = {}
    for f_idx, (_s, _d, path) in enumerate(flows):
        L = len(path) - 1
        starts = list(range(0, t_upper - L + 1))
        if not starts:
            raise RuntimeError(
                f"t_upper={t_upper} too small: flow {f_idx} has L={L}, needs t_upper>=L+1"
            )
        var_list = []
        for s in starts:
            v = model.NewBoolVar(f"y_{f_idx}_{s}")
            y[(f_idx, s)] = v
            var_list.append(v)
        model.AddExactlyOne(var_list)

    # Edge capacity: for each (edge, tau), at most one (f, s) with path[f] containing edge at hop h, s+h=tau.
    edge_hops: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for f_idx, (_s, _d, path) in enumerate(flows):
        for h in range(len(path) - 1):
            edge_hops[(path[h], path[h + 1])].append((f_idx, h))

    for _edge, demands in edge_hops.items():
        for tau in range(t_upper):
            vars_here = []
            for f_idx, h in demands:
                s = tau - h
                if (f_idx, s) in y:
                    vars_here.append(y[(f_idx, s)])
            if len(vars_here) >= 2:
                model.AddAtMostOne(vars_here)

    if minimize:
        M = model.NewIntVar(0, t_upper, "M")
        for f_idx, (_s, _d, path) in enumerate(flows):
            L = len(path) - 1
            for s in range(0, t_upper - L + 1):
                model.Add(M >= s + L).OnlyEnforceIf(y[(f_idx, s)])
        model.Minimize(M)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = int(n_workers)
    if solver_msg:
        solver.parameters.log_search_progress = True

    status = solver.Solve(model)

    if status in (cp_model.INFEASIBLE,):
        raise RuntimeError(f"cpsat_literal: infeasible at t_upper={t_upper}")
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(
            f"cpsat_literal: solver returned status={status} (no incumbent found "
            f"within time_limit_s={time_limit_s})"
        )

    rounds: dict[tuple[int, int], int] = {}
    for f_idx, (src, dst, path) in enumerate(flows):
        L = len(path) - 1
        chosen = None
        for s in range(0, t_upper - L + 1):
            if solver.Value(y[(f_idx, s)]) == 1:
                chosen = s
                break
        if chosen is None:
            raise RuntimeError(f"cpsat_literal: no start picked for flow {f_idx}")
        rounds[(src, dst)] = chosen

    entries = []
    for s in range(n):
        for d in range(n):
            if s == d:
                continue
            entries.append({
                "round": rounds[(s, d)],
                "src": s, "dst": d,
                "path": list(table[s][d]),
            })
    entries.sort(key=lambda e: (e["round"], e["src"]))
    return entries
