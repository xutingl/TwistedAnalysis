"""Exact ILP for literal-flow scheduling.

Variables:
  x[f, t] in {0, 1} for each (flow_idx, start_time) feasible pair.
  M (makespan) in {0, ..., T_upper}.

Constraints:
  (one start)  sum_t x[f, t] == 1                              for each f
  (makespan)   M >= sum_t (t + L_f) * x[f, t]                  for each f
  (capacity)   for each physical edge e and each time tau:
                  sum over (f, i) with path[f][i:i+2]==(u,v),
                  and t = tau - i in dom(f):
                      x[f, t] <= 1

Objective: minimize M.

Intractable for N >= 64; intended for small validation cells (2x4, 2x2x4)
and for cross-checking heuristic schedules.
"""
from __future__ import annotations
from collections import defaultdict

from twisted_analysis.topology import Topology


def _flow_set(table: list[list[list[int]]], n: int) -> list[tuple[int, int, list[int]]]:
    flows: list[tuple[int, int, list[int]]] = []
    for s in range(n):
        for d in range(n):
            if s == d:
                continue
            flows.append((s, d, list(table[s][d])))
    return flows


def _initial_upper_bound(flows: list[tuple[int, int, list[int]]]) -> int:
    """A safe makespan upper bound: total path-length sum (always feasible)."""
    return sum(len(p) - 1 for _, _, p in flows)


def ilp_literal(
    topology: Topology,
    table: list[list[list[int]]],
    *,
    t_upper: int | None = None,
    time_limit_s: int = 600,
    solver_msg: bool = False,
) -> list[dict]:
    """Solve the literal-flow scheduling ILP. Returns schedule entries.

    Raises ImportError if pulp is not installed.
    """
    try:
        import pulp
    except ImportError as exc:
        raise ImportError(
            "ilp_literal requires `pulp`. Install with `uv pip install pulp` "
            "or pick a different scheduler."
        ) from exc

    n = topology.n_nodes
    flows = _flow_set(table, n)
    if t_upper is None:
        # Use the literal_greedy makespan as a tighter upper bound.
        from twisted_analysis.schedules.literal_greedy import literal_greedy
        from twisted_analysis.schedules.verify import schedule_makespan
        t_upper = schedule_makespan(literal_greedy(topology, table, order="lpt"))

    prob = pulp.LpProblem("literal_flow_schedule", pulp.LpMinimize)
    M = pulp.LpVariable("M", lowBound=0, upBound=t_upper, cat="Integer")

    x: dict[tuple[int, int], "pulp.LpVariable"] = {}
    for f_idx, (_s, _d, path) in enumerate(flows):
        L = len(path) - 1
        for t in range(t_upper - L + 1):
            x[(f_idx, t)] = pulp.LpVariable(f"x_{f_idx}_{t}", cat="Binary")
        prob += pulp.lpSum(x[(f_idx, t)] for t in range(t_upper - L + 1)) == 1
        prob += M >= pulp.lpSum(
            (t + L) * x[(f_idx, t)] for t in range(t_upper - L + 1)
        )

    # Edge capacity.
    edge_demands: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for f_idx, (_s, _d, path) in enumerate(flows):
        for i in range(len(path) - 1):
            edge_demands[(path[i], path[i + 1])].append((f_idx, i))
    for _edge, demands in edge_demands.items():
        for tau in range(t_upper):
            terms = []
            for f_idx, i in demands:
                t = tau - i
                if (f_idx, t) in x:
                    terms.append(x[(f_idx, t)])
            if len(terms) >= 2:
                prob += pulp.lpSum(terms) <= 1

    prob += M

    solver = pulp.PULP_CBC_CMD(timeLimit=time_limit_s, msg=int(solver_msg))
    status = prob.solve(solver)
    if pulp.LpStatus[status] not in ("Optimal", "Not Solved"):
        # CBC returns "Not Solved" when it hits the time limit with a feasible
        # incumbent. Accept that case.
        if not any(pulp.value(v) is not None for v in x.values()):
            raise RuntimeError(
                f"ilp_literal: CBC returned status={pulp.LpStatus[status]}"
            )

    rounds: dict[tuple[int, int], int] = {}
    for f_idx, (src, dst, path) in enumerate(flows):
        L = len(path) - 1
        chosen: int | None = None
        for t in range(t_upper - L + 1):
            val = pulp.value(x[(f_idx, t)])
            if val is not None and val > 0.5:
                chosen = t
                break
        if chosen is None:
            raise RuntimeError(
                f"ilp_literal: no start assignment for flow ({src}->{dst}); "
                f"check t_upper={t_upper}, time_limit_s={time_limit_s}"
            )
        rounds[(src, dst)] = chosen

    entries: list[dict] = []
    for s in range(n):
        for d in range(n):
            if s == d:
                continue
            entries.append({
                "round": rounds[(s, d)],
                "src": s,
                "dst": d,
                "path": list(table[s][d]),
            })
    entries.sort(key=lambda e: (e["round"], e["src"]))
    return entries
