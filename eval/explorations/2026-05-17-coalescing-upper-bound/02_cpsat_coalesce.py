"""CP-SAT model: minimize coalesced DMA descriptor count subject to makespan + fixed routing.

Variables (Boolean):
  y[f, s]          = 1 iff flow f starts at round s (0 <= s <= t_upper - L_f)
  a[(u,v), tau]    = 1 iff physical edge (u, v) is active at round tau
  b[(u,v), tau]    = 1 iff (u, v) "starts" a new active run at tau
                     (i.e., active at tau and not at tau-1)

Constraints:
  - sum_s y[f, s] == 1                                      (exactly-one start)
  - a[e, tau] == sum over (f, h) with e in path(f) at hop h of y[f, tau - h]
                                                            (capacity = 1 enforces a in {0, 1})
  - b[e, tau] >= a[e, tau] - a[e, tau - 1]                  (break detection)
  - b[e, tau] <= a[e, tau]
  - b[e, tau] <= 1 - a[e, tau - 1]                          (for tau >= 1)
  - b[e, 0]   == a[e, 0]

Objective:
  - minimize sum_{e, tau} b[e, tau]                         (total coalesced descriptors)
"""
from __future__ import annotations
from collections import defaultdict


def _flow_set(table, n):
    flows = []
    for s in range(n):
        for d in range(n):
            if s == d:
                continue
            flows.append((s, d, list(table[s][d])))
    return flows


def cpsat_coalesce(
    topology,
    table,
    *,
    t_upper: int,
    time_limit_s: int = 600,
    n_workers: int = 8,
    solver_msg: bool = False,
):
    """Solve the coalescing-minimization model.

    Returns (entries, coalesced_count) where entries is a list of
    {round, src, dst, path} dicts and coalesced_count is the best-found
    objective value (or None on infeasibility).
    """
    from ortools.sat.python import cp_model

    n = topology.n_nodes
    flows = _flow_set(table, n)
    model = cp_model.CpModel()

    # y[f, s]
    y: dict[tuple[int, int], cp_model.IntVar] = {}
    flow_starts: dict[int, list[int]] = {}
    for f_idx, (_s, _d, path) in enumerate(flows):
        L = len(path) - 1
        starts = list(range(0, t_upper - L + 1))
        if not starts:
            raise RuntimeError(
                f"t_upper={t_upper} too small: flow {f_idx} has L={L}"
            )
        var_list = []
        for s in starts:
            v = model.NewBoolVar(f"y_{f_idx}_{s}")
            y[(f_idx, s)] = v
            var_list.append(v)
        flow_starts[f_idx] = starts
        model.AddExactlyOne(var_list)

    # edge_hops[(u, v)] = list of (f_idx, h)
    edge_hops: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for f_idx, (_s, _d, path) in enumerate(flows):
        for h in range(len(path) - 1):
            edge_hops[(path[h], path[h + 1])].append((f_idx, h))

    # a[edge, tau] and capacity (sum of contributing y == a)
    a: dict[tuple[tuple[int, int], int], cp_model.IntVar] = {}
    for edge, demands in edge_hops.items():
        for tau in range(t_upper):
            contributing = []
            for f_idx, h in demands:
                s = tau - h
                if (f_idx, s) in y:
                    contributing.append(y[(f_idx, s)])
            v = model.NewBoolVar(f"a_{edge[0]}_{edge[1]}_{tau}")
            a[(edge, tau)] = v
            if contributing:
                model.Add(v == sum(contributing))
            else:
                model.Add(v == 0)

    # b[edge, tau]: break = active and not previously active
    b_terms = []
    for edge in edge_hops:
        for tau in range(t_upper):
            bv = model.NewBoolVar(f"b_{edge[0]}_{edge[1]}_{tau}")
            if tau == 0:
                model.Add(bv == a[(edge, 0)])
            else:
                a_curr = a[(edge, tau)]
                a_prev = a[(edge, tau - 1)]
                model.Add(bv >= a_curr - a_prev)
                model.Add(bv <= a_curr)
                model.Add(bv <= 1 - a_prev)
            b_terms.append(bv)

    model.Minimize(sum(b_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = int(n_workers)
    solver.parameters.log_search_progress = bool(solver_msg)

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, None

    entries = []
    for f_idx, (s_node, d_node, path) in enumerate(flows):
        chosen_s = None
        for s in flow_starts[f_idx]:
            if solver.Value(y[(f_idx, s)]) == 1:
                chosen_s = s
                break
        assert chosen_s is not None, f"no start chosen for flow {f_idx}"
        entries.append({
            "round": chosen_s, "src": s_node, "dst": d_node, "path": path,
        })
    return entries, int(solver.ObjectiveValue())
