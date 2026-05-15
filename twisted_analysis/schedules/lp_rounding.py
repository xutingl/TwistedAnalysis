"""LP relaxation of the literal scheduling ILP + randomized rounding.

Steps:
  1. Build the literal_ilp LP (no integrality constraint on x[f,t]).
  2. Solve once. Read out fractional values x_lp[f, t] ∈ [0, 1].
  3. For each trial: for each flow f, sample start time t from
     distribution x_lp[f, ·] (clipped+renormalized). Build raw schedule.
  4. Repair feasibility: greedily walk flows in increasing chosen start;
     if a flow conflicts on any edge-time, shift its start to the smallest
     later time with no conflict (this is the literal_greedy repair).
  5. Return the trial whose final makespan is smallest.

Polynomial-time. No LB-tightness guarantee, but in practice close to LP
bound (which equals the LP relaxation's optimum, a valid lower bound)."""
from __future__ import annotations
import random
from collections import defaultdict

from twisted_analysis.topology import Topology


def _flow_set(table, n):
    flows = []
    for s in range(n):
        for d in range(n):
            if s == d:
                continue
            flows.append((s, d, list(table[s][d])))
    return flows


def _solve_lp(flows, t_upper):
    """Return: dict (f_idx -> list[float] of length len(starts))."""
    import pulp
    prob = pulp.LpProblem("literal_lp_relaxation", pulp.LpMinimize)
    M = pulp.LpVariable("M", lowBound=0, upBound=t_upper, cat="Continuous")
    x = {}
    for f_idx, (_s, _d, path) in enumerate(flows):
        L = len(path) - 1
        starts = list(range(0, t_upper - L + 1))
        for s in starts:
            x[(f_idx, s)] = pulp.LpVariable(f"x_{f_idx}_{s}",
                                            lowBound=0, upBound=1,
                                            cat="Continuous")
        prob += pulp.lpSum(x[(f_idx, s)] for s in starts) == 1
        prob += M >= pulp.lpSum((s + L) * x[(f_idx, s)] for s in starts)

    edge_hops = defaultdict(list)
    for f_idx, (_s, _d, path) in enumerate(flows):
        for h in range(len(path) - 1):
            edge_hops[(path[h], path[h + 1])].append((f_idx, h))
    for _e, demands in edge_hops.items():
        for tau in range(t_upper):
            terms = []
            for f_idx, h in demands:
                s = tau - h
                if (f_idx, s) in x:
                    terms.append(x[(f_idx, s)])
            if len(terms) >= 2:
                prob += pulp.lpSum(terms) <= 1
    prob += M
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    out = {}
    for f_idx, (_s, _d, path) in enumerate(flows):
        L = len(path) - 1
        vals = [max(0.0, pulp.value(x[(f_idx, s)]) or 0.0)
                for s in range(0, t_upper - L + 1)]
        z = sum(vals)
        if z > 0:
            vals = [v / z for v in vals]
        else:
            vals = [1.0 / len(vals)] * len(vals) if vals else [1.0]
        out[f_idx] = vals
    return out, pulp.value(M)


def _sample_round(flows, x_lp, t_upper, rng):
    edge_busy: dict[tuple[int, int], set[int]] = defaultdict(set)
    rounds = {}
    # Sample start per flow.
    starts = [None] * len(flows)
    for f_idx, (_s, _d, path) in enumerate(flows):
        L = len(path) - 1
        probs = x_lp[f_idx]
        # rng.choices returns a list of 1 element.
        choice = rng.choices(range(len(probs)), weights=probs, k=1)[0]
        starts[f_idx] = choice
    # Repair: process in increasing chosen start, displace conflicts forward.
    order = sorted(range(len(flows)), key=lambda i: starts[i])
    for f_idx in order:
        path = flows[f_idx][2]
        L = len(path) - 1
        start = starts[f_idx]
        while True:
            conflict = False
            for h in range(L):
                if (start + h) in edge_busy[(path[h], path[h + 1])]:
                    conflict = True
                    break
            if not conflict:
                break
            start += 1
        for h in range(L):
            edge_busy[(path[h], path[h + 1])].add(start + h)
        rounds[f_idx] = start
    return rounds


def _makespan(flows, rounds):
    m = 0
    for f_idx, (_s, _d, path) in enumerate(flows):
        L = len(path) - 1
        f = rounds[f_idx] + L
        if f > m:
            m = f
    return m


def lp_rounding(
    topology: Topology,
    table: list[list[list[int]]],
    *,
    t_upper: int,
    n_trials: int = 100,
    seed: int = 0,
) -> list[dict]:
    """Solve LP relaxation, randomly round, repair, take best of n_trials.

    Raises ImportError if pulp is missing.
    """
    try:
        import pulp  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "lp_rounding requires pulp. Install: `uv pip install pulp`."
        ) from exc

    n = topology.n_nodes
    flows = _flow_set(table, n)
    x_lp, lp_m = _solve_lp(flows, t_upper)

    best_rounds = None
    best_m = None
    rng = random.Random(seed)
    for _ in range(n_trials):
        rounds = _sample_round(flows, x_lp, t_upper, rng)
        m = _makespan(flows, rounds)
        if best_m is None or m < best_m:
            best_m = m
            best_rounds = rounds

    entries = []
    for f_idx, (src, dst, path) in enumerate(flows):
        entries.append({
            "round": int(best_rounds[f_idx]),
            "src": src, "dst": dst,
            "path": list(path),
        })
    entries.sort(key=lambda e: (e["round"], e["src"]))
    return entries
