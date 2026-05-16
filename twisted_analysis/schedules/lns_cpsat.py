"""Large-Neighborhood Search (LNS) repair on top of cpsat_literal.

Iteratively destroys a subset D of flow assignments from a seed schedule,
fixes the remaining flows in place, and asks CP-SAT to re-optimize the
destroy set with t_upper = current_makespan - 1. On FEASIBLE we accept
the strictly-better incumbent; on TIMEOUT/INFEASIBLE we keep the old one
and try a different destroy on the next iteration.

Destroy strategies:
  - "time_window":   pick all flows in the trailing `k` rounds of the
                     current incumbent.
  - "random_subset": pick K random flow keys (uniform).
  - "makespan_flows": find the bottleneck physical edge in the trailing
                     `k` rounds (most heavily used) and pick all flows
                     whose path contains that edge.

`destroy_size_frac` controls how big the destroy set is (fraction of N(N-1)).
"""
from __future__ import annotations
import random
from collections import Counter, defaultdict

from twisted_analysis.schedules.cpsat_literal import cpsat_literal
from twisted_analysis.schedules.verify import schedule_makespan


def _physical_edge_lb(table, n):
    c: Counter = Counter()
    for s in range(n):
        for d in range(n):
            if s == d:
                continue
            path = table[s][d]
            for h in range(len(path) - 1):
                c[(path[h], path[h + 1])] += 1
    return max(c.values()) if c else 0


def _makespan_defining_flows(schedule):
    """Must be unioned into every destroy set: pinning these flows with
    round + L >= M would immediately violate t_upper = M - 1."""
    M = max(int(e["round"]) + (len(e["path"]) - 1) for e in schedule)
    return {(int(e["src"]), int(e["dst"])) for e in schedule
            if int(e["round"]) + (len(e["path"]) - 1) >= M}


def _destroy_time_window(schedule, k_rounds):
    M = max(int(e["round"]) for e in schedule)
    cutoff = max(0, M - k_rounds + 1)
    return {(int(e["src"]), int(e["dst"])) for e in schedule
            if int(e["round"]) >= cutoff}


def _destroy_random_subset(schedule, K, rng):
    keys = [(int(e["src"]), int(e["dst"])) for e in schedule]
    rng.shuffle(keys)
    return set(keys[:K])


def _destroy_makespan_flows(schedule, k_rounds):
    """Find the bottleneck edge in the trailing window and destroy every
    flow that traverses it."""
    M = max(int(e["round"]) for e in schedule)
    cutoff = max(0, M - k_rounds + 1)
    edge_load: Counter = Counter()
    flows_on_edge: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for e in schedule:
        if int(e["round"]) < cutoff:
            continue
        path = e["path"]
        for h in range(len(path) - 1):
            edge_load[(path[h], path[h + 1])] += 1
            flows_on_edge[(path[h], path[h + 1])].append(
                (int(e["src"]), int(e["dst"]))
            )
    if not edge_load:
        return {(int(e["src"]), int(e["dst"])) for e in schedule}
    bottleneck = edge_load.most_common(1)[0][0]
    return set(flows_on_edge[bottleneck])


def lns_cpsat_repair(
    topology,
    table: list[list[list[int]]],
    seed_schedule: list[dict],
    *,
    n_iters: int = 100,
    per_subproblem_budget_s: int = 300,
    destroy_strategies: tuple[str, ...] = (
        "time_window", "random_subset", "makespan_flows",
    ),
    destroy_size_frac: float = 0.05,
    rng_seed: int = 0,
    n_workers: int = 8,
    log_fn=None,
) -> list[dict]:
    n = topology.n_nodes
    lb = _physical_edge_lb(table, n)
    rng = random.Random(rng_seed)

    incumbent: dict[tuple[int, int], int] = {
        (int(e["src"]), int(e["dst"])): int(e["round"]) for e in seed_schedule
    }
    incumbent_makespan = max(
        r + (len(table[s][d]) - 1) for (s, d), r in incumbent.items()
    )

    def _schedule_from_incumbent() -> list[dict]:
        out = []
        for (s, d), r in incumbent.items():
            out.append({
                "round": r, "src": s, "dst": d,
                "path": list(table[s][d]),
            })
        out.sort(key=lambda e: (e["round"], e["src"]))
        return out

    K = max(1, int(destroy_size_frac * (n * (n - 1))))
    k_rounds = max(1, int(0.10 * max(1, incumbent_makespan)))

    for it in range(n_iters):
        if incumbent_makespan <= lb:
            break
        strat = destroy_strategies[it % len(destroy_strategies)]
        cur_sched = _schedule_from_incumbent()
        if strat == "time_window":
            D = _destroy_time_window(cur_sched, k_rounds)
        elif strat == "random_subset":
            D = _destroy_random_subset(cur_sched, K, rng)
        elif strat == "makespan_flows":
            D = _destroy_makespan_flows(cur_sched, k_rounds)
        else:
            raise ValueError(f"unknown destroy strategy: {strat!r}")
        D = D | _makespan_defining_flows(cur_sched)
        if not D:
            continue
        fixed = {k: r for k, r in incumbent.items() if k not in D}
        if log_fn is not None:
            log_fn(it, {
                "strategy": strat, "destroy_size": len(D),
                "fixed_size": len(fixed),
                "current_makespan": incumbent_makespan,
                "target_t_upper": incumbent_makespan - 1,
            })
        try:
            new = cpsat_literal(
                topology, table,
                t_upper=incumbent_makespan - 1,
                time_limit_s=per_subproblem_budget_s,
                n_workers=n_workers,
                fixed_assignments=fixed,
                warm_start_schedule=cur_sched,
            )
        except RuntimeError:
            continue
        new_makespan = schedule_makespan(new)
        if new_makespan < incumbent_makespan:
            incumbent = {(int(e["src"]), int(e["dst"])): int(e["round"]) for e in new}
            incumbent_makespan = new_makespan

    return _schedule_from_incumbent()
