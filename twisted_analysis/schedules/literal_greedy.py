"""LMR-style deterministic per-flow earliest-feasible greedy.

No orbit reduction. For each (src, dst) in `order`, pick the smallest start
time `t` such that for every hop i in the flow's path, the physical edge
(path[i], path[i+1]) is free at time t + i. Mark all those slots busy.

This is the simplest deterministic schedule that respects physical-edge
capacity. Worst-case makespan is bounded by LMR's O(congestion + dilation),
though we don't prove that bound here - we just rely on construction-time
feasibility.

Ordering options:
  - "lpt": longest-path first, tiebreak by (src, dst).
  - "spt": shortest-path first.
  - "natural": iterate sources outer, destinations inner (round-robin-ish).
"""
from __future__ import annotations
from collections import defaultdict

from twisted_analysis.topology import Topology

_VALID_ORDERS = {"lpt", "spt", "natural"}


def literal_greedy(
    topology: Topology,
    table: list[list[list[int]]],
    *,
    order: str = "lpt",
) -> list[dict]:
    """Schedule the AllToAll workload literally, one flow at a time.

    Returns: list of `{round, src, dst, path}` entries (sorted by round, src).
    """
    if order not in _VALID_ORDERS:
        raise ValueError(f"order must be one of {sorted(_VALID_ORDERS)}; got {order!r}")
    n = topology.n_nodes

    flows: list[tuple[int, int, list[int]]] = []
    for s in range(n):
        for d in range(n):
            if s == d:
                continue
            flows.append((s, d, list(table[s][d])))

    if order == "lpt":
        flows.sort(key=lambda f: (-len(f[2]), f[0], f[1]))
    elif order == "spt":
        flows.sort(key=lambda f: (len(f[2]), f[0], f[1]))
    # "natural": leave as constructed.

    edge_busy: dict[tuple[int, int], set[int]] = defaultdict(set)
    rounds: dict[tuple[int, int], int] = {}
    for src, dst, path in flows:
        L = len(path) - 1
        start = 0
        while True:
            conflict = False
            for i in range(L):
                u, v = path[i], path[i + 1]
                if (start + i) in edge_busy[(u, v)]:
                    conflict = True
                    break
            if not conflict:
                break
            start += 1
        for i in range(L):
            edge_busy[(path[i], path[i + 1])].add(start + i)
        rounds[(src, dst)] = start

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
