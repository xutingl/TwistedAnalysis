"""Per-flow greedy AllToAll scheduler with per-device DMA cap.

For each flow `(src, dst, path)` in the chosen order, picks the smallest
start time `t` such that:
  (a) every hop's physical edge `(path[i], path[i+1])` is free at time `t+i`;
  (b) `out_count[(src, t)] < k`  — src device has not yet issued K outgoing
      DMAs at this round;
  (c) `in_count[(dst, t)] < k`   — dst device has not yet received K
      incoming DMAs at this round.

Then marks the chosen slots busy and increments both counters.

Tradeoff vs `literal_greedy` (which has no device cap, i.e. effectively
K = infinity):
  - K = 1: each device emits/receives one DMA per round; makespan >= N-1
           (device LB binds). Equivalent in structure to the reference P2P
           rotation kernel, but with LB-aware per-flow ordering instead of
           fixed rotation. On loaded 8x4x4 (N=128), makespan >= 127.
  - K = 2, 3, 4: moderate pipelining; makespan between max(N-1)/K and the
           physical-edge LB.
  - K = N: equivalent to `literal_greedy` (cap non-binding).

Motivation: wall-clock on TPU v5e showed that per-device DMA-engine
concurrency and ICI link bandwidth dominate per-round latency, so
minimising simultaneous DMAs per device (at the cost of higher round
count) may reduce wall-clock time. `spread_greedy` lets us test that
hypothesis by varying K before committing to hardware experiments.
"""
from __future__ import annotations
from collections import defaultdict

from twisted_analysis.topology import Topology

_VALID_ORDERS = {"lpt", "spt", "natural"}


def spread_greedy(
    topology: Topology,
    table: list[list[list[int]]],
    *,
    k: int,
    order: str = "lpt",
) -> list[dict]:
    """Schedule the AllToAll workload with a per-device DMA cap.

    Args:
      topology: source of `n_nodes`.
      table: routing table; `table[src][dst]` is the list of flat-IDs
        traversed from src to dst.
      k: max outgoing AND max incoming DMAs per device per round. Must be
        a positive integer.
      order: flow ordering at greedy time. One of `lpt`, `spt`, `natural`.

    Returns:
      List of `{round, src, dst, path}` entries (sorted by round, src).
    """
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be a positive integer; got {k!r}")
    if order not in _VALID_ORDERS:
        raise ValueError(
            f"order must be one of {sorted(_VALID_ORDERS)}; got {order!r}"
        )
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
    out_count: dict[tuple[int, int], int] = defaultdict(int)
    in_count: dict[tuple[int, int], int] = defaultdict(int)
    rounds: dict[tuple[int, int], int] = {}

    for src, dst, path in flows:
        L = len(path) - 1
        start = 0
        while True:
            if out_count[(src, start)] >= k or in_count[(dst, start)] >= k:
                start += 1
                continue
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
        out_count[(src, start)] += 1
        in_count[(dst, start)] += 1
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
