"""Integral (rate=1) earliest-feasible greedy for ragged workloads.

Flows are scheduled one at a time in a deterministic order. Each flow's
d_f = size/quantum quanta are placed at rate 1 under the pipelined-stream
model: a quantum placed at time t occupies path edge i at time t + i, and
a contiguous run of m quanta starting at t is one chunk entry
{round: t, rate: 1.0, size: m * quantum} occupying edge i over [t+i, t+m+i).

Variants:
  - non-preemptive (default): smallest start where every edge i is free
    throughout [start+i, start+i+d_f) -> exactly one entry per flow.
  - preemptive: quanta are placed at the earliest feasible times
    individually; each maximal contiguous run becomes one entry. Lower
    makespan, more entries — the makespan-vs-descriptor-count tradeoff.

Orders:
  - "lpt" (default): sort by (-size, -hops, src, dst).
  - "spt": (size, hops, src, dst).
  - "natural": workload iteration (= file) order.
"""
from __future__ import annotations
from collections import defaultdict

from twisted_analysis.model.ragged import RaggedWorkload

_VALID_ORDERS = {"lpt", "spt", "natural"}


def _runs(sorted_times: list[int]) -> list[tuple[int, int]]:
    """[3,4,5,9,10] -> [(3, 3), (9, 2)]: maximal (start, length) runs."""
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(sorted_times):
        j = i
        while j + 1 < len(sorted_times) and sorted_times[j + 1] == sorted_times[j] + 1:
            j += 1
        runs.append((sorted_times[i], j - i + 1))
        i = j + 1
    return runs


def ragged_greedy(
    table: list[list[list[int]]],
    workload: RaggedWorkload,
    *,
    order: str = "lpt",
    preemptive: bool = False,
) -> list[dict]:
    if order not in _VALID_ORDERS:
        raise ValueError(
            f"order must be one of {sorted(_VALID_ORDERS)}; got {order!r}"
        )
    q = workload.quantum
    flows = [
        (s, d, size, table[s][d])
        for (s, d), size in workload.demand.items()
    ]
    if order == "lpt":
        flows.sort(key=lambda f: (-f[2], -(len(f[3]) - 1), f[0], f[1]))
    elif order == "spt":
        flows.sort(key=lambda f: (f[2], len(f[3]) - 1, f[0], f[1]))
    # "natural": keep workload iteration order.

    edge_busy: dict[tuple[int, int], set[int]] = defaultdict(set)
    entries: list[dict] = []
    for s, d, size, path in flows:
        d_f = size // q
        hops = list(zip(path, path[1:]))
        if preemptive:
            starts: list[int] = []
            t = 0
            while len(starts) < d_f:
                if all((t + i) not in edge_busy[e] for i, e in enumerate(hops)):
                    starts.append(t)
                    for i, e in enumerate(hops):
                        edge_busy[e].add(t + i)
                t += 1
            for run_start, run_len in _runs(starts):
                entries.append({
                    "round": run_start, "src": s, "dst": d,
                    "path": list(path), "rate": 1.0, "size": run_len * q,
                })
        else:
            start = 0
            while any(
                (start + k + i) in edge_busy[e]
                for i, e in enumerate(hops)
                for k in range(d_f)
            ):
                start += 1
            for i, e in enumerate(hops):
                busy = edge_busy[e]
                for k in range(d_f):
                    busy.add(start + k + i)
            entries.append({
                "round": start, "src": s, "dst": d,
                "path": list(path), "rate": 1.0, "size": size,
            })
    entries.sort(key=lambda e: (e["round"], e["src"], e["dst"]))
    return entries
