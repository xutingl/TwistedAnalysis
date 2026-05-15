"""Physical-edge capacity verification for schedules.

A schedule is a list of `{round, src, dst, path}` dicts (the on-disk format
in twisted_analysis/io/schedule.py). The verifier assumes PIPELINED firing:
hop `i` of a flow with `round = r` fires at absolute time `t = r + i`.

A capacity violation is a (physical_edge, time) pair where two distinct
flows traverse the same directed edge at the same time. This is exactly
what the canonical step-synchronous store-and-forward cost model
(docs/algorithm.md) forbids.

Use this to:
  1. Sanity-check any schedule before emitting a kernel.
  2. Compare schedulers across routings.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class CapacityViolation:
    edge: tuple[int, int]
    time: int
    flows: tuple[tuple[int, int, int], ...]  # (round, src, dst) per colliding flow


def verify_capacity(schedule: Iterable[Mapping[str, object]]) -> list[CapacityViolation]:
    """Return all (edge, time) pairs with >1 flow using them simultaneously.

    Assumes pipelined hop firing: hop i of a flow with round r fires at t = r + i.
    """
    bucket: dict[tuple[tuple[int, int], int], list[tuple[int, int, int]]] = defaultdict(list)
    for entry in schedule:
        r = int(entry["round"])
        src = int(entry["src"])
        dst = int(entry["dst"])
        path = entry["path"]
        for i in range(len(path) - 1):
            u, v = int(path[i]), int(path[i + 1])
            t = r + i
            bucket[((u, v), t)].append((r, src, dst))

    violations: list[CapacityViolation] = []
    for (edge, t), flows in bucket.items():
        if len(flows) > 1:
            violations.append(CapacityViolation(edge=edge, time=t, flows=tuple(flows)))
    violations.sort(key=lambda v: (v.time, v.edge))
    return violations


def schedule_makespan(schedule: Iterable[Mapping[str, object]]) -> int:
    """Latest finish time + 1. A flow with round r and path length L finishes
    after its last hop fires at t = r + L - 1, contributing makespan r + L."""
    m = 0
    for entry in schedule:
        L = len(entry["path"]) - 1
        finish = int(entry["round"]) + L
        if finish > m:
            m = finish
    return m
