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
from itertools import groupby
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


@dataclass(frozen=True)
class RateViolation:
    edge: tuple[int, int]
    time: float
    total_rate: float
    flows: tuple[tuple[int, int, int], ...]  # (round, src, dst) per active chunk


def _chunk_params(entry: Mapping[str, object], quantum: int) -> tuple[float, float, float]:
    """(start_round, rate, duration_in_quanta) under the pipelined-stream model.

    Defaults reproduce the legacy uniform semantics: rate=1, size=1,
    quantum=1 give duration 1.
    """
    r = float(int(entry["round"]))
    rate = float(entry.get("rate", 1.0))
    size = int(entry.get("size", 1))
    duration = (size / quantum) / rate
    return r, rate, duration


def verify_capacity_ragged(
    schedule: Iterable[Mapping[str, object]],
    *,
    quantum: int = 1,
    tol: float = 1e-6,
) -> list[RateViolation]:
    """Rate-capacity check under the pipelined-stream model.

    A chunk {round=r, rate, size, path} occupies directed edge i of its
    path during the half-open interval [r + i, r + i + (size/quantum)/rate),
    consuming `rate` of the edge's unit capacity. A violation is any
    (edge, time) where accumulated rate exceeds 1 + tol.

    With defaulted fields and quantum=1 this reduces to verify_capacity's
    one-flow-per-edge-per-step semantics.
    """
    events: dict[
        tuple[int, int],
        list[tuple[float, int, float, tuple[int, int, int]]],
    ] = defaultdict(list)
    for entry in schedule:
        r, rate, duration = _chunk_params(entry, quantum)
        key = (int(entry["round"]), int(entry["src"]), int(entry["dst"]))
        path = entry["path"]
        for i in range(len(path) - 1):
            u, v = int(path[i]), int(path[i + 1])
            events[(u, v)].append((r + i, 1, rate, key))
            events[(u, v)].append((r + i + duration, 0, rate, key))

    violations: list[RateViolation] = []
    for edge, evs in events.items():
        # Sort ends (kind 0) before starts (kind 1) at equal times: intervals
        # are half-open, so back-to-back chunks never overlap.
        evs.sort(key=lambda t: (t[0], t[1]))
        acc = 0.0
        active: set[tuple[int, int, int]] = set()
        for time, group in groupby(evs, key=lambda t: t[0]):
            saw_start = False
            for _time, kind, rate, key in group:
                if kind == 1:
                    acc += rate
                    active.add(key)
                    saw_start = True
                else:
                    acc -= rate
                    active.discard(key)
            if saw_start and acc > 1 + tol:
                violations.append(RateViolation(
                    edge=edge, time=time, total_rate=acc,
                    flows=tuple(sorted(active)),
                ))
    violations.sort(key=lambda v: (v.time, v.edge))
    return violations


def schedule_makespan_ragged(
    schedule: Iterable[Mapping[str, object]],
    *,
    quantum: int = 1,
) -> float:
    """Latest chunk finish: round + (L-1) + (size/quantum)/rate, L path hops.

    Reduces to schedule_makespan (round + L) on legacy entries.
    """
    m = 0.0
    for entry in schedule:
        r, _rate, duration = _chunk_params(entry, quantum)
        hops = len(entry["path"]) - 1
        m = max(m, r + (hops - 1) + duration)
    return m


def verify_workload_coverage(
    schedule: Iterable[Mapping[str, object]],
    workload,
) -> list[str]:
    """Check per-pair chunk sizes sum exactly to the workload demand.

    `workload` is a twisted_analysis.model.ragged.RaggedWorkload. Returns
    human-readable problem strings; empty list = pass.
    """
    sums: dict[tuple[int, int], int] = defaultdict(int)
    for entry in schedule:
        sums[(int(entry["src"]), int(entry["dst"]))] += int(entry.get("size", 1))
    problems: list[str] = []
    for pair, size in sorted(workload.demand.items()):
        got = sums.pop(pair, 0)
        if got != size:
            problems.append(f"pair {pair}: scheduled {got} != demand {size}")
    for pair, got in sorted(sums.items()):
        problems.append(f"pair {pair}: scheduled {got} but not in workload")
    return problems
