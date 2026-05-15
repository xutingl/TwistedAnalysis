# Multi-Algorithm AllToAll Scheduling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three new scheduling algorithms (orbit-greedy with full-physical-edge accounting, literal-flow greedy, exact ILP on literal flows) alongside the existing orbit-greedy, and generalize the Pallas kernel generator to select among them. The plan also empirically decides whether to keep the existing orbit-greedy (which classifies edges by `(dim, dir)`) by running a capacity verifier against an ILP-routed fixture.

**Architecture:** Each scheduler is its own module under `twisted_analysis/schedules/` and produces the existing JSON schedule format (`[{round, src, dst, path}, ...]`). A new capacity verifier checks that physical-edge usage at each step (under pipelined `t_i = round + i` semantics) is at most 1. The generator gains a `--scheduler` flag and a post-schedule verification pass. New kernel files and schedule fixtures get regenerated for `(8,4,4)` against `routing_table_8x4x4_twist.json`.

**Tech Stack:** Python 3, numpy, pulp (CBC) for ILP, existing `twisted_analysis` topology / IO / orbit machinery. Tests use pytest.

**Key files / responsibilities:**
- `twisted_analysis/schedules/verify.py` (NEW) — physical-edge capacity verifier.
- `twisted_analysis/schedules/orbit_greedy_full.py` (NEW) — option 1: orbit-greedy keyed on the full set of physical edges an orbit's flows traverse at each hop.
- `twisted_analysis/schedules/literal_greedy.py` (NEW) — option 2: per-flow earliest-feasible greedy (LMR-style, deterministic).
- `twisted_analysis/schedules/ilp_literal.py` (NEW) — option 3: exact ILP on the literal `N(N-1)` flow set.
- `twisted_analysis/io/schedule.py` (MODIFY) — add `schedule_from_*` adapters and a `schedule_from_algorithm(name, ...)` dispatcher.
- `pallas_kernel/gen_orbit_greedy_kernel.py` (MODIFY) — add `--scheduler` flag, dispatch, mandatory post-schedule verification.
- Tests: `tests/test_schedule_verify.py`, `tests/test_orbit_greedy_full.py`, `tests/test_literal_greedy.py`, `tests/test_ilp_literal.py`, `tests/test_orbit_greedy_dimdir_correctness.py` (NEW, regression).

**Decision point (Task 3 result):**
- If existing orbit-greedy produces 0 violations on every ILP-routed cell → keep it as algorithm `orbit_greedy` (the `(dim, dir)` keying is provably-correct under translation-equivariant routings). `orbit_greedy_full` is added as a new selectable option for non-equivariant routings (loaded TPU table).
- If it produces violations on any ILP-routed cell → the `(dim, dir)` keying is generally unsound; replace `orbit_greedy` with `orbit_greedy_full` and drop the old code.

**Decision (Task 3 result, 2026-05-15):** FAIL — replace orbit_greedy with orbit_greedy_full

---

## Task 1: Physical-edge capacity verifier

**Files:**
- Create: `twisted_analysis/schedules/verify.py`
- Create: `tests/test_schedule_verify.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_schedule_verify.py
import pytest

from twisted_analysis.schedules.verify import (
    CapacityViolation,
    schedule_makespan,
    verify_capacity,
)


def test_verify_no_violations_on_disjoint_paths():
    schedule = [
        {"round": 0, "src": 0, "dst": 1, "path": [0, 1]},
        {"round": 0, "src": 2, "dst": 3, "path": [2, 3]},
    ]
    assert verify_capacity(schedule) == []


def test_verify_detects_same_edge_same_time_violation():
    schedule = [
        # Two flows both use edge (1->2) at time t=1.
        {"round": 0, "src": 0, "dst": 2, "path": [0, 1, 2]},
        {"round": 1, "src": 1, "dst": 2, "path": [1, 2]},
    ]
    violations = verify_capacity(schedule)
    assert len(violations) == 1
    v = violations[0]
    assert v.edge == (1, 2)
    assert v.time == 1
    assert len(v.flows) == 2


def test_verify_same_edge_different_time_ok():
    schedule = [
        # Same edge but at different absolute times (0 vs 2).
        {"round": 0, "src": 0, "dst": 1, "path": [0, 1]},
        {"round": 2, "src": 0, "dst": 1, "path": [0, 1]},
    ]
    assert verify_capacity(schedule) == []


def test_schedule_makespan():
    schedule = [
        {"round": 0, "src": 0, "dst": 2, "path": [0, 1, 2]},   # uses t=0, t=1
        {"round": 3, "src": 5, "dst": 6, "path": [5, 6]},      # uses t=3
    ]
    # Latest hop fires at t=3, makespan = 4.
    assert schedule_makespan(schedule) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_schedule_verify.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'twisted_analysis.schedules.verify'`

- [ ] **Step 3: Implement the verifier**

```python
# twisted_analysis/schedules/verify.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_schedule_verify.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add twisted_analysis/schedules/verify.py tests/test_schedule_verify.py
git commit -m "feat(schedules): add physical-edge capacity verifier"
```

---

## Task 2: Confirm loaded-routing schedule has violations (regression test)

This locks in the empirical fact that motivated this work: the existing fixture `schedule_8x4x4_loaded_lpt_tail_asc.json` is NOT physical-edge-feasible under the loaded routing. The test will fail later if a future fix accidentally papers over this finding without addressing the root cause.

**Files:**
- Create: `tests/test_loaded_routing_violations.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_loaded_routing_violations.py
"""Regression: the orbit_greedy schedule baked against the LOADED
8x4x4 routing table has physical-edge capacity violations.

This documents that orbit_greedy's (dim, dir) keying is unsound for
routings that are not strictly translation-equivariant under that key
(see docs/superpowers/plans/2026-05-15-multi-algorithm-scheduling.md
Task 3 for the explanation).
"""
from pathlib import Path

from twisted_analysis.io.schedule import load_schedule
from twisted_analysis.schedules.verify import verify_capacity

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def test_loaded_8x4x4_orbit_greedy_has_violations():
    schedule = load_schedule(FIXTURES / "schedule_8x4x4_loaded_lpt_tail_asc.json")
    violations = verify_capacity(schedule)
    # The exact count was 640 at the time this regression was added; we assert
    # "many" rather than the exact number to avoid brittleness if greedy
    # tie-breaking changes harmlessly.
    assert len(violations) >= 100, (
        f"Expected loaded-routing orbit_greedy to have >=100 physical-edge "
        f"violations under (dim, dir) keying; got {len(violations)}. "
        f"If this test now passes with 0, document why and update the plan."
    )
```

- [ ] **Step 2: Run test to verify it fails (initially) or passes (if fixture present)**

Run: `.venv/bin/python -m pytest tests/test_loaded_routing_violations.py -v`
Expected: PASS (the fixture already exists and has violations).

- [ ] **Step 3: Commit**

```bash
git add tests/test_loaded_routing_violations.py
git commit -m "test: lock in loaded-routing capacity violations regression"
```

---

## Task 3: Empirical correctness check of existing orbit_greedy on ILP routings

This is the decision point. Run the verifier on a fresh schedule produced by the existing `orbit_greedy` on every ILP-routed cell. If 0 violations on every cell, the `(dim, dir)` keying is empirically correct for translation-equivariant routings — keep `orbit_greedy` and add `orbit_greedy_full` as a new option. If any cell has violations, the existing algorithm is unsound in general and must be replaced.

**Files:**
- Create: `tests/test_orbit_greedy_dimdir_correctness.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_orbit_greedy_dimdir_correctness.py
"""Empirical correctness of the existing (dim, dir)-keyed orbit_greedy
on every (slice, ILP-router) cell that the optimality proof covers.

Pass condition: 0 physical-edge capacity violations on every cell. This is
the decision criterion for whether to keep orbit_greedy as-is or replace
it with orbit_greedy_full (see plan Task 3).
"""
import pytest

from twisted_analysis.io.routing_table import save_routing_table
from twisted_analysis.io.schedule import schedule_from_orbit_greedy
from twisted_analysis.io.routing_table import load_routing_table
from twisted_analysis.schedules.verify import verify_capacity
from twisted_analysis.topology import Topology, ILPRouter


CELLS = [
    (2, 4),
    (2, 2, 4),
    (2, 4, 4),
    (4, 8),
    # (4, 4, 8) excluded: ILP router takes ~21 s; covered in a slow-test
    # variant if needed.
]


@pytest.mark.parametrize("slice_", CELLS)
def test_orbit_greedy_dimdir_no_violations_under_ilp(tmp_path, slice_):
    topology = Topology(slice=slice_)
    router = ILPRouter(topology=topology)
    rt_path = tmp_path / "rt.json"
    save_routing_table(topology, router, rt_path)
    table = load_routing_table(rt_path)
    schedule = schedule_from_orbit_greedy(topology, table, order="lpt_tail_asc")
    violations = verify_capacity(schedule)
    assert violations == [], (
        f"orbit_greedy with (dim, dir) keying on ILP-routed {slice_} produced "
        f"{len(violations)} physical-edge violations. First 3: {violations[:3]}"
    )
```

- [ ] **Step 2: Run the test**

Run: `.venv/bin/python -m pytest tests/test_orbit_greedy_dimdir_correctness.py -v`
Expected outcomes:
- All PASS → existing `orbit_greedy` is correct on translation-equivariant routings. Keep it; add `orbit_greedy_full` as a separate algorithm in later tasks.
- Any FAIL → existing `orbit_greedy` is unsound in general. Plan Task 7 must alias `orbit_greedy` → `orbit_greedy_full` and remove the old `_emit_orbit_greedy` body.

- [ ] **Step 3: Record decision in the plan**

After Step 2, add a one-line note at the top of this file (just below "Decision point"):
```
**Decision (Task 3 result, YYYY-MM-DD):** [PASS — keep orbit_greedy] or [FAIL — replace orbit_greedy with orbit_greedy_full]
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_orbit_greedy_dimdir_correctness.py docs/superpowers/plans/2026-05-15-multi-algorithm-scheduling.md
git commit -m "test: validate orbit_greedy (dim,dir) correctness on ILP-routed cells"
```

---

## Task 4: `orbit_greedy_full` algorithm (option 1)

Per-orbit greedy where each hop's busy table is keyed on the SET of physical edges the orbit's `N` flows actually traverse at that hop — not on `(dim, dir)`. This is correct for any translation-symmetric workload regardless of whether the routing is `(dim, dir)`-equivariant.

**Files:**
- Create: `twisted_analysis/schedules/orbit_greedy_full.py`
- Create: `tests/test_orbit_greedy_full.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_orbit_greedy_full.py
from pathlib import Path

import pytest

from twisted_analysis.io.routing_table import load_routing_table, save_routing_table
from twisted_analysis.schedules.orbit_greedy_full import (
    compute_hop0_firing_times_full,
)
from twisted_analysis.schedules.verify import verify_capacity, schedule_makespan
from twisted_analysis.topology import Topology, ILPRouter

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _schedule_from_full(topology, table, order="lpt_tail_asc"):
    # Inline adapter; the io/schedule.py dispatcher in Task 7 will subsume this.
    from twisted_analysis.io.coords import flatten
    from twisted_analysis.lp.orbit import compute_orbits
    t0 = compute_hop0_firing_times_full(topology, table, order=order)
    orbits = compute_orbits(topology)
    entries = []
    for orbit_id, members in orbits.items():
        r = int(t0[orbit_id])
        for src, dst in members:
            src_flat = flatten(src, topology.slice)
            dst_flat = flatten(dst, topology.slice)
            entries.append({"round": r, "src": src_flat, "dst": dst_flat,
                            "path": list(table[src_flat][dst_flat])})
    entries.sort(key=lambda e: (e["round"], e["src"]))
    return entries


def test_orbit_greedy_full_zero_violations_on_loaded_8x4x4():
    topology = Topology(slice=(8, 4, 4))
    table = load_routing_table(FIXTURES / "routing_table_8x4x4_twist.json")
    schedule = _schedule_from_full(topology, table)
    violations = verify_capacity(schedule)
    assert violations == [], (
        f"orbit_greedy_full produced {len(violations)} physical-edge "
        f"violations on loaded 8x4x4 routing. First: {violations[0]}"
    )


def test_orbit_greedy_full_zero_violations_on_ilp_4x8(tmp_path):
    topology = Topology(slice=(4, 8))
    router = ILPRouter(topology=topology)
    rt_path = tmp_path / "rt.json"
    save_routing_table(topology, router, rt_path)
    table = load_routing_table(rt_path)
    schedule = _schedule_from_full(topology, table)
    assert verify_capacity(schedule) == []
    # On ILP-routed cells the LB-tight makespan should match docs/results.md.
    # For 4x8 ILP, LB = 21; makespan_full <= LB + diameter = 21 + 4 = 25.
    assert schedule_makespan(schedule) <= 25
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/bin/python -m pytest tests/test_orbit_greedy_full.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the algorithm**

```python
# twisted_analysis/schedules/orbit_greedy_full.py
"""Orbit-greedy with FULL physical-edge accounting.

Differs from twisted_analysis.schedules.orbit_greedy in one place: the
busy table is keyed on the literal set of physical edges an orbit's `N`
flows traverse at each hop, rather than on the `(dim, dir)` class label.

When the routing is translation-equivariant under the (dim, dir) action
(DOR, ILP), the two formulations are equivalent: each (dim, dir) class is
saturated by exactly N distinct physical edges, so checking the class is
the same as checking the edge set. When the routing is NOT equivariant
under (dim, dir) — e.g., a TPU OCS-derived "loaded" routing where
twist-wrap edges and standard edges are intermixed — the two diverge,
and only the full-edge formulation respects physical-edge capacity.
"""
from __future__ import annotations
from collections import Counter, defaultdict

from twisted_analysis.io.coords import flatten
from twisted_analysis.lp.orbit import compute_orbits
from twisted_analysis.topology import Topology

_VALID_ORDERS = {"lpt", "spt", "lpt_tail_asc", "tail_asc"}


def _orbit_hop_edge_sets(
    topology: Topology,
    table: list[list[list[int]]],
) -> dict:
    """For each orbit, return [set_of_edges_at_hop_0, set_at_hop_1, ...]."""
    orbits = compute_orbits(topology)
    slice_ = topology.slice
    per_orbit: dict = {}
    for orbit_id, members in orbits.items():
        hop_sets: list[set[tuple[int, int]]] = []
        for src, dst in members:
            src_flat = flatten(src, slice_)
            dst_flat = flatten(dst, slice_)
            path = table[src_flat][dst_flat]
            for i in range(len(path) - 1):
                if i >= len(hop_sets):
                    hop_sets.append(set())
                hop_sets[i].add((path[i], path[i + 1]))
        per_orbit[orbit_id] = hop_sets
    return per_orbit


def _edge_orbit_load_full(per_orbit: dict) -> Counter:
    """Per-physical-edge total demand across the workload."""
    c: Counter = Counter()
    for hop_sets in per_orbit.values():
        for edges in hop_sets:
            for e in edges:
                c[e] += 1
    return c


def _ordered_orbits_full(per_orbit: dict, edge_load: Counter, order: str) -> list:
    if order == "lpt":
        return sorted(per_orbit, key=lambda o: (-len(per_orbit[o]), o))
    if order == "spt":
        return sorted(per_orbit, key=lambda o: (len(per_orbit[o]), o))
    if order == "lpt_tail_asc":
        def tail_load(o):
            hops = per_orbit[o]
            if not hops:
                return 0
            return max(edge_load[e] for e in hops[-1])
        return sorted(per_orbit, key=lambda o: (-len(per_orbit[o]), tail_load(o), o))
    if order == "tail_asc":
        def tail_load(o):
            hops = per_orbit[o]
            if not hops:
                return 0
            return max(edge_load[e] for e in hops[-1])
        return sorted(per_orbit, key=lambda o: (tail_load(o), -len(per_orbit[o]), o))
    raise ValueError(f"unknown order: {order}; valid={sorted(_VALID_ORDERS)}")


def compute_hop0_firing_times_full(
    topology: Topology,
    table: list[list[list[int]]],
    *,
    order: str = "lpt_tail_asc",
) -> dict:
    """Return per-orbit hop-0 firing time under orbit_greedy_full.

    Greedy: process orbits in `order`; for each orbit, find the earliest
    `t_0` such that for every hop i, ALL physical edges in the orbit's
    hop-i set are free at time t_0 + i. Mark them busy after firing.
    """
    if order not in _VALID_ORDERS:
        raise ValueError(f"order must be one of {sorted(_VALID_ORDERS)}; got {order!r}")
    per_orbit = _orbit_hop_edge_sets(topology, table)
    edge_load = _edge_orbit_load_full(per_orbit)
    edge_busy: dict[tuple[int, int], set[int]] = defaultdict(set)
    t_hop0: dict = {}
    for orbit_id in _ordered_orbits_full(per_orbit, edge_load, order):
        hops = per_orbit[orbit_id]
        # Find smallest start such that for every i, hops[i] is fully free at start+i.
        start = 0
        while True:
            ok = True
            for i, edges in enumerate(hops):
                t = start + i
                for e in edges:
                    if t in edge_busy[e]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                break
            start += 1
        for i, edges in enumerate(hops):
            for e in edges:
                edge_busy[e].add(start + i)
        t_hop0[orbit_id] = start
    return t_hop0
```

- [ ] **Step 4: Run test, verify it passes**

Run: `.venv/bin/python -m pytest tests/test_orbit_greedy_full.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add twisted_analysis/schedules/orbit_greedy_full.py tests/test_orbit_greedy_full.py
git commit -m "feat(schedules): add orbit_greedy_full (physical-edge keyed)"
```

---

## Task 5: `literal_greedy` algorithm (option 2)

Per-flow earliest-feasible greedy with no orbit reduction. Operates on the literal `N(N-1)` flow set; for each flow in a chosen order, finds the smallest start time such that all hops are free on physical edges. Always feasible by construction.

**Files:**
- Create: `twisted_analysis/schedules/literal_greedy.py`
- Create: `tests/test_literal_greedy.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_literal_greedy.py
from pathlib import Path

import pytest

from twisted_analysis.io.routing_table import load_routing_table, save_routing_table
from twisted_analysis.schedules.literal_greedy import literal_greedy
from twisted_analysis.schedules.verify import verify_capacity, schedule_makespan
from twisted_analysis.topology import Topology, ILPRouter

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def test_literal_greedy_zero_violations_on_loaded_8x4x4():
    topology = Topology(slice=(8, 4, 4))
    table = load_routing_table(FIXTURES / "routing_table_8x4x4_twist.json")
    schedule = literal_greedy(topology, table, order="lpt")
    assert verify_capacity(schedule) == []
    # Sanity: every (src, dst) pair with src != dst appears exactly once.
    pairs = {(e["src"], e["dst"]) for e in schedule}
    n = topology.n_nodes
    assert len(pairs) == n * (n - 1)


@pytest.mark.parametrize("order", ["lpt", "spt", "natural"])
def test_literal_greedy_orderings_all_feasible(tmp_path, order):
    topology = Topology(slice=(2, 4))
    router = ILPRouter(topology=topology)
    rt_path = tmp_path / "rt.json"
    save_routing_table(topology, router, rt_path)
    table = load_routing_table(rt_path)
    schedule = literal_greedy(topology, table, order=order)
    assert verify_capacity(schedule) == []
    # Loose upper bound: c + d. For 2x4 ILP, LB=3, d=2, so makespan <= some small value.
    # Use 6 * LB as a very generous bound.
    assert schedule_makespan(schedule) <= 6 * 3
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/bin/python -m pytest tests/test_literal_greedy.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the algorithm**

```python
# twisted_analysis/schedules/literal_greedy.py
"""LMR-style deterministic per-flow earliest-feasible greedy.

No orbit reduction. For each (src, dst) in `order`, pick the smallest start
time `t` such that for every hop i in the flow's path, the physical edge
(path[i], path[i+1]) is free at time t + i. Mark all those slots busy.

This is the simplest deterministic schedule that respects physical-edge
capacity. Worst-case makespan is bounded by LMR's O(congestion + dilation),
though we don't prove that bound here — we just rely on construction-time
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
```

- [ ] **Step 4: Run test, verify it passes**

Run: `.venv/bin/python -m pytest tests/test_literal_greedy.py -v`
Expected: PASS (4 tests, including 3 parametrized).

- [ ] **Step 5: Commit**

```bash
git add twisted_analysis/schedules/literal_greedy.py tests/test_literal_greedy.py
git commit -m "feat(schedules): add literal_greedy (LMR-style per-flow earliest-feasible)"
```

---

## Task 6: `ilp_literal` algorithm (option 3)

Exact ILP for the literal-flow scheduling problem: minimize makespan subject to chain (pipelined hops) and edge capacity (≤1 flow per edge per step). Tractable only on small cells; the test pins it to 2×4 where CBC finishes in <5 s.

**Files:**
- Create: `twisted_analysis/schedules/ilp_literal.py`
- Create: `tests/test_ilp_literal.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ilp_literal.py
import pytest

from twisted_analysis.io.routing_table import load_routing_table, save_routing_table
from twisted_analysis.schedules.ilp_literal import ilp_literal
from twisted_analysis.schedules.verify import verify_capacity, schedule_makespan
from twisted_analysis.topology import Topology, ILPRouter


@pytest.mark.timeout(60)
def test_ilp_literal_matches_lb_on_2x4(tmp_path):
    topology = Topology(slice=(2, 4))
    router = ILPRouter(topology=topology)
    rt_path = tmp_path / "rt.json"
    save_routing_table(topology, router, rt_path)
    table = load_routing_table(rt_path)
    schedule = ilp_literal(topology, table, time_limit_s=30)
    assert verify_capacity(schedule) == []
    # For 2x4 ILP: LB = 3 per docs/results.md, diameter d = 2.
    # Literal-ILP makespan should match LB (the symmetric ILP cross-validated
    # T = LB on this cell).
    assert schedule_makespan(schedule) == 3


def test_ilp_literal_raises_without_pulp(monkeypatch):
    # Defensive: surface a clean error if pulp isn't installed.
    import sys
    monkeypatch.setitem(sys.modules, "pulp", None)
    topology = Topology(slice=(2, 4))
    # Build minimal table from DOR to avoid pulp dependency for setup.
    from twisted_analysis.topology import DORRouter
    from twisted_analysis.io.routing_table import save_routing_table, load_routing_table
    from pathlib import Path
    rt = Path("/tmp/_test_ilp_lit_no_pulp.json")
    save_routing_table(topology, DORRouter(topology=topology), rt)
    table = load_routing_table(rt)
    with pytest.raises((ImportError, RuntimeError, TypeError)):
        ilp_literal(topology, table, time_limit_s=5)
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ilp_literal.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the algorithm**

```python
# twisted_analysis/schedules/ilp_literal.py
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
```

- [ ] **Step 4: Install pulp if not present**

Ask user to run: `uv pip install pulp` (the user prefers to run install commands themselves per CLAUDE.md preferences).
Expected: pulp installed in the project venv.

- [ ] **Step 5: Run test, verify it passes**

Run: `.venv/bin/python -m pytest tests/test_ilp_literal.py -v`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add twisted_analysis/schedules/ilp_literal.py tests/test_ilp_literal.py
git commit -m "feat(schedules): add ilp_literal (exact ILP on literal flows)"
```

---

## Task 7: Unified `schedule_from_algorithm` dispatcher

**Files:**
- Modify: `twisted_analysis/io/schedule.py`
- Create: `tests/test_schedule_from_algorithm.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_schedule_from_algorithm.py
import pytest

from twisted_analysis.io.routing_table import load_routing_table, save_routing_table
from twisted_analysis.io.schedule import schedule_from_algorithm
from twisted_analysis.schedules.verify import verify_capacity
from twisted_analysis.topology import Topology, ILPRouter


@pytest.mark.parametrize("algo", ["orbit_greedy", "orbit_greedy_full", "literal_greedy"])
def test_dispatcher_runs_each_algorithm_on_2x4(tmp_path, algo):
    topology = Topology(slice=(2, 4))
    router = ILPRouter(topology=topology)
    rt_path = tmp_path / "rt.json"
    save_routing_table(topology, router, rt_path)
    table = load_routing_table(rt_path)
    schedule = schedule_from_algorithm(algo, topology, table)
    assert verify_capacity(schedule) == []
    # Exactly N*(N-1) entries.
    n = topology.n_nodes
    assert len(schedule) == n * (n - 1)


def test_dispatcher_rejects_unknown_algorithm(tmp_path):
    topology = Topology(slice=(2, 4))
    router = ILPRouter(topology=topology)
    rt_path = tmp_path / "rt.json"
    save_routing_table(topology, router, rt_path)
    table = load_routing_table(rt_path)
    with pytest.raises(ValueError, match="unknown algorithm"):
        schedule_from_algorithm("does_not_exist", topology, table)
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/bin/python -m pytest tests/test_schedule_from_algorithm.py -v`
Expected: FAIL with `ImportError: cannot import name 'schedule_from_algorithm'`.

- [ ] **Step 3: Add the adapter functions and dispatcher to `twisted_analysis/io/schedule.py`**

Append the following to the end of `twisted_analysis/io/schedule.py`:

```python
def schedule_from_orbit_greedy_full(
    topology: Topology,
    table: list[list[list[int]]],
    *,
    order: str = "lpt_tail_asc",
) -> list[dict]:
    """Adapter: orbit_greedy_full -> schedule entries."""
    from twisted_analysis.io.coords import flatten
    from twisted_analysis.io.routing_table import validate_routing_table_shape
    from twisted_analysis.lp.orbit import compute_orbits
    from twisted_analysis.schedules.orbit_greedy_full import (
        compute_hop0_firing_times_full,
    )

    validate_routing_table_shape(table, topology.n_nodes)
    t0 = compute_hop0_firing_times_full(topology, table, order=order)
    orbits = compute_orbits(topology)
    slice_ = topology.slice

    entries: list[dict] = []
    for orbit_id, members in orbits.items():
        round_t = int(t0[orbit_id])
        for (src, dst) in members:
            src_flat = flatten(src, slice_)
            dst_flat = flatten(dst, slice_)
            entries.append({
                "round": round_t,
                "src": src_flat,
                "dst": dst_flat,
                "path": list(table[src_flat][dst_flat]),
            })
    entries.sort(key=lambda e: (e["round"], e["src"]))
    return entries


def schedule_from_literal_greedy(
    topology: Topology,
    table: list[list[list[int]]],
    *,
    order: str = "lpt",
) -> list[dict]:
    """Adapter: literal_greedy -> schedule entries (already in correct format)."""
    from twisted_analysis.io.routing_table import validate_routing_table_shape
    from twisted_analysis.schedules.literal_greedy import literal_greedy

    validate_routing_table_shape(table, topology.n_nodes)
    return literal_greedy(topology, table, order=order)


def schedule_from_ilp_literal(
    topology: Topology,
    table: list[list[list[int]]],
    *,
    t_upper: int | None = None,
    time_limit_s: int = 600,
) -> list[dict]:
    """Adapter: ilp_literal -> schedule entries."""
    from twisted_analysis.io.routing_table import validate_routing_table_shape
    from twisted_analysis.schedules.ilp_literal import ilp_literal

    validate_routing_table_shape(table, topology.n_nodes)
    return ilp_literal(
        topology, table, t_upper=t_upper, time_limit_s=time_limit_s,
    )


_SCHEDULER_DISPATCH = {
    "orbit_greedy": schedule_from_orbit_greedy,
    "orbit_greedy_full": schedule_from_orbit_greedy_full,
    "literal_greedy": schedule_from_literal_greedy,
    "ilp_literal": schedule_from_ilp_literal,
}


def schedule_from_algorithm(
    algorithm: str,
    topology: Topology,
    table: list[list[list[int]]],
    **kwargs,
) -> list[dict]:
    """Dispatch to the named scheduler.

    Available algorithms:
      - "orbit_greedy":      original, (dim, dir)-keyed orbit greedy.
        Provably correct on translation-equivariant routings only.
      - "orbit_greedy_full": orbit greedy with full physical-edge accounting.
        Correct under any translation-symmetric workload (including loaded TPU routings).
      - "literal_greedy":    LMR-style per-flow earliest-feasible greedy.
      - "ilp_literal":       exact ILP on literal flows. Small cells only.

    Per-algorithm kwargs (e.g., `order`, `time_limit_s`) are passed through.
    """
    if algorithm not in _SCHEDULER_DISPATCH:
        raise ValueError(
            f"unknown algorithm: {algorithm!r}; "
            f"choices: {sorted(_SCHEDULER_DISPATCH)}"
        )
    return _SCHEDULER_DISPATCH[algorithm](topology, table, **kwargs)
```

- [ ] **Step 4: Run test, verify it passes**

Run: `.venv/bin/python -m pytest tests/test_schedule_from_algorithm.py -v`
Expected: PASS (4 tests, 3 parametrized + 1 negative).

- [ ] **Step 5: Commit**

```bash
git add twisted_analysis/io/schedule.py tests/test_schedule_from_algorithm.py
git commit -m "feat(io): add schedule_from_algorithm dispatcher"
```

---

## Task 8: Generalize `gen_orbit_greedy_kernel.py` with `--scheduler` flag

Add a `--scheduler` CLI flag that selects the algorithm; pipe its output through the same dest-table + kernel-emission code as before. Also: run the verifier on the produced schedule and fail loudly if violations exist.

**Files:**
- Modify: `pallas_kernel/gen_orbit_greedy_kernel.py`
- Modify: `tests/test_gen_orbit_greedy_kernel_pipeline.py` (extend existing)

- [ ] **Step 1: Write the failing test (extend existing pipeline test)**

Add at the end of `tests/test_gen_orbit_greedy_kernel_pipeline.py`:

```python
def test_cli_supports_scheduler_flag(tmp_path):
    """The --scheduler flag selects among the registered algorithms."""
    from pallas_kernel.gen_orbit_greedy_kernel import main

    rt_out = tmp_path / "rt.json"
    sched_out = tmp_path / "sched.json"
    kern_out = tmp_path / "kern.py"
    # 2x4 ILP — fast.
    rc = main([
        "--slice", "2,4",
        "--router", "ilp",
        "--scheduler", "orbit_greedy_full",
        "--routing-table-out", str(rt_out),
        "--schedule-out", str(sched_out),
        "--out", str(kern_out),
    ])
    assert rc == 0
    assert rt_out.exists()
    assert sched_out.exists()
    assert kern_out.exists()
    # Generated kernel should mention the scheduler in its docstring.
    src = kern_out.read_text()
    assert "orbit_greedy_full" in src


def test_cli_verifier_fails_on_violating_schedule(tmp_path, monkeypatch):
    """Verifier integration: pipeline raises if it produces a violating schedule."""
    from pallas_kernel import gen_orbit_greedy_kernel as gen

    # Monkeypatch the dispatcher to return a schedule with a known double-booking.
    def bad_schedule(algorithm, topology, table, **kwargs):
        return [
            {"round": 0, "src": 0, "dst": 1, "path": [0, 1]},
            {"round": 0, "src": 2, "dst": 1, "path": [2, 1]},  # different src->same edge? no, edge differs.
            # Force a real collision: both use edge (0,1) at t=0.
            {"round": 0, "src": 0, "dst": 1, "path": [0, 1]},
        ]
    monkeypatch.setattr(gen, "schedule_from_algorithm", bad_schedule)

    import pytest
    with pytest.raises(SystemExit, match="capacity violation"):
        gen.main([
            "--slice", "2,4",
            "--router", "ilp",
            "--scheduler", "orbit_greedy_full",
            "--routing-table-out", str(tmp_path / "rt.json"),
            "--schedule-out", str(tmp_path / "sched.json"),
            "--out", str(tmp_path / "kern.py"),
        ])
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/bin/python -m pytest tests/test_gen_orbit_greedy_kernel_pipeline.py::test_cli_supports_scheduler_flag tests/test_gen_orbit_greedy_kernel_pipeline.py::test_cli_verifier_fails_on_violating_schedule -v`
Expected: FAIL (CLI doesn't know `--scheduler`).

- [ ] **Step 3: Modify `pallas_kernel/gen_orbit_greedy_kernel.py`**

Change the import at line 42 to include the dispatcher:

```python
# OLD:
from twisted_analysis.io.schedule import save_schedule, schedule_from_orbit_greedy
# NEW:
from twisted_analysis.io.schedule import (
    save_schedule,
    schedule_from_algorithm,
)
from twisted_analysis.schedules.verify import verify_capacity
```

Inside `main()`, change the argparse block (around line 416) to add the `--scheduler` flag, and replace the schedule call:

```python
# Add this argparse line after the existing --order line:
p.add_argument(
    "--scheduler",
    default="orbit_greedy",
    choices=["orbit_greedy", "orbit_greedy_full", "literal_greedy", "ilp_literal"],
    help="Which scheduling algorithm to run on the routing table. "
         "orbit_greedy: original (dim, dir)-keyed greedy — only correct on "
         "translation-equivariant routings (DOR, ILP). "
         "orbit_greedy_full: same greedy but keyed on full physical-edge sets — "
         "correct on any translation-symmetric workload. "
         "literal_greedy: LMR-style per-flow earliest-feasible greedy. "
         "ilp_literal: exact ILP on the literal N*(N-1) flow set (small cells only).",
)
p.add_argument(
    "--ilp-time-limit-s",
    type=int,
    default=600,
    help="Time limit (s) for the ilp_literal solver. Ignored otherwise.",
)
```

Replace the Stage-2 schedule call (around line 466) with:

```python
# Stage 2: schedule.
sched_kwargs = {}
if args.scheduler in ("orbit_greedy", "orbit_greedy_full"):
    sched_kwargs["order"] = args.order
elif args.scheduler == "literal_greedy":
    # literal_greedy has its own valid orders; map lpt_tail_asc/lpt -> lpt.
    sched_kwargs["order"] = "lpt" if args.order == "lpt_tail_asc" else args.order
elif args.scheduler == "ilp_literal":
    sched_kwargs["time_limit_s"] = args.ilp_time_limit_s

schedule = schedule_from_algorithm(
    args.scheduler, topology, table, **sched_kwargs,
)
sched_path = args.schedule_out or (
    fixtures
    / f"schedule_{slice_slug}_{router_slug}_{args.scheduler}_{args.order}.json"
)
save_schedule(schedule, sched_path)
print(f"[2/4] wrote schedule     {sched_path}", file=sys.stderr)

# Stage 3 (new): verify physical-edge capacity.
violations = verify_capacity(schedule)
if violations:
    print(
        f"\nERROR: schedule has {len(violations)} physical-edge capacity violation(s). "
        f"First 3: {violations[:3]}",
        file=sys.stderr,
    )
    raise SystemExit(
        f"refusing to emit kernel for violating schedule "
        f"(scheduler={args.scheduler}, routing={rt_path}); "
        f"capacity violation count = {len(violations)}"
    )
print(f"[3/4] verified schedule  ({len(schedule)} flows, 0 violations)", file=sys.stderr)
```

Update the final stage number and the kernel-emit doc string. Change the existing `[3/3] wrote kernel` line to `[4/4] wrote kernel`. Pass the scheduler name down so the kernel doc mentions it: extend `generate_kernel_source` signature by adding a `scheduler_name: str` keyword, and add a `Scheduler:` line to the kernel's header doc block (insert after the existing `Router:` line):

```python
# In generate_kernel_source, after the `Router:` line (around line 144):
L.append(f'Scheduler:       {scheduler_name}')
```

And pass `scheduler_name=args.scheduler` into the `generate_kernel_source(...)` call in `main()`.

- [ ] **Step 4: Run test, verify it passes**

Run: `.venv/bin/python -m pytest tests/test_gen_orbit_greedy_kernel_pipeline.py -v`
Expected: ALL PASS (the existing tests still pass, the two new tests pass).

- [ ] **Step 5: Commit**

```bash
git add pallas_kernel/gen_orbit_greedy_kernel.py tests/test_gen_orbit_greedy_kernel_pipeline.py
git commit -m "feat(gen): --scheduler flag + post-schedule capacity verification"
```

---

## Task 9: Regenerate fixtures and kernel files for 8×4×4 (loaded routing) under each algorithm

Run the generalized pipeline against `fixtures/routing_table_8x4x4_twist.json` for each algorithm. The original `orbit_greedy` is expected to FAIL the verifier (per Task 2) — that's fine and lockable in a test. The other three produce clean fixtures + kernels.

**Files:**
- Create: `eval/regenerate_8x4x4_all_schedulers.sh`
- Modify (expected output): `fixtures/schedule_8x4x4_loaded_orbit_greedy_full_lpt_tail_asc.json`, `fixtures/schedule_8x4x4_loaded_literal_greedy_lpt_tail_asc.json`, plus matching `pallas_kernel/outputs/_ragged_a2a_kernel_orbit_greedy_8_4_4_*.py` files.

- [ ] **Step 1: Write the regeneration script**

```bash
#!/usr/bin/env bash
# eval/regenerate_8x4x4_all_schedulers.sh
#
# Re-run the kernel-generation pipeline against the loaded 8x4x4 routing
# table for every scheduler we support, so each algorithm has fresh
# fixtures + kernel for benchmarking.
#
# orbit_greedy is expected to fail the post-schedule verifier (the whole
# point of this plan); we mark its failure as expected and continue.
#
# ilp_literal is intractable at N=128 within a 10-minute time budget; we
# skip it here but provide a stub line for documentation.
set -u

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_ROOT"

ROUTING_TABLE="fixtures/routing_table_8x4x4_twist.json"
SLICE="8,4,4"
PY=".venv/bin/python"

run() {
  local sched="$1"
  local order="${2:-lpt_tail_asc}"
  local out_kern="pallas_kernel/outputs/_ragged_a2a_kernel_${sched}_8_4_4.py"
  local out_sched="fixtures/schedule_8x4x4_loaded_${sched}_${order}.json"
  echo "=== Running scheduler=${sched} order=${order} ==="
  if "$PY" pallas_kernel/gen_orbit_greedy_kernel.py \
        --slice "$SLICE" \
        --routing-table "$ROUTING_TABLE" \
        --scheduler "$sched" \
        --order "$order" \
        --schedule-out "$out_sched" \
        --out "$out_kern"; then
    echo "OK: $sched -> $out_kern"
  else
    echo "FAILED (likely capacity-violation refusal): $sched"
  fi
  echo
}

# Expected to fail: original orbit_greedy on loaded routing.
run orbit_greedy lpt_tail_asc || true

# Expected to succeed.
run orbit_greedy_full lpt_tail_asc
run literal_greedy lpt

# ilp_literal at N=128 has ~16k binary vars * ~80 time slots; CBC won't
# solve in <10 min. Documented as future work.
echo "# Skipping ilp_literal on 8x4x4 (intractable at this scale)."
```

- [ ] **Step 2: Make it executable and run**

```bash
chmod +x eval/regenerate_8x4x4_all_schedulers.sh
./eval/regenerate_8x4x4_all_schedulers.sh
```

Expected output:
- `orbit_greedy`: FAILED with "refusing to emit kernel for violating schedule (capacity violation count = N)".
- `orbit_greedy_full`: OK; new `_ragged_a2a_kernel_orbit_greedy_full_8_4_4.py` written.
- `literal_greedy`: OK; new `_ragged_a2a_kernel_literal_greedy_8_4_4.py` written.
- Final line: `# Skipping ilp_literal on 8x4x4 ...`.

- [ ] **Step 3: Sanity-check the new schedule fixtures**

```bash
.venv/bin/python -c "
from twisted_analysis.io.schedule import load_schedule
from twisted_analysis.schedules.verify import verify_capacity, schedule_makespan
for p in [
    'fixtures/schedule_8x4x4_loaded_orbit_greedy_full_lpt_tail_asc.json',
    'fixtures/schedule_8x4x4_loaded_literal_greedy_lpt.json',
]:
    s = load_schedule(p)
    v = verify_capacity(s)
    print(f'{p}: makespan={schedule_makespan(s)} violations={len(v)} flows={len(s)}')
"
```

Expected: both lines show `violations=0` and `flows=16256`. Makespans may differ — record them.

- [ ] **Step 4: Commit**

```bash
git add eval/regenerate_8x4x4_all_schedulers.sh \
        fixtures/schedule_8x4x4_loaded_orbit_greedy_full_lpt_tail_asc.json \
        fixtures/schedule_8x4x4_loaded_literal_greedy_lpt.json \
        pallas_kernel/outputs/_ragged_a2a_kernel_orbit_greedy_full_8_4_4.py \
        pallas_kernel/outputs/_ragged_a2a_kernel_literal_greedy_8_4_4.py
git commit -m "fixtures: regenerate 8x4x4 schedules + kernels under each scheduler"
```

---

## Task 10: Decision-point follow-through (Task 3 result handling)

Implement the conditional cleanup based on Task 3's result.

**Files:**
- Possibly modify: `twisted_analysis/schedules/orbit_greedy.py`
- Possibly modify: `twisted_analysis/io/schedule.py`
- Modify: `docs/orbit_greedy_optimality.md` to add a note about scope.

- [ ] **Step 1: Branch on Task 3 result**

Read the decision recorded in this plan's header (added in Task 3 Step 3).

**Case A — orbit_greedy is correct under ILP routings (the expected outcome):**

- [ ] **Step A1: Add a scope note to `docs/orbit_greedy_optimality.md`**

Append the following paragraph at the end of `## 6. Open Questions` in `docs/orbit_greedy_optimality.md`:

```markdown
6. **Routings without (dim, dir) translation-equivariance.** The proof's
   `(dim, dir)` edge-orbit class abstraction assumes the routing puts
   physical edges in disjoint translation-orbits keyed by `(dim, dir)`.
   DOR and the ILP-router both satisfy this by construction. The TPU
   OCS-derived "loaded" routing in `fixtures/routing_table_8x4x4_twist.json`
   does *not*: physical edges of the same `(dim, dir)` are shared across
   multiple translation orbits when the wrap interacts with the twist.
   For such routings, use `OrbitGreedyFullSchedule`
   (`twisted_analysis.schedules.orbit_greedy_full`), which keys the busy
   table on the full set of physical edges an orbit's `N` flows traverse
   at each hop. It produces equivalent schedules on DOR/ILP cells and
   capacity-feasible schedules on loaded routings.
```

- [ ] **Step A2: Add a wrapper schedule class in `twisted_analysis/schedules/orbit_greedy_full.py`**

Append to that file:

```python
from dataclasses import dataclass

from twisted_analysis.model.flow import AllToAll
from twisted_analysis.schedules.base import Injection
from twisted_analysis.schedules.lp_symmetric import (
    symmetric_assignment_to_injections,
)


@dataclass
class OrbitGreedyFullSchedule:
    """Schedule wrapper: orbit_greedy_full with the canonical Injection format.

    Use this when you need the same `Schedule` Protocol surface that
    `OrbitGreedySchedule` exposes, with full-physical-edge accounting under
    the hood. See module docstring for when each is appropriate.
    """
    order: str = "lpt_tail_asc"
    name: str = "orbit_greedy_full"

    def emit(self, workload: AllToAll) -> list[Injection]:
        # Build the routing table from the workload's router, then compute
        # the assignment dict in the same format the legacy adapter expects.
        from twisted_analysis.io.coords import flatten

        topology = workload.topology
        router = workload.router
        n = topology.n_nodes
        table: list[list[list[int]]] = [
            [[] for _ in range(n)] for _ in range(n)
        ]
        for src in topology.nodes():
            src_flat = flatten(src, topology.slice)
            for dst in topology.nodes():
                dst_flat = flatten(dst, topology.slice)
                if src == dst:
                    table[src_flat][dst_flat] = [src_flat]
                    continue
                path = router.path(src, dst)
                table[src_flat][dst_flat] = [src_flat] + [
                    flatten(v, topology.slice) for (_u, v, _, _) in path
                ]
        t0 = compute_hop0_firing_times_full(topology, table, order=self.order)

        # Convert to (orbit_id, hop_i, t) assignment expected by
        # symmetric_assignment_to_injections. Each hop fires PIPELINED at t0+i.
        from twisted_analysis.lp.orbit import compute_orbits
        orbits = compute_orbits(topology)
        per_orbit = _orbit_hop_edge_sets(topology, table)
        assignment: dict[tuple, float] = {}
        for orbit_id in orbits:
            hops = per_orbit[orbit_id]
            for i, _edges in enumerate(hops):
                assignment[(orbit_id, i, int(t0[orbit_id]) + i)] = 1.0
        return symmetric_assignment_to_injections(
            topology, router, list(workload.flows), assignment,
        )
```

- [ ] **Step A3: Commit**

```bash
git add docs/orbit_greedy_optimality.md twisted_analysis/schedules/orbit_greedy_full.py
git commit -m "docs: scope note for non-(dim,dir)-equivariant routings"
```

**Case B — orbit_greedy fails on some ILP cell (replace it):**

- [ ] **Step B1: Replace `_emit_orbit_greedy` body in `twisted_analysis/schedules/orbit_greedy.py`**

Locate the function `_emit_orbit_greedy` (around line 87) and replace its body with a call into `orbit_greedy_full`:

```python
def _emit_orbit_greedy(
    topology: Topology, router: Router, order: str,
) -> dict[tuple, float]:
    """Delegate to orbit_greedy_full (the (dim, dir) keying was unsound;
    see plan 2026-05-15-multi-algorithm-scheduling.md Task 3)."""
    from twisted_analysis.io.coords import flatten
    from twisted_analysis.lp.orbit import compute_orbits
    from twisted_analysis.schedules.orbit_greedy_full import (
        _orbit_hop_edge_sets,
        compute_hop0_firing_times_full,
    )

    n = topology.n_nodes
    table = [[[] for _ in range(n)] for _ in range(n)]
    for src in topology.nodes():
        src_flat = flatten(src, topology.slice)
        for dst in topology.nodes():
            dst_flat = flatten(dst, topology.slice)
            if src == dst:
                table[src_flat][dst_flat] = [src_flat]
                continue
            path = router.path(src, dst)
            table[src_flat][dst_flat] = [src_flat] + [
                flatten(v, topology.slice) for (_u, v, _, _) in path
            ]
    t0 = compute_hop0_firing_times_full(topology, table, order=order)
    per_orbit = _orbit_hop_edge_sets(topology, table)
    orbits = compute_orbits(topology)
    assignment: dict[tuple, float] = {}
    for orbit_id in orbits:
        for i, _edges in enumerate(per_orbit[orbit_id]):
            assignment[(orbit_id, i, int(t0[orbit_id]) + i)] = 1.0
    return assignment
```

- [ ] **Step B2: Note the change in `docs/orbit_greedy_optimality.md`**

Add the following at the end of `## 6. Open Questions`:

```markdown
**Note (2026-05-XX):** The `(dim, dir)` keying empirically failed on
`<cell>`-`<router>` (see `tests/test_orbit_greedy_dimdir_correctness.py`).
`OrbitGreedySchedule` now delegates to `OrbitGreedyFullSchedule`. Sections
4.3.1–4.3.17 still hold *under the full-physical-edge formulation*; the
(dim, dir) shorthand was incorrect for non-equivariant routings.
```

- [ ] **Step B3: Commit**

```bash
git add twisted_analysis/schedules/orbit_greedy.py docs/orbit_greedy_optimality.md
git commit -m "fix(schedules): orbit_greedy delegates to orbit_greedy_full"
```

---

## Task 11: Update top-level README and pallas_kernel README

**Files:**
- Modify: `pallas_kernel/README.md`
- Possibly modify: `README.md` (project root)

- [ ] **Step 1: Add the new algorithms to `pallas_kernel/README.md`**

Find the "What problem this kernel solves" section in `pallas_kernel/README.md`. After it, add a new section:

```markdown
## Scheduler choice

The generator supports four scheduling algorithms via `--scheduler`:

| Scheduler | Approach | When to use |
|---|---|---|
| `orbit_greedy` (default) | Orbit greedy keyed on `(dim, dir)` edge classes | DOR/ILP routings (translation-equivariant). Provably LB-tight on the (S, 2S) cells in `docs/orbit_greedy_optimality.md`. |
| `orbit_greedy_full` | Orbit greedy keyed on full physical-edge sets | Loaded TPU routings or any case where (dim, dir) does not partition physical edges cleanly. Capacity-feasible by construction. |
| `literal_greedy` | LMR-style per-flow earliest-feasible greedy | Sanity baseline; works on any routing. Makespan is bounded by O(c + d) (LMR), constants are uncomputed but practical. |
| `ilp_literal` | Exact ILP on the literal N(N-1) flow set | Small cells only (≤ 32 nodes); intractable at N = 128. Use as a ground-truth oracle. |

The post-schedule capacity verifier refuses to emit a kernel whose schedule
has any physical-edge collisions, so a `--scheduler` choice that doesn't
fit the routing fails fast at generation time.

Example invocations:

\`\`\`bash
# Loaded TPU routing on 8x4x4 — use orbit_greedy_full or literal_greedy:
python pallas_kernel/gen_orbit_greedy_kernel.py \
    --slice 8,4,4 \
    --routing-table fixtures/routing_table_8x4x4_twist.json \
    --scheduler orbit_greedy_full

# Small-cell oracle:
python pallas_kernel/gen_orbit_greedy_kernel.py \
    --slice 2,4 --router ilp \
    --scheduler ilp_literal --ilp-time-limit-s 60
\`\`\`
```

- [ ] **Step 2: Commit**

```bash
git add pallas_kernel/README.md
git commit -m "docs(pallas_kernel): document new scheduler choices"
```

---

## Task 12: Final integration test — full pipeline run

A smoke test that exercises the entire chain end-to-end with each scheduler that supports N=32 (2×4×4). This is the regression net that prevents future refactors from breaking the integration.

**Files:**
- Modify: `tests/test_gen_orbit_greedy_kernel_pipeline.py`

- [ ] **Step 1: Add an end-to-end parametrized test**

Append to `tests/test_gen_orbit_greedy_kernel_pipeline.py`:

```python
import pytest


@pytest.mark.parametrize("scheduler", [
    "orbit_greedy", "orbit_greedy_full", "literal_greedy",
])
def test_end_to_end_pipeline_2x4x4_ilp(tmp_path, scheduler):
    """Whole-pipeline smoke test: generate routing -> schedule -> verify -> kernel."""
    from pallas_kernel.gen_orbit_greedy_kernel import main

    rt_out = tmp_path / "rt.json"
    sched_out = tmp_path / "sched.json"
    kern_out = tmp_path / "kern.py"
    rc = main([
        "--slice", "2,4,4",
        "--router", "ilp",
        "--scheduler", scheduler,
        "--routing-table-out", str(rt_out),
        "--schedule-out", str(sched_out),
        "--out", str(kern_out),
    ])
    assert rc == 0
    assert rt_out.exists() and sched_out.exists() and kern_out.exists()

    # The generated kernel mentions both router and scheduler in its docstring.
    src = kern_out.read_text()
    assert scheduler in src
    assert "ILP" in src or "loaded" in src
```

- [ ] **Step 2: Run the test**

Run: `.venv/bin/python -m pytest tests/test_gen_orbit_greedy_kernel_pipeline.py::test_end_to_end_pipeline_2x4x4_ilp -v`
Expected: 3 PASSes.

- [ ] **Step 3: Run the whole test suite to ensure nothing else broke**

Run: `.venv/bin/python -m pytest tests/ -x -q`
Expected: all green, except possibly slow tests (`test_symmetric_scale`, `test_ilp.py`) — skip those with `-k "not symmetric_scale and not slow"` if needed.

- [ ] **Step 4: Commit**

```bash
git add tests/test_gen_orbit_greedy_kernel_pipeline.py
git commit -m "test: end-to-end pipeline parametrized over all schedulers"
```

---

## Self-Review Notes

**Spec coverage check:**
- Option 1 (orbit_greedy with full-physical-edge accounting) — Task 4. ✓
- Option 2 (LMR-style literal-flow greedy) — Task 5. ✓
- Option 3 (Exact ILP on literal flows) — Task 6. ✓
- Generalize `gen_orbit_greedy_kernel.py` — Task 8. ✓
- Decide whether to keep current orbit_greedy via empirical test on ILP routing — Task 3 + Task 10. ✓
- Generate new schedules and kernel code — Task 9. ✓

**Type / signature consistency:**
- All `schedule_from_*` adapters return `list[dict]` (schedule entries). ✓
- All scheduler internals consume a routing `table: list[list[list[int]]]`. ✓
- Verifier returns `list[CapacityViolation]`. ✓
- `schedule_from_algorithm` dispatcher takes algorithm name + topology + table, passes through kwargs. ✓
- `compute_hop0_firing_times_full` mirrors `compute_hop0_firing_times`'s signature (returns dict[OrbitId, int]). ✓

**Placeholder scan:** No TBDs, no "implement later", no "similar to". Each step contains the literal code.

**Risk notes for the executing engineer:**
- `ilp_literal` on 8×4×4 is intentionally skipped — CBC will not solve at that scale within practical limits.
- The `_orbit_hop_edge_sets` builder assumes the routing table is `path[src][dst] = [node_id, ...]` and that consecutive node-ids are physical neighbors. The existing `validate_routing_table_shape` in `twisted_analysis/io/routing_table.py` covers shape but not neighbor-feasibility; if you suspect a malformed routing, add a `topology.neighbor`-verification pass.
- The Case B branch in Task 10 is conditional. If Task 3 passes (the expected outcome), only Case A runs.
- `OrbitGreedyFullSchedule.emit` in Task 10 Step A2 reconstructs the routing table from a `Router` — this is the same conversion done in `save_routing_table` and is intentionally duplicated to keep the `Schedule` Protocol surface stable.
