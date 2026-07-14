# Ragged A2A Scheduling (fluid + greedy) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Schedule a ragged (per-pair-sized) AllToAll on the loaded 8×4×4 twisted-torus routing, adding `rate`/`size` chunk semantics to the schedule format, a closed-form fluid scheduler that certifies the LB, and an integral earliest-feasible greedy (non-preemptive + preemptive).

**Architecture:** New flat-ID `RaggedWorkload` model + workload loader; pipelined-stream time model (chunk `{round=r, rate=ρ, size}` occupies path edge `i` during `[r+i, r+i+(size/quantum)/ρ)` quanta, Σρ ≤ 1 per edge) that reduces exactly to the existing uniform model at `rate=1, size=1, quantum=1`; two schedulers behind the existing adapter/dispatch/CLI pattern; sweepline verifier alongside the existing set-based one.

**Tech Stack:** Pure Python 3 (stdlib only for new code), pytest, existing `twisted_analysis` package layout.

**Spec:** `docs/superpowers/specs/2026-07-14-ragged-a2a-scheduling-design.md` — read it first.

## Global Constraints

- Time is measured in **quanta** (quantum = gcd of workload sizes; 32 for the shipped fixture). Entry `size` is in **workload units** (bytes), not quanta.
- Hop offset is **+i quanta per hop** (constant pipeline fill), never `i/rate`.
- Chunk finish time = `round + (L−1) + (size/quantum)/rate`, where `L = len(path) − 1` hops. Legacy entries (`rate=1, size=1, quantum=1`) must give `round + L`, matching `schedule_makespan`.
- Existing uniform schedules must remain valid: all new entry fields optional (`rate` default 1.0, `size` default 1); existing `verify.py` functions untouched.
- Fixture ground truth (assert in tests): workload has 16,256 flows, quantum 32, LB = 12,608 units = 394 quanta on `fixtures/routing_table_8x4x4_twist.json`; max path length 6 hops; fluid makespan = 399.0 quanta; uniform all-pairs demand through the same code path gives LB 75.
- Run tests with `.venv/bin/python -m pytest` (NOT `uv run` — it may re-sync the env).
- Every commit message ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: `RaggedWorkload` model

**Files:**
- Create: `twisted_analysis/model/ragged.py`
- Test: `tests/test_ragged_workload.py`

**Interfaces:**
- Consumes: routing tables in the loaded form `table[src][dst] = [flat-id, ...]` (from `twisted_analysis.io.routing_table.load_routing_table`).
- Produces: `RaggedWorkload(demand: Mapping[tuple[int,int], int])` frozen dataclass with `quantum: int` (cached property), `link_load(table) -> dict[tuple[int,int], int]`, `lower_bound(table) -> int`, `bottleneck_edges(table) -> list[tuple[int,int]]`. Tasks 2, 5, 6, 7 all consume exactly these names.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ragged_workload.py`:

```python
"""RaggedWorkload model: quantum, link_load, lower_bound, validation."""
import pytest

from twisted_analysis.model.ragged import RaggedWorkload

# 3-node line 0-1-2, same shape load_routing_table returns.
LINE_TABLE = [
    [[0], [0, 1], [0, 1, 2]],
    [[1, 0], [1], [1, 2]],
    [[2, 1, 0], [2, 1], [2]],
]


def test_quantum_is_gcd_of_sizes():
    w = RaggedWorkload(demand={(0, 2): 64, (1, 2): 32, (0, 1): 96})
    assert w.quantum == 32


def test_link_load_is_size_weighted():
    w = RaggedWorkload(demand={(0, 2): 64, (1, 2): 32})
    assert w.link_load(LINE_TABLE) == {(0, 1): 64, (1, 2): 96}


def test_lower_bound_and_bottleneck():
    w = RaggedWorkload(demand={(0, 2): 64, (1, 2): 32})
    assert w.lower_bound(LINE_TABLE) == 96
    assert w.bottleneck_edges(LINE_TABLE) == [(1, 2)]


def test_rejects_self_pair():
    with pytest.raises(ValueError, match="self-pair"):
        RaggedWorkload(demand={(1, 1): 32})


def test_rejects_nonpositive_size():
    with pytest.raises(ValueError, match="positive int"):
        RaggedWorkload(demand={(0, 1): 0})


def test_rejects_bool_size():
    with pytest.raises(ValueError, match="positive int"):
        RaggedWorkload(demand={(0, 1): True})


def test_rejects_empty():
    with pytest.raises(ValueError, match="at least one flow"):
        RaggedWorkload(demand={})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ragged_workload.py -v`
Expected: FAIL / error with `ModuleNotFoundError: No module named 'twisted_analysis.model.ragged'`

- [ ] **Step 3: Write the implementation**

Create `twisted_analysis/model/ragged.py`:

```python
"""Ragged (per-pair-sized) AllToAll workload over flat node IDs.

Unlike `twisted_analysis.model.flow.AllToAll` (coordinate-based, uniform
msg_size), a ragged workload arrives as flat-ID (src, dst) -> size pairs
loaded from JSON, and its paths come from an already-loaded routing table
(`table[src][dst] = [flat-id, ...]`).

Sizes are in workload units (bytes for the shipped fixtures). The quantum
is the gcd of all sizes; schedule time is measured in quanta. See
docs/superpowers/specs/2026-07-14-ragged-a2a-scheduling-design.md.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from math import gcd
from typing import Mapping

Edge = tuple[int, int]


@dataclass(frozen=True)
class RaggedWorkload:
    demand: Mapping[tuple[int, int], int]  # (src, dst) -> size > 0

    def __post_init__(self) -> None:
        if not self.demand:
            raise ValueError("workload must contain at least one flow")
        for (s, d), size in self.demand.items():
            if s == d:
                raise ValueError(f"self-pair ({s}, {d}) not allowed")
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                raise ValueError(
                    f"flow ({s}, {d}): size must be positive int, got {size!r}"
                )

    @cached_property
    def quantum(self) -> int:
        q = 0
        for size in self.demand.values():
            q = gcd(q, size)
        return q

    def link_load(self, table: list[list[list[int]]]) -> dict[Edge, int]:
        """Size-weighted directed-edge load (same convention as AllToAll)."""
        c: Counter[Edge] = Counter()
        for (s, d), size in self.demand.items():
            path = table[s][d]
            for u, v in zip(path, path[1:]):
                c[(u, v)] += size
        return dict(c)

    def lower_bound(self, table: list[list[list[int]]]) -> int:
        """Max size-weighted edge load: hard makespan LB in workload units."""
        return max(self.link_load(table).values())

    def bottleneck_edges(self, table: list[list[list[int]]]) -> list[Edge]:
        loads = self.link_load(table)
        lb = max(loads.values())
        return [e for e, load in loads.items() if load == lb]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ragged_workload.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add twisted_analysis/model/ragged.py tests/test_ragged_workload.py
git commit -m "Add RaggedWorkload model

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Workload I/O + fixture ground-truth tests

**Files:**
- Create: `twisted_analysis/io/workload.py`
- Test: `tests/test_ragged_workload_io.py`

**Interfaces:**
- Consumes: `RaggedWorkload` from Task 1.
- Produces: `load_workload(path: Path | str) -> RaggedWorkload`. The returned `demand` dict preserves file order (Python dict insertion order) — Task 6's `natural` order relies on this. Tasks 7–8 consume `load_workload`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ragged_workload_io.py`:

```python
"""Workload JSON loader: validation, file-order preservation, fixture truth."""
import json
from pathlib import Path

import pytest

from twisted_analysis.io.routing_table import load_routing_table
from twisted_analysis.io.workload import load_workload

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
WORKLOAD_FIXTURE = FIXTURES / "ragged_a2a_workload_node_128_min_32_max_1024_discrete.json"
ROUTING_FIXTURE = FIXTURES / "routing_table_8x4x4_twist.json"


def test_load_small_workload_preserves_file_order(tmp_path):
    p = tmp_path / "w.json"
    p.write_text(json.dumps([
        {"src": 2, "dst": 0, "size": 96},
        {"src": 0, "dst": 1, "size": 32},
    ]))
    w = load_workload(p)
    assert list(w.demand.items()) == [((2, 0), 96), ((0, 1), 32)]
    assert w.quantum == 32


def test_rejects_duplicate_pair(tmp_path):
    p = tmp_path / "w.json"
    p.write_text(json.dumps([
        {"src": 0, "dst": 1, "size": 32},
        {"src": 0, "dst": 1, "size": 64},
    ]))
    with pytest.raises(ValueError, match="duplicate pair"):
        load_workload(p)


def test_rejects_missing_key(tmp_path):
    p = tmp_path / "w.json"
    p.write_text(json.dumps([{"src": 0, "dst": 1}]))
    with pytest.raises(ValueError, match="missing 'size'"):
        load_workload(p)


def test_rejects_non_list_toplevel(tmp_path):
    p = tmp_path / "w.json"
    p.write_text(json.dumps({"src": 0}))
    with pytest.raises(ValueError, match="top-level must be a list"):
        load_workload(p)


def test_fixture_ground_truth():
    """Spec key quantities: 16,256 flows, quantum 32, LB 12,608 = 394 quanta."""
    w = load_workload(WORKLOAD_FIXTURE)
    table = load_routing_table(ROUTING_FIXTURE)
    assert len(w.demand) == 128 * 127
    assert w.quantum == 32
    assert w.lower_bound(table) == 12_608
    assert w.lower_bound(table) // w.quantum == 394
    assert max(
        len(table[s][d]) - 1 for (s, d) in w.demand
    ) == 6


def test_uniform_demand_reproduces_lb_75():
    """Uniform all-pairs size-1 demand through RaggedWorkload matches prior LB."""
    from twisted_analysis.model.ragged import RaggedWorkload

    table = load_routing_table(ROUTING_FIXTURE)
    n = len(table)
    w = RaggedWorkload(demand={
        (s, d): 1 for s in range(n) for d in range(n) if s != d
    })
    assert w.quantum == 1
    assert w.lower_bound(table) == 75
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ragged_workload_io.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'twisted_analysis.io.workload'`

- [ ] **Step 3: Write the implementation**

Create `twisted_analysis/io/workload.py`:

```python
"""Ragged-workload on-disk I/O.

Format: top-level list of `{"src": int, "dst": int, "size": int}` dicts
(matches fixtures/ragged_a2a_workload_node_128_min_32_max_1024_discrete.json).
Pair order in the file is preserved in the returned demand dict; the
`natural` scheduler order iterates it.
"""
from __future__ import annotations
import json
from pathlib import Path

from twisted_analysis.model.ragged import RaggedWorkload


def load_workload(path: Path | str) -> RaggedWorkload:
    path = Path(path)
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(
            f"{path}: top-level must be a list, got {type(raw).__name__}"
        )
    demand: dict[tuple[int, int], int] = {}
    for i, e in enumerate(raw):
        if not isinstance(e, dict):
            raise ValueError(
                f"{path}: entry {i} must be a dict, got {type(e).__name__}"
            )
        for k in ("src", "dst", "size"):
            if k not in e:
                raise ValueError(f"{path}: entry {i} missing {k!r}")
            if not isinstance(e[k], int) or isinstance(e[k], bool):
                raise ValueError(
                    f"{path}: entry {i}: {k}={e[k]!r} must be int, "
                    f"got {type(e[k]).__name__}"
                )
        key = (e["src"], e["dst"])
        if key in demand:
            raise ValueError(f"{path}: duplicate pair {key} at entry {i}")
        demand[key] = e["size"]
    return RaggedWorkload(demand=demand)
```

(`size > 0` and `src != dst` are enforced by `RaggedWorkload.__post_init__`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ragged_workload_io.py -v`
Expected: 6 passed (fixture tests take a few seconds — the routing table is 128×128).

- [ ] **Step 5: Commit**

```bash
git add twisted_analysis/io/workload.py tests/test_ragged_workload_io.py
git commit -m "Add ragged workload loader + fixture ground-truth tests

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Optional `rate`/`size` fields in schedule validation

**Files:**
- Modify: `twisted_analysis/io/schedule.py` (function `_validate`, after the `path` endpoint checks inside the entry loop, currently around line 48)
- Test: `tests/test_io_schedule.py` (append new tests)

**Interfaces:**
- Consumes: existing `_validate` / `save_schedule` / `load_schedule` in `twisted_analysis/io/schedule.py`.
- Produces: entries may carry optional `rate` (number, `0 < rate <= 1`) and `size` (positive int); both round-trip through `save_schedule`/`load_schedule` (extra keys already pass through — this task only adds validation). Tasks 5–8 emit such entries.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_io_schedule.py`:

```python
def test_rate_and_size_fields_roundtrip(tmp_path):
    from twisted_analysis.io.schedule import load_schedule, save_schedule

    entries = [{
        "round": 0, "src": 0, "dst": 2, "path": [0, 1, 2],
        "rate": 0.5, "size": 128,
    }]
    p = tmp_path / "s.json"
    save_schedule(entries, p)
    loaded = load_schedule(p)
    assert loaded[0]["rate"] == 0.5
    assert loaded[0]["size"] == 128


def test_rate_zero_rejected(tmp_path):
    import pytest
    from twisted_analysis.io.schedule import save_schedule

    entries = [{"round": 0, "src": 0, "dst": 1, "path": [0, 1], "rate": 0.0}]
    with pytest.raises(ValueError, match="rate"):
        save_schedule(entries, tmp_path / "s.json")


def test_rate_above_one_rejected(tmp_path):
    import pytest
    from twisted_analysis.io.schedule import save_schedule

    entries = [{"round": 0, "src": 0, "dst": 1, "path": [0, 1], "rate": 1.5}]
    with pytest.raises(ValueError, match="rate"):
        save_schedule(entries, tmp_path / "s.json")


def test_size_zero_rejected(tmp_path):
    import pytest
    from twisted_analysis.io.schedule import save_schedule

    entries = [{"round": 0, "src": 0, "dst": 1, "path": [0, 1], "size": 0}]
    with pytest.raises(ValueError, match="size"):
        save_schedule(entries, tmp_path / "s.json")
```

- [ ] **Step 2: Run tests to verify the validation ones fail**

Run: `.venv/bin/python -m pytest tests/test_io_schedule.py -v -k "rate or size"`
Expected: `test_rate_and_size_fields_roundtrip` PASSES already (extra keys pass through); the three rejection tests FAIL (no ValueError raised).

- [ ] **Step 3: Add validation**

In `twisted_analysis/io/schedule.py`, function `_validate`, insert immediately before `out.append(dict(e))`:

```python
        if "rate" in e:
            rate = e["rate"]
            if isinstance(rate, bool) or not isinstance(rate, (int, float)):
                raise ValueError(f"entry {i}: rate={rate!r} must be a number")
            if not (0 < rate <= 1):
                raise ValueError(f"entry {i}: rate={rate!r} must be in (0, 1]")
        if "size" in e:
            size = e["size"]
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise ValueError(
                    f"entry {i}: size={size!r} must be a positive int"
                )
```

Also update the module docstring's format line (top of file) from
`{"round": int, "src": int, "dst": int, "path": [int, ...]}` to mention:
optional `"rate"` (float in (0, 1], default 1.0) and `"size"` (positive
int in workload units, default 1) for ragged schedules.

- [ ] **Step 4: Run the full existing schedule-io test file**

Run: `.venv/bin/python -m pytest tests/test_io_schedule.py -v`
Expected: all pass (new + pre-existing; no regression on legacy entries, which lack both fields).

- [ ] **Step 5: Commit**

```bash
git add twisted_analysis/io/schedule.py tests/test_io_schedule.py
git commit -m "Validate optional rate/size schedule fields

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Ragged verifier + makespan + coverage check

**Files:**
- Modify: `twisted_analysis/schedules/verify.py` (append; existing functions untouched)
- Test: `tests/test_ragged_verify.py`

**Interfaces:**
- Consumes: schedule entries (list of dicts with optional `rate`/`size`), `RaggedWorkload` from Task 1, existing `verify_capacity`/`schedule_makespan` for the compat test.
- Produces (Tasks 5, 6, 8 consume these exact signatures):
  - `RateViolation(edge: tuple[int,int], time: float, total_rate: float, flows: tuple[tuple[int,int,int], ...])` frozen dataclass
  - `verify_capacity_ragged(schedule, *, quantum: int = 1, tol: float = 1e-6) -> list[RateViolation]`
  - `schedule_makespan_ragged(schedule, *, quantum: int = 1) -> float`
  - `verify_workload_coverage(schedule, workload: RaggedWorkload) -> list[str]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ragged_verify.py`:

```python
"""Ragged verifier: pipelined-stream time model, sweepline capacity, coverage."""
from pathlib import Path

from twisted_analysis.io.schedule import load_schedule
from twisted_analysis.model.ragged import RaggedWorkload
from twisted_analysis.schedules.verify import (
    schedule_makespan,
    schedule_makespan_ragged,
    verify_capacity,
    verify_capacity_ragged,
    verify_workload_coverage,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _chunk(round_, src, dst, path, rate, size):
    return {"round": round_, "src": src, "dst": dst, "path": path,
            "rate": rate, "size": size}


def test_time_model_single_chunk():
    """Spec test 6: 2-hop chunk (m=4 quanta, rate=0.5) at r=0, quantum=32:
    edge 0 occupied [0, 8), edge 1 occupied [1, 9); finish = 0 + 1 + 8 = 9."""
    sched = [_chunk(0, 0, 2, [0, 1, 2], 0.5, 128)]
    assert verify_capacity_ragged(sched, quantum=32) == []
    assert schedule_makespan_ragged(sched, quantum=32) == 9.0


def test_capacity_violation_detected():
    """Chunk A holds edge (1,2) at rate 0.5 over [1, 9); chunk B holds it at
    rate 0.6 over [0, 10/3). Overlap [1, 10/3) sums to 1.1 > 1."""
    sched = [
        _chunk(0, 0, 2, [0, 1, 2], 0.5, 128),
        _chunk(0, 1, 2, [1, 2], 0.6, 64),
    ]
    violations = verify_capacity_ragged(sched, quantum=32)
    assert len(violations) == 1
    v = violations[0]
    assert v.edge == (1, 2)
    assert v.time == 1.0
    assert abs(v.total_rate - 1.1) < 1e-9
    assert v.flows == ((0, 0, 2), (0, 1, 2))


def test_rates_summing_to_one_are_feasible():
    sched = [
        _chunk(0, 0, 2, [0, 1, 2], 0.5, 128),
        _chunk(0, 1, 2, [1, 2], 0.5, 64),
    ]
    assert verify_capacity_ragged(sched, quantum=32) == []


def test_half_open_intervals_do_not_collide():
    """A ends on edge (0,1) exactly when B starts: [0,2) then [2,3) is legal
    at full rate."""
    sched = [
        _chunk(0, 0, 1, [0, 1], 1.0, 64),
        _chunk(2, 0, 1, [0, 1], 1.0, 32),
    ]
    # Both entries are the same (src, dst) pair; capacity is what's under test.
    assert verify_capacity_ragged(sched, quantum=32) == []


def test_legacy_schedule_verifies_and_matches_makespan():
    """Spec test 5: defaults (rate=1, size=1, quantum=1) reproduce the
    existing uniform semantics on a real fixture."""
    sched = load_schedule(
        FIXTURES / "schedule_8x4x4_loaded_cpsat_literal_warm.json"
    )
    assert verify_capacity(sched) == []
    assert verify_capacity_ragged(sched) == []
    assert schedule_makespan_ragged(sched) == float(schedule_makespan(sched))


def test_workload_coverage_pass_and_failures():
    w = RaggedWorkload(demand={(0, 2): 96, (1, 2): 32})
    ok = [
        _chunk(0, 0, 2, [0, 1, 2], 1.0, 64),
        _chunk(4, 0, 2, [0, 1, 2], 1.0, 32),
        _chunk(0, 1, 2, [1, 2], 1.0, 32),
    ]
    assert verify_workload_coverage(ok, w) == []

    short = ok[:2]  # (1,2) missing entirely
    problems = verify_workload_coverage(short, w)
    assert problems == ["pair (1, 2): scheduled 0 != demand 32"]

    extra = ok + [_chunk(9, 2, 0, [2, 1, 0], 1.0, 32)]
    problems = verify_workload_coverage(extra, w)
    assert problems == ["pair (2, 0): scheduled 32 but not in workload"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ragged_verify.py -v`
Expected: FAIL with `ImportError: cannot import name 'verify_capacity_ragged'`

- [ ] **Step 3: Write the implementation**

Append to `twisted_analysis/schedules/verify.py` (do not modify existing code; add `Mapping` usage via the existing `typing` import line — extend it to `from typing import Iterable, Mapping` if not already):

```python
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
        for time, kind, rate, key in evs:
            if kind == 1:
                acc += rate
                active.add(key)
                if acc > 1 + tol:
                    violations.append(RateViolation(
                        edge=edge, time=time, total_rate=acc,
                        flows=tuple(sorted(active)),
                    ))
            else:
                acc -= rate
                active.discard(key)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ragged_verify.py tests/test_schedule_verify.py -v`
Expected: all pass (including the pre-existing `test_schedule_verify.py` — regression check).

- [ ] **Step 5: Commit**

```bash
git add twisted_analysis/schedules/verify.py tests/test_ragged_verify.py
git commit -m "Add ragged sweepline verifier, makespan, coverage check

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: `ragged_fluid` scheduler

**Files:**
- Create: `twisted_analysis/schedules/ragged_fluid.py`
- Test: `tests/test_ragged_fluid.py`

**Interfaces:**
- Consumes: `RaggedWorkload` (Task 1), verifier trio (Task 4).
- Produces: `ragged_fluid(table: list[list[list[int]]], workload: RaggedWorkload) -> list[dict]` — one entry per flow, `{round: 0, src, dst, path, rate: size/LB, size}`, sorted by `(round, src, dst)`. Task 7's adapter consumes this exact signature.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ragged_fluid.py`:

```python
"""Closed-form water-filling schedule: feasibility and LB-tightness."""
from pathlib import Path

from twisted_analysis.io.routing_table import load_routing_table
from twisted_analysis.io.workload import load_workload
from twisted_analysis.model.ragged import RaggedWorkload
from twisted_analysis.schedules.ragged_fluid import ragged_fluid
from twisted_analysis.schedules.verify import (
    schedule_makespan_ragged,
    verify_capacity_ragged,
    verify_workload_coverage,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

LINE_TABLE = [
    [[0], [0, 1], [0, 1, 2]],
    [[1, 0], [1], [1, 2]],
    [[2, 1, 0], [2, 1], [2]],
]


def test_small_case_rates_and_makespan():
    w = RaggedWorkload(demand={(0, 2): 64, (1, 2): 32})
    sched = ragged_fluid(LINE_TABLE, w)
    assert len(sched) == 2
    by_pair = {(e["src"], e["dst"]): e for e in sched}
    assert by_pair[(0, 2)]["rate"] == 64 / 96
    assert by_pair[(1, 2)]["rate"] == 32 / 96
    assert all(e["round"] == 0 for e in sched)
    assert verify_capacity_ragged(sched, quantum=32) == []
    assert verify_workload_coverage(sched, w) == []
    # Every flow streams for LB = 3 quanta; the 2-hop flow adds 1 fill quantum.
    assert abs(schedule_makespan_ragged(sched, quantum=32) - 4.0) < 1e-9


def test_fixture_scale_lb_certificate():
    """Spec test 3: fluid schedule is feasible at makespan 399 = LB + 5."""
    w = load_workload(
        FIXTURES / "ragged_a2a_workload_node_128_min_32_max_1024_discrete.json"
    )
    table = load_routing_table(FIXTURES / "routing_table_8x4x4_twist.json")
    sched = ragged_fluid(table, w)
    assert len(sched) == len(w.demand)  # one entry per flow
    assert all(0 < e["rate"] <= 1 for e in sched)
    assert verify_capacity_ragged(sched, quantum=32) == []
    assert verify_workload_coverage(sched, w) == []
    ms = schedule_makespan_ragged(sched, quantum=32)
    assert abs(ms - 399.0) < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ragged_fluid.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'twisted_analysis.schedules.ragged_fluid'`

- [ ] **Step 3: Write the implementation**

Create `twisted_analysis/schedules/ragged_fluid.py`:

```python
"""Closed-form water-filling schedule for ragged workloads.

Static rates rate_f = size_f / LB (LB = max size-weighted edge load) give
every flow one chunk active over the whole horizon: every edge carries
exactly load(e)/LB <= 1, and every flow finishes at exactly LB quanta plus
its (L-1)-quantum pipeline fill. This is makespan-optimal (no schedule can
move load(bottleneck) across a unit-capacity edge faster than LB) and
entry-count-optimal (one entry per flow) in the fluid model — the LP
relaxation of the integral one-flow-per-edge-per-step problem, which fixed
paths make solvable in closed form. See
docs/superpowers/specs/2026-07-14-ragged-a2a-scheduling-design.md.

rate_f <= 1 always holds: LB >= load on the flow's own first edge >= size_f.
"""
from __future__ import annotations

from twisted_analysis.model.ragged import RaggedWorkload


def ragged_fluid(
    table: list[list[list[int]]],
    workload: RaggedWorkload,
) -> list[dict]:
    lb = workload.lower_bound(table)
    entries: list[dict] = []
    for (s, d), size in workload.demand.items():
        entries.append({
            "round": 0,
            "src": s,
            "dst": d,
            "path": list(table[s][d]),
            "rate": size / lb,
            "size": size,
        })
    entries.sort(key=lambda e: (e["round"], e["src"], e["dst"]))
    return entries
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ragged_fluid.py -v`
Expected: 2 passed. The fixture-scale test sweeplines ~65k chunk-hop intervals — should finish well under a minute.

- [ ] **Step 5: Commit**

```bash
git add twisted_analysis/schedules/ragged_fluid.py tests/test_ragged_fluid.py
git commit -m "Add ragged_fluid closed-form scheduler (LB certificate)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: `ragged_greedy` scheduler (non-preemptive + preemptive)

**Files:**
- Create: `twisted_analysis/schedules/ragged_greedy.py`
- Test: `tests/test_ragged_greedy.py`

**Interfaces:**
- Consumes: `RaggedWorkload` (Task 1), verifier trio (Task 4).
- Produces: `ragged_greedy(table, workload, *, order: str = "lpt", preemptive: bool = False) -> list[dict]`; orders `{"lpt", "spt", "natural"}`; entries carry `rate: 1.0` and `size` (workload units); non-preemptive emits exactly one entry per flow. Task 7's adapter consumes this exact signature.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ragged_greedy.py`:

```python
"""Integral earliest-feasible greedy: both variants, orders, fixture subsample."""
import json
from itertools import islice
from pathlib import Path

import pytest

from twisted_analysis.io.routing_table import load_routing_table
from twisted_analysis.io.workload import load_workload
from twisted_analysis.model.ragged import RaggedWorkload
from twisted_analysis.schedules.ragged_greedy import ragged_greedy
from twisted_analysis.schedules.verify import (
    schedule_makespan_ragged,
    verify_capacity_ragged,
    verify_workload_coverage,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

LINE_TABLE = [
    [[0], [0, 1], [0, 1, 2]],
    [[1, 0], [1], [1, 2]],
    [[2, 1, 0], [2, 1], [2]],
]

# 4-node line 0-1-2-3 (only the cells the tests touch need to be right).
LINE4_TABLE = [
    [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3]],
    [[1, 0], [1], [1, 2], [1, 2, 3]],
    [[2, 1, 0], [2, 1], [2], [2, 3]],
    [[3, 2, 1, 0], [3, 2, 1], [3, 2], [3]],
]


def test_rejects_unknown_order():
    w = RaggedWorkload(demand={(0, 1): 32})
    with pytest.raises(ValueError, match="order"):
        ragged_greedy(LINE_TABLE, w, order="bogus")


def test_nonpreemptive_disjoint_flows_both_start_at_zero():
    """(0,2) occupies (0,1)@{0,1}, (1,2)@{1,2}; (1,2)'s single quantum fits
    at t=0 on edge (1,2) — hop-offset pipelining interleaves them."""
    w = RaggedWorkload(demand={(0, 2): 64, (1, 2): 32})
    sched = ragged_greedy(LINE_TABLE, w, order="lpt")
    by_pair = {(e["src"], e["dst"]): e for e in sched}
    assert by_pair[(0, 2)]["round"] == 0
    assert by_pair[(1, 2)]["round"] == 0
    assert all(e["rate"] == 1.0 for e in sched)
    assert verify_capacity_ragged(sched, quantum=32) == []
    assert verify_workload_coverage(sched, w) == []
    assert schedule_makespan_ragged(sched, quantum=32) == 3.0


def test_nonpreemptive_conflicting_flow_waits():
    """(0,2) size 64 takes (0,1)@{0,1}; (0,1) size 32 must wait until t=2."""
    w = RaggedWorkload(demand={(0, 2): 64, (0, 1): 32})
    sched = ragged_greedy(LINE_TABLE, w, order="lpt")
    by_pair = {(e["src"], e["dst"]): e for e in sched}
    assert by_pair[(0, 2)]["round"] == 0
    assert by_pair[(0, 1)]["round"] == 2
    assert verify_capacity_ragged(sched, quantum=32) == []


def test_preemptive_splits_around_busy_slot():
    """Natural order on LINE4: (0,3) size 32 first marks (1,2)@1. Then (1,2)
    size 96 (3 quanta) can use t=0 but not t=1 -> chunks [0] and [2,3]."""
    w = RaggedWorkload(demand={(0, 3): 32, (1, 2): 96})
    sched = ragged_greedy(LINE4_TABLE, w, order="natural", preemptive=True)
    chunks_12 = sorted(
        (e["round"], e["size"]) for e in sched
        if (e["src"], e["dst"]) == (1, 2)
    )
    assert chunks_12 == [(0, 32), (2, 64)]
    assert verify_capacity_ragged(sched, quantum=32) == []
    assert verify_workload_coverage(sched, w) == []
    # Non-preemptive on the same workload must wait for 3 contiguous slots.
    ns = ragged_greedy(LINE4_TABLE, w, order="natural", preemptive=False)
    by_pair = {(e["src"], e["dst"]): e for e in ns}
    assert by_pair[(1, 2)]["round"] == 2
    assert schedule_makespan_ragged(sched, quantum=32) <= \
        schedule_makespan_ragged(ns, quantum=32)


def _subsample_workload(n_flows):
    raw = json.loads((
        FIXTURES / "ragged_a2a_workload_node_128_min_32_max_1024_discrete.json"
    ).read_text())
    return RaggedWorkload(demand={
        (e["src"], e["dst"]): e["size"] for e in islice(raw, n_flows)
    })


@pytest.mark.parametrize("preemptive", [False, True])
def test_fixture_subsample_feasible(preemptive):
    """Spec test 4 at reduced scale (first 300 flows) to keep runtime low."""
    w = _subsample_workload(300)
    table = load_routing_table(FIXTURES / "routing_table_8x4x4_twist.json")
    sched = ragged_greedy(table, w, order="lpt", preemptive=preemptive)
    assert verify_capacity_ragged(sched, quantum=w.quantum) == []
    assert verify_workload_coverage(sched, w) == []
    if not preemptive:
        assert len(sched) == 300  # exactly one entry per flow
    else:
        assert len(sched) >= 300


def test_fixture_subsample_preemptive_no_worse():
    w = _subsample_workload(300)
    table = load_routing_table(FIXTURES / "routing_table_8x4x4_twist.json")
    ms_non = schedule_makespan_ragged(
        ragged_greedy(table, w, order="lpt", preemptive=False),
        quantum=w.quantum,
    )
    ms_pre = schedule_makespan_ragged(
        ragged_greedy(table, w, order="lpt", preemptive=True),
        quantum=w.quantum,
    )
    assert ms_pre <= ms_non
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ragged_greedy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'twisted_analysis.schedules.ragged_greedy'`

- [ ] **Step 3: Write the implementation**

Create `twisted_analysis/schedules/ragged_greedy.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ragged_greedy.py -v`
Expected: 8 passed (subsample tests take seconds; if any exceeds ~1 min, reduce the subsample to 200 flows in both subsample tests — do NOT run full-fixture greedy in tests; that belongs to the eval script).

- [ ] **Step 5: Commit**

```bash
git add twisted_analysis/schedules/ragged_greedy.py tests/test_ragged_greedy.py
git commit -m "Add ragged_greedy scheduler (non-preemptive + preemptive)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Adapters + dispatch in `io/schedule.py`

**Files:**
- Modify: `twisted_analysis/io/schedule.py` (add two adapters before `_SCHEDULER_DISPATCH`; add dispatch entries; extend `schedule_from_algorithm` docstring)
- Test: `tests/test_schedule_from_algorithm.py` (append)

**Interfaces:**
- Consumes: `ragged_fluid` (Task 5), `ragged_greedy` (Task 6), `RaggedWorkload` (Task 1), existing `validate_routing_table_shape` and `_SCHEDULER_DISPATCH`.
- Produces (Task 8's CLI consumes via `schedule_from_algorithm`):
  - `schedule_from_ragged_fluid(topology, table, *, workload) -> list[dict]`
  - `schedule_from_ragged_greedy(topology, table, *, workload, order="lpt", preemptive=False) -> list[dict]`
  - dispatch keys `"ragged_fluid"` and `"ragged_greedy"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_schedule_from_algorithm.py`:

```python
def test_dispatch_ragged_fluid_and_greedy():
    from pathlib import Path

    from twisted_analysis.io.routing_table import load_routing_table
    from twisted_analysis.io.schedule import schedule_from_algorithm
    from twisted_analysis.model.ragged import RaggedWorkload
    from twisted_analysis.topology import Topology

    fixtures = Path(__file__).resolve().parent.parent / "fixtures"
    topology = Topology(slice=(8, 4, 4))
    table = load_routing_table(fixtures / "routing_table_8x4x4_twist.json")
    w = RaggedWorkload(demand={(0, 5): 64, (3, 9): 32, (100, 2): 96})

    fluid = schedule_from_algorithm(
        "ragged_fluid", topology, table, workload=w,
    )
    assert len(fluid) == 3
    assert all("rate" in e and "size" in e for e in fluid)

    greedy = schedule_from_algorithm(
        "ragged_greedy", topology, table,
        workload=w, order="lpt", preemptive=False,
    )
    assert len(greedy) == 3
    assert all(e["rate"] == 1.0 for e in greedy)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_schedule_from_algorithm.py -v -k ragged`
Expected: FAIL with `ValueError: unknown algorithm: 'ragged_fluid'`

- [ ] **Step 3: Add adapters and dispatch entries**

In `twisted_analysis/io/schedule.py`, insert before the `_SCHEDULER_DISPATCH` dict:

```python
def schedule_from_ragged_fluid(
    topology: Topology,
    table: list[list[list[int]]],
    *,
    workload,
) -> list[dict]:
    """Adapter: ragged_fluid -> schedule entries.

    `workload` is a twisted_analysis.model.ragged.RaggedWorkload (load one
    via twisted_analysis.io.workload.load_workload).
    """
    from twisted_analysis.io.routing_table import validate_routing_table_shape
    from twisted_analysis.schedules.ragged_fluid import ragged_fluid

    validate_routing_table_shape(table, topology.n_nodes)
    return ragged_fluid(table, workload)


def schedule_from_ragged_greedy(
    topology: Topology,
    table: list[list[list[int]]],
    *,
    workload,
    order: str = "lpt",
    preemptive: bool = False,
) -> list[dict]:
    """Adapter: ragged_greedy -> schedule entries.

    `workload` as in schedule_from_ragged_fluid. `order` in
    {"lpt", "spt", "natural"}; `preemptive` allows chunk splitting.
    """
    from twisted_analysis.io.routing_table import validate_routing_table_shape
    from twisted_analysis.schedules.ragged_greedy import ragged_greedy

    validate_routing_table_shape(table, topology.n_nodes)
    return ragged_greedy(table, workload, order=order, preemptive=preemptive)
```

Add to `_SCHEDULER_DISPATCH`:

```python
    "ragged_fluid": schedule_from_ragged_fluid,
    "ragged_greedy": schedule_from_ragged_greedy,
```

Append to the `schedule_from_algorithm` docstring's algorithm list:

```
      - "ragged_fluid":      closed-form water-filling for ragged workloads.
        Makespan-optimal in the fluid (continuous-rate) model; one entry per
        flow at rate size/LB. Requires `workload` kwarg (RaggedWorkload).
      - "ragged_greedy":     integral (rate=1) earliest-feasible greedy for
        ragged workloads. Requires `workload` kwarg; optional `order`
        ("lpt"/"spt"/"natural", default "lpt") and `preemptive` (default
        False; True splits flows into chunks for lower makespan).
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_schedule_from_algorithm.py -v`
Expected: all pass (new + pre-existing).

- [ ] **Step 5: Commit**

```bash
git add twisted_analysis/io/schedule.py tests/test_schedule_from_algorithm.py
git commit -m "Wire ragged schedulers into adapter dispatch

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: CLI `scripts/generate_ragged_schedule.py`

**Files:**
- Create: `scripts/generate_ragged_schedule.py`
- Test: `tests/test_generate_ragged_schedule_cli.py`

**Interfaces:**
- Consumes: `load_workload` (Task 2), `load_routing_table`, `schedule_from_algorithm` (Task 7), `save_schedule`, verifier trio (Task 4), `Topology(slice=...)`.
- Produces: CLI writing `fixtures/schedule_<slice>_loaded_ragged_<fluid|greedy_<order>[_pre]>.json` (or `--out`), printing metrics to stdout, optionally appending a CSV row via `--csv-append` with header
  `scheduler,order,preemptive,lb_quanta,makespan_quanta,gap_pct,entries,max_chunks_per_flow`. Task 9's eval script consumes this CLI.

- [ ] **Step 1: Write the failing test**

Create `tests/test_generate_ragged_schedule_cli.py`:

```python
"""CLI smoke test on the real fixture (fluid only — greedy is covered by
unit tests and is too slow at full fixture scale for the test suite)."""
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def test_cli_fluid_end_to_end(tmp_path, capsys):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import generate_ragged_schedule

    out = tmp_path / "sched.json"
    csv = tmp_path / "metrics.csv"
    rc = generate_ragged_schedule.main([
        "--routing-table", str(FIXTURES / "routing_table_8x4x4_twist.json"),
        "--slice", "8,4,4",
        "--workload", str(
            FIXTURES / "ragged_a2a_workload_node_128_min_32_max_1024_discrete.json"
        ),
        "--scheduler", "ragged_fluid",
        "--out", str(out),
        "--csv-append", str(csv),
    ])
    assert rc == 0
    assert out.exists()

    captured = capsys.readouterr().out
    assert "lb_quanta=394" in captured
    assert "makespan_quanta=399.00" in captured
    assert "entries=16256" in captured

    lines = csv.read_text().strip().splitlines()
    assert lines[0].startswith("scheduler,order,preemptive,lb_quanta")
    assert lines[1].startswith("ragged_fluid,,False,394,399.00")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_generate_ragged_schedule_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'generate_ragged_schedule'`

- [ ] **Step 3: Write the CLI**

Create `scripts/generate_ragged_schedule.py`:

```python
"""Generate a ragged-workload schedule JSON from a routing table + workload.

Usage:
    python scripts/generate_ragged_schedule.py \\
        --routing-table fixtures/routing_table_8x4x4_twist.json \\
        --slice 8,4,4 \\
        --workload fixtures/ragged_a2a_workload_node_128_min_32_max_1024_discrete.json \\
        --scheduler ragged_greedy --order lpt [--preemptive]

Verifies capacity and workload coverage before writing; exits non-zero on
any violation. Optionally appends a metrics row via --csv-append (header:
scheduler,order,preemptive,lb_quanta,makespan_quanta,gap_pct,entries,max_chunks_per_flow).
"""
from __future__ import annotations
import argparse
import sys
from collections import Counter
from pathlib import Path

# Make `python scripts/generate_ragged_schedule.py` work without install.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from twisted_analysis.io.routing_table import load_routing_table
from twisted_analysis.io.schedule import save_schedule, schedule_from_algorithm
from twisted_analysis.io.workload import load_workload
from twisted_analysis.schedules.verify import (
    schedule_makespan_ragged,
    verify_capacity_ragged,
    verify_workload_coverage,
)
from twisted_analysis.topology import Topology

CSV_HEADER = (
    "scheduler,order,preemptive,lb_quanta,makespan_quanta,"
    "gap_pct,entries,max_chunks_per_flow"
)


def _parse_slice(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Generate a ragged-A2A schedule JSON.",
    )
    p.add_argument("--routing-table", required=True, type=Path)
    p.add_argument("--slice", required=True,
                   help="Comma-separated slice, e.g. 8,4,4")
    p.add_argument("--workload", required=True, type=Path,
                   help="Ragged workload JSON ([{src, dst, size}, ...])")
    p.add_argument("--scheduler", required=True,
                   choices=["ragged_fluid", "ragged_greedy"])
    p.add_argument("--order", default="lpt",
                   choices=["lpt", "spt", "natural"],
                   help="Flow order for ragged_greedy (ignored by ragged_fluid)")
    p.add_argument("--preemptive", action="store_true",
                   help="ragged_greedy only: allow chunk splitting")
    p.add_argument("--out", default=None,
                   help="Output path (default: fixtures/schedule_<slice>_loaded_"
                        "ragged_<fluid|greedy_<order>[_pre]>.json)")
    p.add_argument("--csv-append", default=None, type=Path,
                   help="Append a metrics row to this CSV (header written if new)")
    args = p.parse_args(argv)

    slice_ = _parse_slice(args.slice)
    topology = Topology(slice=slice_)
    table = load_routing_table(args.routing_table)
    if len(table) != topology.n_nodes:
        raise SystemExit(
            f"routing table has {len(table)} sources; "
            f"slice {slice_} expects {topology.n_nodes}"
        )
    workload = load_workload(args.workload)
    quantum = workload.quantum
    lb_quanta = workload.lower_bound(table) // quantum

    kwargs = {"workload": workload}
    if args.scheduler == "ragged_greedy":
        kwargs.update(order=args.order, preemptive=args.preemptive)
    entries = schedule_from_algorithm(args.scheduler, topology, table, **kwargs)

    violations = verify_capacity_ragged(entries, quantum=quantum)
    if violations:
        raise SystemExit(
            f"capacity check FAILED: {len(violations)} violations; "
            f"first: {violations[0]}"
        )
    problems = verify_workload_coverage(entries, workload)
    if problems:
        raise SystemExit(
            f"coverage check FAILED: {len(problems)} problems; "
            f"first: {problems[0]}"
        )

    makespan = schedule_makespan_ragged(entries, quantum=quantum)
    gap_pct = 100.0 * (makespan - lb_quanta) / lb_quanta
    chunks_per_flow = Counter((e["src"], e["dst"]) for e in entries)
    max_chunks = max(chunks_per_flow.values())

    if args.out is None:
        slice_str = "x".join(str(s) for s in slice_)
        suffix = ("fluid" if args.scheduler == "ragged_fluid"
                  else f"greedy_{args.order}" + ("_pre" if args.preemptive else ""))
        out_path = _HERE.parent / "fixtures" / (
            f"schedule_{slice_str}_loaded_ragged_{suffix}.json"
        )
    else:
        out_path = Path(args.out)
    save_schedule(entries, out_path)

    order_str = args.order if args.scheduler == "ragged_greedy" else ""
    print(
        f"{args.scheduler} order={order_str or '-'} preemptive={args.preemptive} "
        f"lb_quanta={lb_quanta} makespan_quanta={makespan:.2f} "
        f"gap_pct={gap_pct:.2f} entries={len(entries)} "
        f"max_chunks_per_flow={max_chunks}"
    )
    print(f"wrote {out_path}", file=sys.stderr)

    if args.csv_append is not None:
        is_new = not args.csv_append.exists()
        with args.csv_append.open("a") as f:
            if is_new:
                f.write(CSV_HEADER + "\n")
            f.write(
                f"{args.scheduler},{order_str},{args.preemptive},"
                f"{lb_quanta},{makespan:.2f},{gap_pct:.2f},"
                f"{len(entries)},{max_chunks}\n"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_generate_ragged_schedule_cli.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_ragged_schedule.py tests/test_generate_ragged_schedule_cli.py
git commit -m "Add generate_ragged_schedule CLI with verify + CSV metrics

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: Eval script, fixture schedules, README

**Files:**
- Create: `eval/run_ragged_a2a.sh`
- Modify: `README.md` (schedulers table + fixtures list)
- Generated artifacts: `fixtures/schedule_8x4x4_loaded_ragged_fluid.json`, `..._ragged_greedy_{lpt,spt,natural}.json`, `..._ragged_greedy_lpt_pre.json`, `results/<date>/ragged_a2a.csv`

**Interfaces:**
- Consumes: the Task 8 CLI (including `--csv-append`).
- Produces: reproducible evaluation + committed schedule fixtures. No downstream consumers in this plan.

- [ ] **Step 1: Write the eval script**

Create `eval/run_ragged_a2a.sh`:

```bash
#!/usr/bin/env bash
# Reproduce the ragged-A2A scheduling comparison on the loaded 8x4x4 routing.
# Writes schedule fixtures + results/<date>/ragged_a2a.csv.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
RESULTS="results/$(date +%Y-%m-%d)"
mkdir -p "$RESULTS"
CSV="$RESULTS/ragged_a2a.csv"
rm -f "$CSV"

COMMON=(
  --routing-table fixtures/routing_table_8x4x4_twist.json
  --slice 8,4,4
  --workload fixtures/ragged_a2a_workload_node_128_min_32_max_1024_discrete.json
  --csv-append "$CSV"
)

"$PY" -u scripts/generate_ragged_schedule.py "${COMMON[@]}" --scheduler ragged_fluid

for ORDER in lpt spt natural; do
  "$PY" -u scripts/generate_ragged_schedule.py "${COMMON[@]}" \
    --scheduler ragged_greedy --order "$ORDER"
done

"$PY" -u scripts/generate_ragged_schedule.py "${COMMON[@]}" \
  --scheduler ragged_greedy --order lpt --preemptive

echo
echo "=== $CSV ==="
column -s, -t < "$CSV"
```

Then: `chmod +x eval/run_ragged_a2a.sh`

- [ ] **Step 2: Run it**

Run: `bash eval/run_ragged_a2a.sh`
Expected: 5 schedules written under `fixtures/`; CSV with 5 rows; fluid row shows `lb_quanta=394, makespan_quanta=399.00, entries=16256, max_chunks_per_flow=1`; greedy rows show makespan ≥ 394 (record the actual values — they are new results). Full-fixture greedy runs may take a few minutes each; that is expected. If a run exceeds ~15 minutes, stop and file the runtime as a finding rather than shipping a slow eval.

- [ ] **Step 3: Update README**

In `README.md`, add two rows to the scheduling-algorithms table:

```markdown
| `ragged_fluid` | Ragged workloads: closed-form water-filling (`rate = size/LB`), one entry per flow | Provably makespan- and entry-count-optimal in the continuous-rate (fluid) model; on the 128-node ragged fixture: LB = 394 quanta, makespan 399 (LB + pipeline fill) |
| `ragged_greedy` | Ragged workloads: integral (`rate = 1`) earliest-feasible greedy; non-preemptive (1 entry/flow) or preemptive (chunked) variants; orders `lpt`/`spt`/`natural` | No LB guarantee; fill in measured makespans from `eval/run_ragged_a2a.sh` |
```

Replace the "fill in measured makespans" placeholder with the actual CSV numbers from Step 2 before committing.

Also add one sentence to the `fixtures/` bullet in the Layout section: ragged workloads are `ragged_a2a_workload_<...>.json` and their schedules `schedule_<slice>_loaded_ragged_<scheduler>.json`; ragged entries carry `rate`/`size` fields (see the 2026-07-14 spec).

- [ ] **Step 4: Run the full test suite as a final regression gate**

Run: `.venv/bin/python -m pytest`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
# results/ is gitignored (per repo convention); commit fixtures + script only.
git add eval/run_ragged_a2a.sh README.md fixtures/schedule_8x4x4_loaded_ragged_*.json
git commit -m "Add ragged A2A eval script, fixture schedules, README rows

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```
