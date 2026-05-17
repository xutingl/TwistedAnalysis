# Spread-Scheduling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a new "spread" scheduling algorithm — `spread_greedy(k)` — that limits per-device outgoing AND incoming DMAs per round to a tunable cap K, then produce a schedule JSON + Pallas kernel on the loaded 8×4×4 routing. Hypothesis (motivated by 2026-05-16 TPU measurement: makespan-78 schedule achieved 132764 gbps vs P2P's 134541 gbps despite +9% simulator gain): real TPU wall-clock is dominated by per-round DMA-engine oversubscription and ICI link bandwidth, not by round count. A schedule with more rounds but ≤K DMAs per device per round may translate to better measured throughput, even if its simulator makespan is worse.

**Architecture:** `spread_greedy` is a small extension of `literal_greedy`: same per-flow earliest-feasible greedy, but with two additional constraints checked at each candidate round t: `out_count[(src, t)] < k` and `in_count[(dst, t)] < k`. At K=1 the schedule is P2P-style (each device sends/receives at most 1 DMA per round; makespan ≥ N-1 = 127 on loaded 8×4×4). At K=∞ it degenerates to `literal_greedy` (makespan 87 on the same routing). Intermediate K trades makespan for per-device uniformity. We sweep K ∈ {1, 2, 3, 4} via a fast greedy probe (each run < 10 s), save all four schedules as fixtures, pick **K=2 as the headline** (lowest non-trivial K → maximum spreading effect while keeping makespan competitive enough to test on TPU), and emit one regular + one `--inline-destinations` Pallas kernel for the headline.

**Tech Stack:** Python 3, existing `twisted_analysis` package, pytest. No new dependencies. Greedy runs in seconds, no long-running compute.

---

## File Structure

**Create:**
- `twisted_analysis/schedules/spread_greedy.py` — new scheduler module.
- `tests/test_spread_greedy.py` — TDD tests.
- `eval/explorations/2026-05-17-spread-scheduling/README.md` — exploration problem + outcome.
- `eval/explorations/2026-05-17-spread-scheduling/RESULTS.md` — K-sweep comparison table.
- `eval/explorations/2026-05-17-spread-scheduling/01_spread_sweep.py` — runs K∈{1,2,3,4} and writes 4 schedule JSONs + comparison metrics.

**Modify:**
- `twisted_analysis/io/schedule.py` — add `schedule_from_spread_greedy` adapter; register `"spread_greedy"` in `_SCHEDULER_DISPATCH`; extend the `schedule_from_algorithm` docstring.
- `scripts/generate_schedule.py` — add `"spread_greedy"` to `--scheduler` choices and a `--k INT` flag.
- `fixtures/cns_schedules/readme.md` — add row for the K=2 headline.
- `pallas_kernel/README.md` — add `spread_greedy(k)` row to scheduler matrix; mention new kernel files; one new "when to use" row.
- `README.md` (root) — add `spread_greedy` row to scheduler matrix; mention new kernel files in Layout.

**Generated (not committed until they exist):**
- `eval/explorations/2026-05-17-spread-scheduling/01_spread_sweep_results.json` — per-K metrics.
- `eval/explorations/2026-05-17-spread-scheduling/schedule_k{1,2,3,4}.json` — four schedules saved by the probe.
- `fixtures/schedule_8x4x4_loaded_spread_greedy_k{1,2,3,4}.json` — all four promoted.
- `fixtures/cns_schedules/schedule_spreadgreedyk2_4x4x8_twisted.json` — headline only.
- `pallas_kernel/outputs/_ragged_a2a_kernel_spread_greedy_k2_8_4_4.py` — headline kernel (regular).
- `pallas_kernel/outputs/_ragged_a2a_kernel_spread_greedy_k2_inline_8_4_4.py` — headline kernel (inline-destinations variant).

---

### Task 1: Implement `spread_greedy` scheduler

**Files:**
- Create: `twisted_analysis/schedules/spread_greedy.py`
- Test: `tests/test_spread_greedy.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_spread_greedy.py`:

```python
"""Spread-greedy AllToAll scheduler tests."""
from __future__ import annotations
from collections import Counter
from pathlib import Path

import pytest

from twisted_analysis.io.routing_table import load_routing_table, save_routing_table
from twisted_analysis.schedules.spread_greedy import spread_greedy
from twisted_analysis.schedules.literal_greedy import literal_greedy
from twisted_analysis.schedules.verify import verify_capacity, schedule_makespan
from twisted_analysis.topology import Topology, ILPRouter

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _per_device_per_round_max(schedule, n):
    out_counts: Counter = Counter()
    in_counts: Counter = Counter()
    for e in schedule:
        out_counts[(e["src"], e["round"])] += 1
        in_counts[(e["dst"], e["round"])] += 1
    return max(out_counts.values()), max(in_counts.values())


def test_spread_greedy_zero_violations_loaded_8x4x4_k2():
    topology = Topology(slice=(8, 4, 4))
    table = load_routing_table(FIXTURES / "routing_table_8x4x4_twist.json")
    schedule = spread_greedy(topology, table, k=2, order="lpt")
    assert verify_capacity(schedule) == []
    pairs = {(e["src"], e["dst"]) for e in schedule}
    n = topology.n_nodes
    assert len(pairs) == n * (n - 1)


def test_spread_greedy_respects_k_cap_loaded_8x4x4():
    topology = Topology(slice=(8, 4, 4))
    table = load_routing_table(FIXTURES / "routing_table_8x4x4_twist.json")
    for k in (1, 2, 3, 4):
        schedule = spread_greedy(topology, table, k=k, order="lpt")
        out_max, in_max = _per_device_per_round_max(schedule, topology.n_nodes)
        assert out_max <= k, f"k={k}: max outgoing per device per round = {out_max} > k"
        assert in_max <= k, f"k={k}: max incoming per device per round = {in_max} > k"


def test_spread_greedy_k1_makespan_at_least_n_minus_one():
    """At K=1, each device sends 1 DMA per round, so makespan must be >= N-1."""
    topology = Topology(slice=(2, 4))
    router = ILPRouter(topology=topology)
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name
    try:
        save_routing_table(topology, router, tmp_path)
        table = load_routing_table(tmp_path)
    finally:
        os.unlink(tmp_path)
    schedule = spread_greedy(topology, table, k=1, order="lpt")
    assert verify_capacity(schedule) == []
    assert schedule_makespan(schedule) >= topology.n_nodes - 1


def test_spread_greedy_large_k_matches_literal_greedy_makespan():
    """At K >= N, spread_greedy's per-device cap is non-binding, so the
    schedule's makespan should equal literal_greedy's (they make the same
    per-flow earliest-feasible choices in the same order)."""
    topology = Topology(slice=(8, 4, 4))
    table = load_routing_table(FIXTURES / "routing_table_8x4x4_twist.json")
    spread = spread_greedy(topology, table, k=topology.n_nodes, order="lpt")
    lit = literal_greedy(topology, table, order="lpt")
    assert schedule_makespan(spread) == schedule_makespan(lit)


@pytest.mark.parametrize("order", ["lpt", "spt", "natural"])
def test_spread_greedy_orderings_all_feasible(order):
    topology = Topology(slice=(8, 4, 4))
    table = load_routing_table(FIXTURES / "routing_table_8x4x4_twist.json")
    schedule = spread_greedy(topology, table, k=2, order=order)
    assert verify_capacity(schedule) == []


def test_spread_greedy_invalid_k_raises():
    topology = Topology(slice=(2, 4))
    router = ILPRouter(topology=topology)
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name
    try:
        save_routing_table(topology, router, tmp_path)
        table = load_routing_table(tmp_path)
    finally:
        os.unlink(tmp_path)
    with pytest.raises(ValueError, match="k must be a positive integer"):
        spread_greedy(topology, table, k=0, order="lpt")
    with pytest.raises(ValueError, match="k must be a positive integer"):
        spread_greedy(topology, table, k=-1, order="lpt")
```

- [ ] **Step 2: Run tests to verify they fail (module doesn't exist yet)**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis && .venv/bin/python -m pytest tests/test_spread_greedy.py -v -x
```

Expected: collection / import error with `ModuleNotFoundError: No module named 'twisted_analysis.schedules.spread_greedy'`.

- [ ] **Step 3: Implement `spread_greedy`**

Create `twisted_analysis/schedules/spread_greedy.py`:

```python
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

Motivation: on the loaded 8x4x4 routing, the makespan-78 schedule from
`cpsat_literal` warm-started measured only 132764 gbps on TPU v5e -- nearly
identical to the orbit_greedy-85 kernel's 132758 gbps and ~1.3% below the
P2P reference's 134541 gbps. The simulator-projected +9 % gain did not
translate to wall-clock. The leading hypothesis is that per-device DMA-
engine concurrency and ICI link bandwidth dominate per-round wall-clock,
making round-count a poor proxy. `spread_greedy` is a direct test: produce
schedules with fewer simultaneous DMAs per device, accept a higher round
count, and measure on TPU.
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
```

- [ ] **Step 4: Run tests to verify all pass**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis && .venv/bin/python -m pytest tests/test_spread_greedy.py -v
```

Expected: all 7 tests PASS (1 loaded-8x4x4 feasibility + 1 k-cap respected + 1 k=1 makespan-bound + 1 large-k matches literal_greedy + 3 orderings + 1 invalid-k).

- [ ] **Step 5: Commit**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
git add twisted_analysis/schedules/spread_greedy.py tests/test_spread_greedy.py
git commit -m "schedules: add spread_greedy (per-device DMA-cap variant of literal_greedy)"
```

---

### Task 2: Register `spread_greedy` in `io/schedule.py` dispatch

**Files:**
- Modify: `twisted_analysis/io/schedule.py` (add `schedule_from_spread_greedy`, register in `_SCHEDULER_DISPATCH`, extend `schedule_from_algorithm` docstring)

- [ ] **Step 1: Add the adapter function**

In `twisted_analysis/io/schedule.py`, after the `schedule_from_literal_greedy` function (around line 153–165) and before the next `schedule_from_*` definition, add:

```python
def schedule_from_spread_greedy(
    topology: Topology,
    table: list[list[list[int]]],
    *,
    k: int,
    order: str = "lpt",
) -> list[dict]:
    """Adapter: spread_greedy -> schedule entries.

    `k` is the per-device-per-round outgoing AND incoming DMA cap. See
    `twisted_analysis.schedules.spread_greedy.spread_greedy` for tradeoffs.
    """
    from twisted_analysis.io.routing_table import validate_routing_table_shape
    from twisted_analysis.schedules.spread_greedy import spread_greedy

    validate_routing_table_shape(table, topology.n_nodes)
    return spread_greedy(topology, table, k=k, order=order)
```

- [ ] **Step 2: Register in `_SCHEDULER_DISPATCH`**

Find the `_SCHEDULER_DISPATCH = {...}` dict and add `"spread_greedy": schedule_from_spread_greedy,` (placement: after `"literal_greedy"` to keep the family together).

- [ ] **Step 3: Extend `schedule_from_algorithm` docstring**

In the `schedule_from_algorithm` function's docstring, add (after the `"literal_greedy"` bullet and before the next):

```
      - "spread_greedy":     `literal_greedy` plus a per-device-per-round DMA cap K.
        Requires `k` kwarg (positive int). K=1 -> P2P-style (each device sends/receives
        at most 1 DMA per round); K=infinity -> equivalent to `literal_greedy`.
        Optional `order` (default "lpt").
```

- [ ] **Step 4: Add a dispatch round-trip test**

Append to `tests/test_spread_greedy.py`:

```python
def test_spread_greedy_via_dispatch():
    """schedule_from_algorithm('spread_greedy', ...) must dispatch correctly."""
    from twisted_analysis.io.schedule import schedule_from_algorithm
    topology = Topology(slice=(8, 4, 4))
    table = load_routing_table(FIXTURES / "routing_table_8x4x4_twist.json")
    schedule = schedule_from_algorithm("spread_greedy", topology, table, k=2)
    assert verify_capacity(schedule) == []
    n = topology.n_nodes
    assert len({(e["src"], e["dst"]) for e in schedule}) == n * (n - 1)
```

- [ ] **Step 5: Run tests**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis && .venv/bin/python -m pytest tests/test_spread_greedy.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
git add twisted_analysis/io/schedule.py tests/test_spread_greedy.py
git commit -m "io/schedule: register spread_greedy in dispatch"
```

---

### Task 3: Add CLI support to `scripts/generate_schedule.py`

**Files:**
- Modify: `scripts/generate_schedule.py`

The current CLI's `--scheduler` argument has very few choices (just `orbit_greedy` per the help-text dump). Add `spread_greedy` and a `--k` flag.

- [ ] **Step 1: Inspect the CLI**

Read `scripts/generate_schedule.py` to find the argparse block. Locate the line that adds `--scheduler` and the dispatch call into `schedule_from_algorithm`.

- [ ] **Step 2: Add `spread_greedy` to `--scheduler` choices**

In the argparse setup, the `--scheduler` argument currently restricts choices (e.g. `choices=["orbit_greedy"]`). Add `"spread_greedy"` (and consider widening to the full dispatch keys; if so, prefer the safer minimal change: just add `spread_greedy`).

- [ ] **Step 3: Add `--k` flag**

In the argparse setup, add:

```python
parser.add_argument(
    "--k",
    type=int,
    default=None,
    help="Per-device-per-round DMA cap for spread_greedy (positive int). "
         "Required when --scheduler spread_greedy.",
)
```

- [ ] **Step 4: Wire the kwarg through**

Where `schedule_from_algorithm` is called with kwargs derived from CLI args, add a branch:

```python
if args.scheduler == "spread_greedy":
    if args.k is None:
        parser.error("--scheduler spread_greedy requires --k INT")
    extra_kwargs = {"k": args.k, "order": args.order}
else:
    extra_kwargs = {"order": args.order}
schedule = schedule_from_algorithm(args.scheduler, topology, table, **extra_kwargs)
```

(Adapt to the script's actual variable names — `extra_kwargs` may already exist under a different name.)

- [ ] **Step 5: Smoke-test the CLI on (2,4)**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis && PATH=".venv/bin:$PATH" .venv/bin/python scripts/generate_schedule.py \
    --slice 2,4 \
    --router ilp \
    --scheduler spread_greedy \
    --k 1 \
    --order lpt
```

Expected: writes a schedule JSON without errors. The output makespan should be >= 7 (N-1 for N=8, since k=1 means device LB binds).

Verify with:

```bash
.venv/bin/python -c "
from twisted_analysis.io.schedule import load_schedule
from twisted_analysis.schedules.verify import verify_capacity, schedule_makespan
import json
import glob
# Find the most recent schedule file matching this run:
files = sorted(glob.glob('fixtures/schedule_2x4_*spread_greedy*lpt*.json'))
sched = load_schedule(files[-1])
print(f'file={files[-1]}, makespan={schedule_makespan(sched)}, viol={len(verify_capacity(sched))}')
"
```

Expected: `makespan >= 7, viol = 0`.

- [ ] **Step 6: Clean up the smoke-test schedule file and commit**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
ls fixtures/schedule_2x4_*spread_greedy*lpt*.json | head -1 | xargs -r rm
git add scripts/generate_schedule.py
git commit -m "scripts/generate_schedule: add spread_greedy scheduler + --k flag"
```

---

### Task 4: Create exploration folder + README + RESULTS scaffold

**Files:**
- Create: `eval/explorations/2026-05-17-spread-scheduling/README.md`
- Create: `eval/explorations/2026-05-17-spread-scheduling/RESULTS.md`

- [ ] **Step 1: Create the folder**

```bash
mkdir -p /home/xutingl/collective_comm/TwistedAnalysis/eval/explorations/2026-05-17-spread-scheduling
```

- [ ] **Step 2: Write README.md**

Create `eval/explorations/2026-05-17-spread-scheduling/README.md` with this content:

```markdown
# Spread-scheduling: per-device DMA-cap variant of literal_greedy

## Problem

The 2026-05-16 exploration produced a makespan-78 schedule (warm-started
CP-SAT) on the loaded 8×4×4 routing — simulator-projected to ~144607 gbps
(+7.5% above P2P's measured 134541 gbps). When the corresponding Pallas
kernel was run on TPU v5e, the actual measured throughput was **132764
gbps** — essentially identical to the orbit_greedy-85 kernel's 132758 gbps,
and ~1.3% **below** the P2P reference. The +9% simulator gain translated
to ~0% wall-clock change.

The leading hypothesis: the simulator's "1 round = 1 unit of wall-clock"
model is wrong on TPU. Real wall-clock per round is dominated by per-device
DMA-engine setup, ICI link bandwidth, HBM bandwidth, and semaphore wait
latency — none of which the simulator models. A schedule that issues many
DMAs from the same device in the same round oversubscribes the DMA engine;
the apparent "shorter makespan" is offset by per-round serialization that
the simulator doesn't see.

## Goal

Implement and ship a scheduling algorithm — `spread_greedy(k)` — that
explicitly limits per-device outgoing AND incoming DMAs per round to a
tunable cap K. Generate schedules at K ∈ {1, 2, 3, 4} on the loaded 8×4×4
routing. Pick K=2 as the headline (lowest non-trivial spread → maximum
"P2P-like" load distribution while keeping makespan competitive for
on-TPU testing). Ship the headline schedule as a fixture + Pallas kernel.

## Approach (one probe)

1. [01_spread_sweep.py](01_spread_sweep.py) — Run `spread_greedy(k, order="lpt")`
   for K ∈ {1, 2, 3, 4} on the loaded 8×4×4 routing. For each K, compute:
   - Makespan
   - Physical-edge capacity violations (must be 0 for all)
   - Max outgoing DMAs per device per round (must be ≤ K)
   - Max incoming DMAs per device per round (must be ≤ K)
   - Average DMAs per device per round (a measure of pipeline density)
   - Number of rounds with at least one DMA (a measure of "spread")

   Save all four schedule JSONs and a `01_spread_sweep_results.json`
   comparison table.

## Headline choice

K=2 is the headline. Rationale: it is the smallest K > 1, so it preserves
the most of the per-device uniformity that makes P2P competitive on TPU,
while permitting two-way pipelining per device per round (vs P2P's strict
one-way). If TPU measurement on this kernel disagrees with the simulator's
makespan ranking — i.e. K=2 outperforms cpsat_literal_warm (makespan 78)
despite a higher makespan — that is direct evidence that per-round
wall-clock, not round count, is the binding constraint. If K=2 is also
beaten on TPU by K=1 or K=4, the other K values are already saved as
fixtures and can be tested without re-running the probe.

The four K values are all promoted to `fixtures/` so a TPU operator can
deploy any of them; only K=2 gets the cns_schedules entry, the recommended
`fixtures/cns_schedules/readme.md` row, and the pre-generated Pallas kernel.

## Compute budget

Minutes. Greedy is fast (each K runs in seconds on N=128). No long
background runs.

## Outcome

(Filled in after the probe completes.)
```

- [ ] **Step 3: Write RESULTS.md (scaffold)**

Create `eval/explorations/2026-05-17-spread-scheduling/RESULTS.md` with:

```markdown
# Results: Spread-scheduling K-sweep on loaded 8×4×4

**Baseline (incoming):**
- `cpsat_literal` warm-started, makespan 78 (the production fixture at
  `fixtures/schedule_8x4x4_loaded_cpsat_literal_warm.json`).
  Measured on TPU v5e: 132764 gbps. Simulator-projected: ~144607 gbps.
  Simulator-to-reality gap: ~9% over-estimate.

**LB:** 75 (max physical-edge load).

## K-sweep (probe 1)

`spread_greedy(k, order="lpt")` on `fixtures/routing_table_8x4x4_twist.json`.

| K | makespan | viol | max DMAs/device-round | avg DMAs/device-round | n_rounds_with_dma | runtime |
|---:|---:|---:|:---:|---:|---:|---:|
| (pending) | | | | | | |

## Summary

(Filled in after probe runs.)
```

- [ ] **Step 4: Commit**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
git add eval/explorations/2026-05-17-spread-scheduling/
git commit -m "exploration: scaffold 2026-05-17-spread-scheduling (README + RESULTS)"
```

---

### Task 5: Write & execute the K-sweep probe

**Files:**
- Create: `eval/explorations/2026-05-17-spread-scheduling/01_spread_sweep.py`

- [ ] **Step 1: Write the probe script**

Create `eval/explorations/2026-05-17-spread-scheduling/01_spread_sweep.py`:

```python
"""Probe 1: K-sweep over spread_greedy(k) on loaded 8x4x4.

Runs K in {1, 2, 3, 4} with order='lpt'. For each K, computes:
  - makespan (max round + L over flows)
  - capacity violations (must be 0)
  - max outgoing DMAs per device per round (must be <= K)
  - max incoming DMAs per device per round (must be <= K)
  - average DMAs per device per active round (pipeline density signal)
  - number of distinct rounds containing at least one DMA (spread signal)

Saves four schedule JSONs (schedule_k1.json ... schedule_k4.json) and a
single 01_spread_sweep_results.json comparison table.
"""
from __future__ import annotations
import json
import time
from collections import Counter
from pathlib import Path

from twisted_analysis.topology import Topology
from twisted_analysis.io.routing_table import load_routing_table
from twisted_analysis.io.schedule import save_schedule
from twisted_analysis.schedules.spread_greedy import spread_greedy
from twisted_analysis.schedules.verify import verify_capacity, schedule_makespan

ROUTING = "fixtures/routing_table_8x4x4_twist.json"
SLICE = (8, 4, 4)
K_VALUES = [1, 2, 3, 4]
ORDER = "lpt"

OUT = Path(__file__).parent / "01_spread_sweep_results.json"


def _metrics(schedule, n):
    out_counts: Counter = Counter()
    in_counts: Counter = Counter()
    rounds_with_dma: set[int] = set()
    for e in schedule:
        out_counts[(e["src"], e["round"])] += 1
        in_counts[(e["dst"], e["round"])] += 1
        rounds_with_dma.add(e["round"])
    max_out = max(out_counts.values())
    max_in = max(in_counts.values())
    # Average DMAs per (device, active-round) pair across both sides.
    total_out_dmas = sum(out_counts.values())
    avg_out_per_device_round = total_out_dmas / len(out_counts)
    return {
        "max_out_per_device_round": max_out,
        "max_in_per_device_round": max_in,
        "avg_out_per_device_round": round(avg_out_per_device_round, 3),
        "n_rounds_with_dma": len(rounds_with_dma),
    }


def run():
    topology = Topology(slice=SLICE)
    table = load_routing_table(ROUTING)
    n = topology.n_nodes
    print(f"Loaded {ROUTING}, n={n}", flush=True)

    rows = []
    for k in K_VALUES:
        t0 = time.time()
        sch = spread_greedy(topology, table, k=k, order=ORDER)
        dt = time.time() - t0
        viol = verify_capacity(sch)
        mks = schedule_makespan(sch)
        m = _metrics(sch, n)
        row = {
            "k": k,
            "makespan": mks,
            "violations": len(viol),
            "runtime_s": round(dt, 2),
            **m,
        }
        rows.append(row)
        print(f"  k={k}: makespan={mks} viol={len(viol)} "
              f"max_out={m['max_out_per_device_round']} "
              f"max_in={m['max_in_per_device_round']} "
              f"avg={m['avg_out_per_device_round']} "
              f"rounds_used={m['n_rounds_with_dma']} "
              f"t={dt:.2f}s",
              flush=True)

        out_sched = Path(__file__).parent / f"schedule_k{k}.json"
        save_schedule(sch, out_sched)
        print(f"    saved {out_sched}", flush=True)

    result = {
        "routing": ROUTING, "slice": list(SLICE), "n": n,
        "order": ORDER, "k_values": K_VALUES,
        "rows": rows,
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Run the probe**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis && PATH=".venv/bin:$PATH" .venv/bin/python -u eval/explorations/2026-05-17-spread-scheduling/01_spread_sweep.py 2>&1 | tee eval/explorations/2026-05-17-spread-scheduling/01_spread_sweep_log.txt
```

Expected: < 60 s total. Each K produces one schedule file and one row of metrics. K=1 should have `max_out=1 max_in=1` and large makespan (>= 127); K=4 should have `max_out<=4 max_in<=4` and small makespan (likely 80-90).

If any K produces nonzero violations, STOP — that indicates a bug in `spread_greedy`. Re-investigate before continuing.

- [ ] **Step 3: Commit the probe script + raw results**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
git add eval/explorations/2026-05-17-spread-scheduling/01_spread_sweep.py \
        eval/explorations/2026-05-17-spread-scheduling/01_spread_sweep_log.txt \
        eval/explorations/2026-05-17-spread-scheduling/01_spread_sweep_results.json \
        eval/explorations/2026-05-17-spread-scheduling/schedule_k1.json \
        eval/explorations/2026-05-17-spread-scheduling/schedule_k2.json \
        eval/explorations/2026-05-17-spread-scheduling/schedule_k3.json \
        eval/explorations/2026-05-17-spread-scheduling/schedule_k4.json
git commit -m "exploration: K-sweep over spread_greedy(k) on loaded 8x4x4"
```

---

### Task 6: Promote all four schedules to `fixtures/`; promote K=2 to `cns_schedules`

**Files:**
- Create: `fixtures/schedule_8x4x4_loaded_spread_greedy_k1.json`
- Create: `fixtures/schedule_8x4x4_loaded_spread_greedy_k2.json`
- Create: `fixtures/schedule_8x4x4_loaded_spread_greedy_k3.json`
- Create: `fixtures/schedule_8x4x4_loaded_spread_greedy_k4.json`
- Create: `fixtures/cns_schedules/schedule_spreadgreedyk2_4x4x8_twisted.json`
- Modify: `fixtures/cns_schedules/readme.md`

- [ ] **Step 1: Copy all four schedules to `fixtures/`**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
for k in 1 2 3 4; do
  cp eval/explorations/2026-05-17-spread-scheduling/schedule_k${k}.json \
     fixtures/schedule_8x4x4_loaded_spread_greedy_k${k}.json
done
ls -la fixtures/schedule_8x4x4_loaded_spread_greedy_k*.json
```

Expected: four files listed, all the same size as the exploration's `schedule_k*.json`.

- [ ] **Step 2: Promote K=2 to `cns_schedules/`**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
cp fixtures/schedule_8x4x4_loaded_spread_greedy_k2.json \
   fixtures/cns_schedules/schedule_spreadgreedyk2_4x4x8_twisted.json
```

- [ ] **Step 3: Update `fixtures/cns_schedules/readme.md`**

Open `fixtures/cns_schedules/readme.md`. The existing table has rows for `cpsatliteralwarm` (top), `cpsatliteral`, `orbitfull`, `literalgreedy`, `orbit`. Insert a new row **above** the `cpsatliteralwarm` row (since spread_greedy K=2 is the new TPU-targeted recommendation):

```markdown
| `schedule_spreadgreedyk2_4x4x8_twisted.json` | `schedule_8x4x4_loaded_spread_greedy_k2.json` | `spread_greedy(k=2)` — per-device DMA-cap variant of `literal_greedy` | <FILL: makespan from probe> | 0 |
```

(Fill in the makespan value from `01_spread_sweep_results.json`'s K=2 row before committing.)

Then update the "Recommended for production measurement runs" paragraph immediately below the table. The current text recommends `cpsatliteralwarm`. Replace the entire paragraph with:

```
**Recommended for production measurement runs: `spreadgreedyk2` (and run a side-by-side TPU benchmark vs `cpsatliteralwarm`).** The makespan-78 `cpsatliteralwarm` schedule measured 132764 gbps on TPU v5e — essentially unchanged from `orbitfull` (132758 gbps) and ~1.3% below the P2P reference (134541 gbps), despite a simulator projection of +7.5%. The leading hypothesis is that per-device DMA-engine oversubscription dominates per-round wall-clock; `spread_greedy(k=2)` caps each device at 2 simultaneous outgoing AND incoming DMAs per round, trading higher simulator makespan for lower per-round contention. The other K values (`spread_greedy_k1` ≈ P2P-style, `spread_greedy_k3`, `spread_greedy_k4`) are shipped in `fixtures/` for TPU-side comparison but not promoted to `cns_schedules/`. Provenance: `eval/explorations/2026-05-17-spread-scheduling/`.

The previously-recommended `cpsatliteralwarm` (makespan 78, projected +7.5% vs P2P; measured ~0%) is retained as the makespan-optimal baseline.
```

- [ ] **Step 4: Commit**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
git add fixtures/schedule_8x4x4_loaded_spread_greedy_k*.json \
        fixtures/cns_schedules/schedule_spreadgreedyk2_4x4x8_twisted.json \
        fixtures/cns_schedules/readme.md
git commit -m "fixtures: ship spread_greedy K={1,2,3,4} schedules; cns recommends K=2"
```

---

### Task 7: Generate Pallas kernels for K=2 (regular + inline-destinations)

**Files:**
- Create: `pallas_kernel/outputs/_ragged_a2a_kernel_spread_greedy_k2_8_4_4.py`
- Create: `pallas_kernel/outputs/_ragged_a2a_kernel_spread_greedy_k2_inline_8_4_4.py`

- [ ] **Step 1: Generate the regular (SMEM dest_table_ref) kernel**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis && PATH=".venv/bin:$PATH" .venv/bin/python -u pallas_kernel/gen_orbit_greedy_kernel.py \
    --slice 8,4,4 \
    --routing-table fixtures/routing_table_8x4x4_twist.json \
    --schedule-in fixtures/schedule_8x4x4_loaded_spread_greedy_k2.json \
    --function-name _ragged_a2a_kernel_spread_greedy_k2_8_4_4 \
    --out pallas_kernel/outputs/_ragged_a2a_kernel_spread_greedy_k2_8_4_4.py
```

Expected output ends with `[3/3] wrote kernel ...` or `[4/4] wrote kernel ...` with `16256 flows, 0 violations`. The file should be ~90 KB.

- [ ] **Step 2: Generate the `--inline-destinations` variant**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis && PATH=".venv/bin:$PATH" .venv/bin/python -u pallas_kernel/gen_orbit_greedy_kernel.py \
    --slice 8,4,4 \
    --routing-table fixtures/routing_table_8x4x4_twist.json \
    --schedule-in fixtures/schedule_8x4x4_loaded_spread_greedy_k2.json \
    --inline-destinations \
    --function-name _ragged_a2a_kernel_spread_greedy_k2_inline_8_4_4 \
    --out pallas_kernel/outputs/_ragged_a2a_kernel_spread_greedy_k2_inline_8_4_4.py
```

Expected: similar success line; the `--inline-destinations` file will be larger (destinations baked as `jax.lax.switch` branches).

- [ ] **Step 3: Smoke-check both kernels parse as valid Python**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
.venv/bin/python -c "import ast; ast.parse(open('pallas_kernel/outputs/_ragged_a2a_kernel_spread_greedy_k2_8_4_4.py').read())" && echo "regular: ok"
.venv/bin/python -c "import ast; ast.parse(open('pallas_kernel/outputs/_ragged_a2a_kernel_spread_greedy_k2_inline_8_4_4.py').read())" && echo "inline: ok"
```

Expected: both print "ok".

- [ ] **Step 4: Commit**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
git add pallas_kernel/outputs/_ragged_a2a_kernel_spread_greedy_k2_8_4_4.py \
        pallas_kernel/outputs/_ragged_a2a_kernel_spread_greedy_k2_inline_8_4_4.py
git commit -m "kernel: spread_greedy(k=2) Pallas kernel (regular + inline-destinations)"
```

---

### Task 8: Update `pallas_kernel/README.md` with `spread_greedy` row

**Files:**
- Modify: `pallas_kernel/README.md`

- [ ] **Step 1: Add `spread_greedy` to the Algorithms table**

In `pallas_kernel/README.md`, find the table titled `### Algorithms` (with columns `Scheduler | Approach | Always physically feasible?`). Add a new row after `literal_greedy`:

```markdown
| `spread_greedy(k)` | `literal_greedy` plus a per-device cap of K outgoing AND K incoming DMAs per round. K=1 is P2P-style (each device sends/receives at most 1 DMA per round); K=∞ is `literal_greedy` | Yes |
```

- [ ] **Step 2: Add `spread_greedy(k=2)` column to the routing × scheduler matrix**

In the same file, find the table titled `### Routing × scheduler performance matrix (physical-edge model)` (columns include `orbit_greedy_full | literal_greedy | ilp_literal | cpsat_literal (warm)`). Add a new column at the end: `spread_greedy(k=2)`. Fill in the loaded-8×4×4 cell from the K-sweep probe's K=2 row (e.g. `94 (+19)`); other rows can be filled with `—` since the K-sweep was only run on the loaded 8×4×4.

- [ ] **Step 3: Add a "When to use which" row**

In the table titled `### When to use which` (columns `Routing class | Recommended scheduler | Why`), update the **Loaded TPU routing** row's recommendation to:

```
**`spread_greedy(k=2)` (current testbed candidate, fixtures shipped) — or `cpsat_literal` warm-started (makespan 78, projected +7.5% but measured ~0% on TPU)** — fall back to `orbit_greedy` (makespan 85) for the no-CP-SAT baseline
```

with the rationale text updated to mention the TPU measurement gap.

- [ ] **Step 4: Update the Files table**

Find the table at the top with the `outputs/` row. Update the list of current outputs to include the two new kernels:

```
... `cpsat_literal_warm_8_4_4.py` ... `cpsat_literal_warm_inline_8_4_4.py` ... `spread_greedy_k2_8_4_4.py`, and `spread_greedy_k2_inline_8_4_4.py` (per-device DMA-cap K=2 schedule; SMEM and inline variants).
```

- [ ] **Step 5: Add a `spread_greedy` example invocation**

In the `### Example invocations` block, add (after the existing cpsat_literal_warm examples):

```bash
# spread_greedy(k=2) — per-device-DMA-capped headline; ship both regular
# and inline-destinations kernels:
python pallas_kernel/gen_orbit_greedy_kernel.py \
    --slice 8,4,4 \
    --routing-table fixtures/routing_table_8x4x4_twist.json \
    --schedule-in fixtures/schedule_8x4x4_loaded_spread_greedy_k2.json \
    --function-name _ragged_a2a_kernel_spread_greedy_k2_8_4_4 \
    --out pallas_kernel/outputs/_ragged_a2a_kernel_spread_greedy_k2_8_4_4.py

# Other K values (1, 3, 4) ship as fixtures only; regenerate the kernel
# with the same command pattern, substituting the K in the schedule path
# and output filename.
```

- [ ] **Step 6: Commit**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
git add pallas_kernel/README.md
git commit -m "pallas_kernel/README: document spread_greedy(k) + K=2 kernels"
```

---

### Task 9: Update root `README.md` with `spread_greedy` row

**Files:**
- Modify: `README.md` (repo root)

- [ ] **Step 1: Add `spread_greedy` to the Scheduling algorithms table**

In `README.md`, find the table titled `### Scheduling algorithms`. Add a new row after `literal_greedy`:

```markdown
| `spread_greedy` | `literal_greedy` plus per-device-per-round DMA cap K | At K=1, each device sends/receives 1 DMA per round (makespan ≥ N-1). At K=∞, identical to `literal_greedy`. Intermediate K trades simulator makespan for per-device uniformity. Motivated by the 2026-05-17 finding that the makespan-78 `cpsat_literal_warm` kernel measured 132764 gbps on TPU v5e — within noise of `orbit_greedy`'s 132758 and below P2P's 134541, despite +9% simulator projection. See [2026-05-17 exploration](eval/explorations/2026-05-17-spread-scheduling/) |
```

- [ ] **Step 2: Add a TL;DR link**

Below the table, in the bulleted list of exploration links, add:

```
- [eval/explorations/2026-05-17-spread-scheduling/](eval/explorations/2026-05-17-spread-scheduling/) — `spread_greedy(k)` shipped to test the hypothesis that per-device DMA-engine oversubscription, not round count, is the binding TPU wall-clock constraint. K=2 is the headline (makespan ~XX, vs cpsat_literal_warm's 78), saved as `fixtures/schedule_8x4x4_loaded_spread_greedy_k2.json` and pre-generated Pallas kernel.
```

(Fill in `XX` with the K=2 makespan from the probe.)

- [ ] **Step 3: Update the Layout entry for `pallas_kernel/`**

Find the bullet starting `- pallas_kernel/`. Update the list of current outputs to include `spread_greedy_k2_8_4_4` and `spread_greedy_k2_inline_8_4_4`. The existing wording mentions `cpsat_literal_warm_inline_8_4_4`; extend it:

```
... and `cpsat_literal_warm_inline_8_4_4` ... ), and `spread_greedy_k2_8_4_4` / `spread_greedy_k2_inline_8_4_4` (per-device DMA-cap K=2; testbed for the DMA-oversubscription hypothesis from 2026-05-17).
```

- [ ] **Step 4: Commit**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
git add README.md
git commit -m "README: register spread_greedy + link 2026-05-17 exploration"
```

---

### Task 10: Document the exploration outcome

**Files:**
- Modify: `eval/explorations/2026-05-17-spread-scheduling/RESULTS.md`
- Modify: `eval/explorations/2026-05-17-spread-scheduling/README.md`

- [ ] **Step 1: Fill in the RESULTS.md table**

Open `eval/explorations/2026-05-17-spread-scheduling/RESULTS.md` and replace the `(pending)` rows with the actual data from `01_spread_sweep_results.json`. The table should look like:

```markdown
| K | makespan | viol | max DMAs/device-round | avg DMAs/device-round | n_rounds_with_dma | runtime |
|---:|---:|---:|:---:|---:|---:|---:|
| 1 | <X> | 0 | 1 / 1 | <X> | <X> | <X>s |
| 2 | <X> | 0 | 2 / 2 | <X> | <X> | <X>s |
| 3 | <X> | 0 | 3 / 3 | <X> | <X> | <X>s |
| 4 | <X> | 0 | 4 / 4 | <X> | <X> | <X>s |
```

Add a Summary section below the table:

```markdown
## Summary

| K | makespan | vs cpsat_warm (78) | shipped as |
|---:|---:|:---:|---|
| 1 | <X> | (much higher) | fixture only |
| 2 | <X> | (higher; headline) | fixture + cns + Pallas kernel |
| 3 | <X> | (close) | fixture only |
| 4 | <X> | (close) | fixture only |

Key takeaway: K=2 makespan is <X>, a <pct>% increase over `cpsat_literal_warm`'s
makespan 78. If TPU measurement on the K=2 kernel comes in at or above the
cpsat_warm kernel's 132764 gbps, the DMA-oversubscription hypothesis is
supported and `spread_greedy` becomes the production-recommended scheduler.
If K=2 measures notably below 132764 gbps, the hypothesis is rejected for this
routing — the makespan-78 schedule is genuinely near-optimal at the simulator
level AND the gap to P2P is due to something other than DMA cap (HBM, ICI
per-link bandwidth, or per-DMA setup).
```

- [ ] **Step 2: Fill in the README.md "Outcome" section**

In `eval/explorations/2026-05-17-spread-scheduling/README.md`, replace the
`(Filled in after the probe completes.)` placeholder with:

- The K-by-K makespan results (one-line summary).
- A pointer to the headline kernel:
  `pallas_kernel/outputs/_ragged_a2a_kernel_spread_greedy_k2_8_4_4.py` and
  the inline variant.
- A pointer to the shipped fixtures:
  `fixtures/schedule_8x4x4_loaded_spread_greedy_k{1,2,3,4}.json`.
- A "Next step" line: empirical TPU measurement of the K=2 kernel vs the
  cpsat_warm kernel and the P2P reference. The decision rule is in
  `RESULTS.md`.

- [ ] **Step 3: Commit docs**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
git add eval/explorations/2026-05-17-spread-scheduling/RESULTS.md \
        eval/explorations/2026-05-17-spread-scheduling/README.md
git commit -m "exploration: document spread-scheduling K-sweep results"
```

---

## Post-implementation hand-off

After Task 10, the implementation is complete:

- New scheduler module `spread_greedy(k, order)` shipped with TDD coverage and CLI support.
- Four shipped fixture schedules (K=1, 2, 3, 4); K=2 promoted to `cns_schedules/`.
- Two Pallas kernels for K=2 (regular + `--inline-destinations`).
- Both READMEs updated; exploration folder documents the K-sweep + the falsifiable
  hypothesis.

The user (or a TPU operator) can now:
1. Run the K=2 kernel on TPU v5e to test the DMA-oversubscription hypothesis.
2. If K=2 doesn't beat cpsat_warm, swap to K=1 or K=4 (fixtures already shipped) without re-running the schedule generator.
3. Use the `--scheduler spread_greedy --k <K>` CLI to generate spread schedules for any other routing.

After this plan completes, invoke `superpowers:finishing-a-development-branch` for merge / PR / cleanup.
