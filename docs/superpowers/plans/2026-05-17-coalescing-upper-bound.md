# Coalescing Upper-Bound Diagnostic (Option D) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute the theoretical maximum DMA-coalescing factor achievable on the loaded 8×4×4 twisted routing under fixed routing + makespan ≤ 78. The single number this probe produces decides whether per-hop kernel coalescing (Options A/B from the throughput analysis) has any headroom at all, or whether Option 3 (data-layout coalescing under fixed routing) is dead.

**Architecture:** Two phases. **Phase 1** measures the coalescing factor of the *existing* makespan-78 schedule via direct counting — cheap, definitive lower bound on what coalescing buys without re-scheduling. **Phase 2** extends the existing `cpsat_literal` model with edge-active and edge-break booleans and a new objective (minimize total per-edge maximal-active-runs across rounds), producing the *upper bound* on coalescing achievable by jointly choosing schedule and assignments under fixed routing + capacity. Phase 2 only runs if Phase 1's number leaves room to gain. All work lives in `eval/explorations/2026-05-17-coalescing-upper-bound/`; no new modules in `twisted_analysis/`.

**Tech Stack:** Python 3.11, pytest, OR-Tools CP-SAT (already a project dependency), existing `twisted_analysis.io.schedule` data format (`{"round": int, "src": int, "dst": int, "path": [int, ...]}`).

---

## Background and decision rule

**Why this matters (one paragraph for the implementer):** The current Pallas kernel issues one `pltpu.make_async_remote_copy` per logical (src, dst) flow = 16,256 kernel-level DMAs total; the TPU hardware NoC handles the multi-hop routing transparently. Per-DMA scalar setup cost dominates measured throughput (inline-kernel datapoint: −50%). To reduce DMA count via coalescing, the kernel must operate at the per-hop level, which expands the kernel-level DMA count by `avg_hop_length ≈ 2`. So coalescing must achieve a ratio > 2 just to break even with the current single-DMA-per-flow kernel.

**Coalescing model (post-hoc):** Given a schedule of flows with `{round, src, dst, path}` and edge capacity = 1, in each absolute round `tau`, edge `(u, v)` is active iff some scheduled flow uses `(u, v)` at hop `h` with start round `s` such that `s + h = tau`. Coalescing = fuse consecutive active rounds on the same edge into a single DMA descriptor. **Uncoalesced descriptor count** = total per-hop DMAs = `sum over flows of (len(path) - 1)`. **Coalesced descriptor count** = number of maximal contiguous runs of active rounds per edge, summed across edges. **Coalescing factor = uncoalesced / coalesced.**

**Decision rule on Phase 1 result:**
- coalescing factor < 1.5 → STOP. Phase 2 unlikely to improve enough. Option 3 is dead.
- coalescing factor 1.5–2.5 → run Phase 2 to see if re-scheduling helps.
- coalescing factor ≥ 2.5 → run Phase 2 as confirmation; Option 3 is alive (A or B worth prototyping).

**Decision rule on Phase 2 result (upper bound):**
- upper bound < 2.0 → Option 3 is dead even with optimal re-scheduling.
- upper bound 2.0–3.0 → break-even territory; Option 3 marginal, A/B not worth kernel engineering cost.
- upper bound ≥ 3.0 → Option 3 has headroom; recommend prototyping Option B (cross-round same-edge coalescing) on a heavy edge first.

---

## File structure

All work in `eval/explorations/2026-05-17-coalescing-upper-bound/`:

| File | Responsibility |
|---|---|
| `README.md` | Goal, hypothesis, approach, decision rule |
| `descriptor_counter.py` | Pure function: `count_dma_descriptors(entries)` → `(uncoalesced, coalesced)` |
| `tests/test_descriptor_counter.py` | TDD tests for the counter |
| `01_count_existing_schedules.py` | Run counter on shipped schedules; save `01_results.json` |
| `01_results.json` | Phase 1 output |
| `02_cpsat_coalesce.py` | CP-SAT model: minimize coalesced count subject to makespan ≤ 78 + fixed routing + edge cap = 1 |
| `tests/test_cpsat_coalesce.py` | Smoke test on a tiny topology |
| `02_run_coalesce_upper_bound.py` | Run 02 on the real makespan-78 problem; save `02_results.json` |
| `02_results.json` | Phase 2 output |
| `RESULTS.md` | Full write-up + recommendation |

---

### Task 1: Exploration folder + README

**Files:**
- Create: `eval/explorations/2026-05-17-coalescing-upper-bound/README.md`

- [ ] **Step 1: Create the exploration folder and README**

```bash
mkdir -p eval/explorations/2026-05-17-coalescing-upper-bound/tests
```

Then write `eval/explorations/2026-05-17-coalescing-upper-bound/README.md` with the following content:

```markdown
# Coalescing upper-bound diagnostic (Option D)

## Problem

The makespan-78 cpsat_literal_warm Pallas kernel measured 132764 gbps on TPU v5e,
~1.3% below the P2P reference (134541 gbps) despite a simulator projection of
+7.5%. Three TPU datapoints (inline kernel = 50% throughput; K=2 spread_greedy
matches cpsat_warm; packet_size=2^17 = -16%) localize the bottleneck to
per-DMA scalar setup cost, with VMEM/pipeline parallelism saturated at ~4
in-flight × 32 KB.

The current Pallas kernel issues one DMA descriptor per logical (src, dst)
flow = 16,256 descriptors total; TPU hardware routes multi-hop transparently.
The only way to reduce descriptor count under fixed routing + no multi-dest
DMA is **per-edge coalescing**: switch the kernel to per-hop DMA structure
(expanding to ~32k descriptors at avg_hop=2), then fuse same-edge DMAs across
flows (in-round) or across adjacent rounds (cross-round) back to a smaller
count. Coalescing must exceed factor 2.0 just to break even with the current
single-DMA-per-flow kernel.

## Goal

Compute the theoretical maximum coalescing factor achievable on the loaded
8×4×4 twisted routing under fixed routing + makespan ≤ 78. If the upper
bound is < 2.0, Option 3 (data-layout coalescing under fixed routing) is
dead and we should not pursue kernel-level per-hop restructuring. If ≥ 3.0,
Option B (cross-round same-edge coalescing) is worth prototyping.

## Approach

### Phase 1: direct measurement on existing schedules (cheap)

For each shipped schedule (cpsat_literal_warm-78, spread_greedy_k2-92,
literal_greedy-87, orbit_greedy_full-85), expand flows into per-(edge, round)
contributions using `path`, then count both uncoalesced and coalesced
descriptors. Output per-schedule coalescing factors.

### Phase 2: CP-SAT re-scheduling for max coalescing (expensive, conditional)

Build a CP-SAT model with the same `y[f, s]` variables as `cpsat_literal` plus
edge-active booleans `a[(u,v), tau]` and edge-break booleans `b[(u,v), tau]`.
Constraints: each flow gets exactly one start round; edge capacity = 1 per
round; `a == sum of y contributing to that (edge, round)`; `b[edge, tau] = a[edge, tau] AND NOT a[edge, tau-1]`.
Objective: minimize `sum of b`. With `t_upper = 78`, this finds the
makespan-feasible schedule whose post-hoc coalescing factor is maximal.
Time budget: 1 hour with 8 workers. Reports best incumbent.

## Decision rules

(See plan: 2026-05-17-coalescing-upper-bound.md for decision-rule thresholds.)

## Compute budget

Phase 1: minutes. Phase 2: ~1 hour CP-SAT run, plus ~1 day of modeling work.
```

- [ ] **Step 2: Verify the folder structure**

Run:
```bash
ls eval/explorations/2026-05-17-coalescing-upper-bound/
```
Expected output includes `README.md` and `tests/`.

- [ ] **Step 3: Commit**

```bash
git add eval/explorations/2026-05-17-coalescing-upper-bound/README.md eval/explorations/2026-05-17-coalescing-upper-bound/tests/
git commit -m "docs(coalesce): scaffold Option D upper-bound exploration folder"
```

---

### Task 2: Descriptor counter with TDD

**Files:**
- Create: `eval/explorations/2026-05-17-coalescing-upper-bound/descriptor_counter.py`
- Create: `eval/explorations/2026-05-17-coalescing-upper-bound/tests/test_descriptor_counter.py`

- [ ] **Step 1: Write failing test for uncoalesced count on disjoint edges**

Create `tests/test_descriptor_counter.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from descriptor_counter import count_dma_descriptors


def test_disjoint_single_hop_flows():
    """Two flows on disjoint edges in same round: no coalescing possible."""
    entries = [
        {"round": 0, "src": 0, "dst": 1, "path": [0, 1]},
        {"round": 0, "src": 2, "dst": 3, "path": [2, 3]},
    ]
    uncoalesced, coalesced = count_dma_descriptors(entries)
    assert uncoalesced == 2
    assert coalesced == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd eval/explorations/2026-05-17-coalescing-upper-bound && .venv/bin/python -m pytest tests/test_descriptor_counter.py::test_disjoint_single_hop_flows -v
```

Expected: `ModuleNotFoundError: No module named 'descriptor_counter'`

(If `.venv/bin/python` does not exist at this path, use the project's main venv: `/home/xutingl/collective_comm/TwistedAnalysis/.venv/bin/python`.)

- [ ] **Step 3: Write minimal implementation**

Create `descriptor_counter.py`:

```python
"""Count per-edge per-round DMA descriptors before/after coalescing.

Schedule entries follow the on-disk schema:
  {"round": int, "src": int, "dst": int, "path": [int, ...]}
where `path` is the sequence of flat device IDs from src to dst (inclusive).

A flow with `round = s` and `path = [a, b, c, d]` uses:
  - edge (a, b) at absolute round s
  - edge (b, c) at absolute round s + 1
  - edge (c, d) at absolute round s + 2
"""
from collections import defaultdict


def count_dma_descriptors(entries):
    """Return (uncoalesced, coalesced) per-edge per-round descriptor counts.

    `uncoalesced` = total per-hop DMAs = sum over flows of (len(path) - 1).
    `coalesced` = sum over physical edges of the number of maximal contiguous
                  runs of active rounds for that edge.
    """
    edge_active = defaultdict(set)  # (u, v) -> set of absolute round ints
    uncoalesced = 0
    for e in entries:
        start = e["round"]
        path = e["path"]
        for h in range(len(path) - 1):
            edge = (path[h], path[h + 1])
            edge_active[edge].add(start + h)
            uncoalesced += 1

    coalesced = 0
    for active_rounds in edge_active.values():
        sorted_rounds = sorted(active_rounds)
        runs = 1
        for i in range(1, len(sorted_rounds)):
            if sorted_rounds[i] != sorted_rounds[i - 1] + 1:
                runs += 1
        coalesced += runs
    return uncoalesced, coalesced
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd eval/explorations/2026-05-17-coalescing-upper-bound && /home/xutingl/collective_comm/TwistedAnalysis/.venv/bin/python -m pytest tests/test_descriptor_counter.py::test_disjoint_single_hop_flows -v
```

Expected: PASS

- [ ] **Step 5: Write failing test for cross-round same-edge coalescing**

Append to `tests/test_descriptor_counter.py`:

```python
def test_cross_round_same_edge_coalesces():
    """Three flows on edge (0,1) in consecutive rounds: 1 coalesced descriptor."""
    entries = [
        {"round": 0, "src": 0, "dst": 1, "path": [0, 1]},
        {"round": 1, "src": 0, "dst": 1, "path": [0, 1]},
        {"round": 2, "src": 0, "dst": 1, "path": [0, 1]},
    ]
    uncoalesced, coalesced = count_dma_descriptors(entries)
    assert uncoalesced == 3
    assert coalesced == 1
```

- [ ] **Step 6: Run test (already passes, but verify)**

```bash
cd eval/explorations/2026-05-17-coalescing-upper-bound && /home/xutingl/collective_comm/TwistedAnalysis/.venv/bin/python -m pytest tests/test_descriptor_counter.py::test_cross_round_same_edge_coalesces -v
```

Expected: PASS

- [ ] **Step 7: Write failing test for multi-hop expansion**

Append to `tests/test_descriptor_counter.py`:

```python
def test_multihop_flow_expands_across_rounds():
    """Flow with 3-edge path starting at round 5 uses edges at rounds 5, 6, 7."""
    entries = [
        {"round": 5, "src": 0, "dst": 3, "path": [0, 1, 2, 3]},
    ]
    uncoalesced, coalesced = count_dma_descriptors(entries)
    assert uncoalesced == 3
    assert coalesced == 3  # three distinct edges, no coalescing


def test_multihop_flows_share_edge_across_rounds():
    """
    f1: round=0, path=[0,1,2] uses (0,1)@0 and (1,2)@1
    f2: round=2, path=[1,2,3] uses (1,2)@2 and (2,3)@3
    Edge (1,2) active at rounds {1, 2} -> contiguous, 1 run.
    """
    entries = [
        {"round": 0, "src": 0, "dst": 2, "path": [0, 1, 2]},
        {"round": 2, "src": 1, "dst": 3, "path": [1, 2, 3]},
    ]
    uncoalesced, coalesced = count_dma_descriptors(entries)
    assert uncoalesced == 4
    assert coalesced == 3
```

- [ ] **Step 8: Run all tests**

```bash
cd eval/explorations/2026-05-17-coalescing-upper-bound && /home/xutingl/collective_comm/TwistedAnalysis/.venv/bin/python -m pytest tests/test_descriptor_counter.py -v
```

Expected: 4 passed.

- [ ] **Step 9: Commit**

```bash
git add eval/explorations/2026-05-17-coalescing-upper-bound/descriptor_counter.py eval/explorations/2026-05-17-coalescing-upper-bound/tests/test_descriptor_counter.py
git commit -m "feat(coalesce): descriptor counter with cross-round + multi-hop tests"
```

---

### Task 3: Apply descriptor counter to shipped schedules (Phase 1)

**Files:**
- Create: `eval/explorations/2026-05-17-coalescing-upper-bound/01_count_existing_schedules.py`

The shipped schedules to measure (all in `fixtures/`):
- `schedule_8x4x4_loaded_cpsat_literal_warm.json` (makespan 78, the production fixture)
- `schedule_8x4x4_loaded_spread_greedy_k2.json` (makespan 92)
- `schedule_8x4x4_loaded_literal_greedy_lpt.json` (makespan 87)
- `schedule_8x4x4_loaded_orbit_greedy_full_lpt_tail_asc.json` (makespan 85)

- [ ] **Step 1: Write the Phase 1 script**

Create `01_count_existing_schedules.py`:

```python
"""Phase 1: measure coalescing factor on existing shipped schedules.

For each shipped fixture, compute uncoalesced and coalesced per-(edge, round)
descriptor counts and the coalescing factor (uncoalesced / coalesced). Also
report avg path length and active-edge-round occupancy.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from twisted_analysis.io.schedule import load_schedule
from descriptor_counter import count_dma_descriptors


FIXTURES = REPO / "fixtures"
OUT_PATH = Path(__file__).resolve().parent / "01_results.json"

SCHEDULES = [
    ("cpsat_literal_warm", "schedule_8x4x4_loaded_cpsat_literal_warm.json"),
    ("spread_greedy_k2",   "schedule_8x4x4_loaded_spread_greedy_k2.json"),
    ("literal_greedy_lpt", "schedule_8x4x4_loaded_literal_greedy_lpt.json"),
    ("orbit_greedy_full",  "schedule_8x4x4_loaded_orbit_greedy_full_lpt_tail_asc.json"),
]


def summarize(name: str, entries: list[dict]) -> dict:
    uncoalesced, coalesced = count_dma_descriptors(entries)
    n_flows = len(entries)
    path_lengths = [len(e["path"]) - 1 for e in entries]
    avg_hops = sum(path_lengths) / n_flows
    makespan = max(e["round"] + (len(e["path"]) - 1) for e in entries)
    factor = uncoalesced / coalesced if coalesced else float("inf")
    return {
        "name": name,
        "n_flows": n_flows,
        "makespan": makespan,
        "avg_hop_length": round(avg_hops, 3),
        "uncoalesced_descriptors": uncoalesced,
        "coalesced_descriptors": coalesced,
        "coalescing_factor": round(factor, 3),
        "break_even_factor_for_kernel_switch": round(avg_hops, 3),
        "headroom_above_break_even": round(factor - avg_hops, 3),
    }


def main() -> None:
    results = []
    for name, fname in SCHEDULES:
        path = FIXTURES / fname
        if not path.exists():
            print(f"SKIP {name}: {path} not found")
            continue
        entries = load_schedule(path)
        summary = summarize(name, entries)
        results.append(summary)
        print(
            f"{name:24s} | makespan={summary['makespan']:3d} | "
            f"uncoalesced={summary['uncoalesced_descriptors']:6d} | "
            f"coalesced={summary['coalesced_descriptors']:6d} | "
            f"factor={summary['coalescing_factor']:.3f} | "
            f"break_even={summary['break_even_factor_for_kernel_switch']:.3f} | "
            f"headroom={summary['headroom_above_break_even']:+.3f}"
        )
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis && .venv/bin/python eval/explorations/2026-05-17-coalescing-upper-bound/01_count_existing_schedules.py
```

Expected: one summary line per schedule, plus "Saved …/01_results.json".

- [ ] **Step 3: Verify results JSON exists and is valid**

```bash
.venv/bin/python -c "import json; r=json.load(open('eval/explorations/2026-05-17-coalescing-upper-bound/01_results.json')); print(f'{len(r)} schedules summarized'); print(json.dumps(r[0], indent=2))"
```

Expected: prints the first schedule's summary dict with all eight keys.

- [ ] **Step 4: Commit**

```bash
git add eval/explorations/2026-05-17-coalescing-upper-bound/01_count_existing_schedules.py eval/explorations/2026-05-17-coalescing-upper-bound/01_results.json
git commit -m "feat(coalesce): Phase 1 measurement on shipped schedules"
```

---

### Task 4: Phase 1 RESULTS write-up + decision gate

**Files:**
- Create: `eval/explorations/2026-05-17-coalescing-upper-bound/RESULTS.md`

- [ ] **Step 1: Read `01_results.json` and decide whether Phase 2 should run**

```bash
.venv/bin/python -c "
import json
results = json.load(open('eval/explorations/2026-05-17-coalescing-upper-bound/01_results.json'))
prod = next((r for r in results if r['name'] == 'cpsat_literal_warm'), None)
if prod is None:
    print('cpsat_literal_warm not found in results')
else:
    cf = prod['coalescing_factor']
    he = prod['headroom_above_break_even']
    print(f'production schedule (makespan 78): coalescing_factor={cf:.3f}, headroom={he:+.3f}')
    if cf < 1.5:
        print('DECISION: STOP. cf < 1.5 -> Option 3 dead. Skip Phase 2.')
    elif cf < 2.5:
        print('DECISION: RUN PHASE 2. cf in [1.5, 2.5) -> CP-SAT re-scheduling may help.')
    else:
        print('DECISION: RUN PHASE 2 (confirmation). cf >= 2.5 -> Option 3 likely alive.')
"
```

- [ ] **Step 2: Write Phase 1 results section**

Create `RESULTS.md` with the following content, substituting the actual Phase 1 numbers from `01_results.json`:

```markdown
# Results: Coalescing upper-bound diagnostic (Option D)

## Background

The makespan-78 cpsat_literal_warm Pallas kernel measured 132764 gbps on TPU
v5e, ~1.3% below the P2P reference (134541 gbps) despite a simulator projection
of +7.5%. Per-DMA scalar setup cost is the dominant bottleneck (inline-kernel
datapoint = -50% throughput). The only way to reduce DMA count under fixed
routing + no multi-dest DMA is per-edge coalescing of per-hop DMAs. This probe
measures the coalescing factor available.

## Phase 1: Direct measurement on shipped schedules

| Schedule | Makespan | n_flows | avg_hops | Uncoalesced DMAs | Coalesced DMAs | Coalescing Factor | Break-Even | Headroom |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| [fill from 01_results.json one row per schedule] |

**Production schedule (`cpsat_literal_warm`, makespan 78):**
- Uncoalesced descriptors: [fill]
- Coalesced descriptors: [fill]
- Coalescing factor: [fill]
- Break-even threshold (= avg_hops): [fill]
- Headroom above break-even: [fill]

**Phase 1 decision:** [STOP / RUN PHASE 2 with rationale based on rules above]
```

(The implementer fills in the bracketed values from `01_results.json` — do NOT leave brackets in the committed file.)

- [ ] **Step 3: Commit Phase 1 write-up**

```bash
git add eval/explorations/2026-05-17-coalescing-upper-bound/RESULTS.md
git commit -m "docs(coalesce): Phase 1 RESULTS + decision gate"
```

- [ ] **Step 4: Branch on decision**

If the decision was STOP (cf < 1.5): append a final "Recommendation" section to `RESULTS.md` stating "Option 3 is dead; do not pursue per-hop kernel restructuring. Per-DMA cost reduction must come from other levers (profiling, scalar-overhead reduction, packet-size sweep at 2^13/2^14/2^16)." Then commit and skip to the very end (no Tasks 5–7).

If the decision was RUN PHASE 2: proceed to Task 5.

---

### Task 5: CP-SAT coalescing model with smoke test

**Files:**
- Create: `eval/explorations/2026-05-17-coalescing-upper-bound/02_cpsat_coalesce.py`
- Create: `eval/explorations/2026-05-17-coalescing-upper-bound/tests/test_cpsat_coalesce.py`

- [ ] **Step 1: Write the CP-SAT coalescing solver**

Create `02_cpsat_coalesce.py`:

```python
"""CP-SAT model: minimize coalesced DMA descriptor count subject to makespan + fixed routing.

Variables (Boolean):
  y[f, s]          = 1 iff flow f starts at round s (0 <= s <= t_upper - L_f)
  a[(u,v), tau]    = 1 iff physical edge (u, v) is active at round tau
  b[(u,v), tau]    = 1 iff (u, v) "starts" a new active run at tau
                     (i.e., active at tau and not at tau-1)

Constraints:
  - sum_s y[f, s] == 1                                      (exactly-one start)
  - a[e, tau] == sum over (f, h) with e in path(f) at hop h of y[f, tau - h]
                                                            (capacity = 1 enforces a in {0, 1})
  - b[e, tau] >= a[e, tau] - a[e, tau - 1]                  (break detection)
  - b[e, tau] <= a[e, tau]
  - b[e, tau] <= 1 - a[e, tau - 1]                          (for tau >= 1)
  - b[e, 0]   == a[e, 0]

Objective:
  - minimize sum_{e, tau} b[e, tau]                         (total coalesced descriptors)
"""
from __future__ import annotations
from collections import defaultdict


def _flow_set(table, n):
    flows = []
    for s in range(n):
        for d in range(n):
            if s == d:
                continue
            flows.append((s, d, list(table[s][d])))
    return flows


def cpsat_coalesce(
    topology,
    table,
    *,
    t_upper: int,
    time_limit_s: int = 600,
    n_workers: int = 8,
    solver_msg: bool = False,
):
    """Solve the coalescing-minimization model.

    Returns (entries, coalesced_count) where entries is a list of
    {round, src, dst, path} dicts and coalesced_count is the best-found
    objective value (or None on infeasibility).
    """
    from ortools.sat.python import cp_model

    n = topology.n_nodes
    flows = _flow_set(table, n)
    model = cp_model.CpModel()

    # y[f, s]
    y: dict[tuple[int, int], cp_model.IntVar] = {}
    flow_starts: dict[int, list[int]] = {}
    for f_idx, (_s, _d, path) in enumerate(flows):
        L = len(path) - 1
        starts = list(range(0, t_upper - L + 1))
        if not starts:
            raise RuntimeError(
                f"t_upper={t_upper} too small: flow {f_idx} has L={L}"
            )
        var_list = []
        for s in starts:
            v = model.NewBoolVar(f"y_{f_idx}_{s}")
            y[(f_idx, s)] = v
            var_list.append(v)
        flow_starts[f_idx] = starts
        model.AddExactlyOne(var_list)

    # edge_hops[(u, v)] = list of (f_idx, h)
    edge_hops: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for f_idx, (_s, _d, path) in enumerate(flows):
        for h in range(len(path) - 1):
            edge_hops[(path[h], path[h + 1])].append((f_idx, h))

    # a[edge, tau] and capacity (sum of contributing y == a)
    a: dict[tuple[tuple[int, int], int], cp_model.IntVar] = {}
    for edge, demands in edge_hops.items():
        for tau in range(t_upper):
            contributing = []
            for f_idx, h in demands:
                s = tau - h
                if (f_idx, s) in y:
                    contributing.append(y[(f_idx, s)])
            v = model.NewBoolVar(f"a_{edge[0]}_{edge[1]}_{tau}")
            a[(edge, tau)] = v
            if contributing:
                model.Add(v == sum(contributing))
            else:
                model.Add(v == 0)

    # b[edge, tau]: break = active and not previously active
    b_terms = []
    for edge in edge_hops:
        for tau in range(t_upper):
            bv = model.NewBoolVar(f"b_{edge[0]}_{edge[1]}_{tau}")
            if tau == 0:
                model.Add(bv == a[(edge, 0)])
            else:
                a_curr = a[(edge, tau)]
                a_prev = a[(edge, tau - 1)]
                model.Add(bv >= a_curr - a_prev)
                model.Add(bv <= a_curr)
                model.Add(bv <= 1 - a_prev)
            b_terms.append(bv)

    model.Minimize(sum(b_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = int(n_workers)
    solver.parameters.log_search_progress = bool(solver_msg)

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, None

    entries = []
    for f_idx, (s_node, d_node, path) in enumerate(flows):
        chosen_s = None
        for s in flow_starts[f_idx]:
            if solver.Value(y[(f_idx, s)]) == 1:
                chosen_s = s
                break
        assert chosen_s is not None, f"no start chosen for flow {f_idx}"
        entries.append({
            "round": chosen_s, "src": s_node, "dst": d_node, "path": path,
        })
    return entries, int(solver.ObjectiveValue())
```

- [ ] **Step 2: Write smoke test on a tiny topology**

Create `tests/test_cpsat_coalesce.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest

from descriptor_counter import count_dma_descriptors


@pytest.fixture
def tiny_topology():
    """Trivial 4-node line topology for smoke testing."""
    from twisted_analysis.topology import Topology
    return Topology(slice=(4,))


def _build_tiny_routing(n):
    """Linear routing: src->dst always goes through min..max indices."""
    table = [[[] for _ in range(n)] for _ in range(n)]
    for s in range(n):
        for d in range(n):
            if s == d:
                table[s][d] = [s]
            elif s < d:
                table[s][d] = list(range(s, d + 1))
            else:
                table[s][d] = list(range(s, d - 1, -1))
    return table


def test_cpsat_coalesce_returns_feasible_schedule(tiny_topology):
    """Smoke test: solver returns a feasible schedule whose coalesced count
    matches the post-hoc counter on the returned entries."""
    from cpsat_coalesce import cpsat_coalesce  # noqa: PLC0415

    n = tiny_topology.n_nodes
    table = _build_tiny_routing(n)
    entries, reported = cpsat_coalesce(
        tiny_topology, table, t_upper=20, time_limit_s=30, n_workers=2,
    )
    assert entries is not None, "solver returned no schedule"
    assert reported is not None
    # Validate post-hoc count matches reported objective
    _uncoalesced, coalesced = count_dma_descriptors(entries)
    assert coalesced == reported, (
        f"post-hoc coalesced count {coalesced} != reported objective {reported}"
    )
    # Validate every flow scheduled exactly once
    n_flows = n * (n - 1)
    assert len(entries) == n_flows
```

Note the test imports `cpsat_coalesce`, but the file is `02_cpsat_coalesce.py`. Create a symlink or import-friendly name:

```bash
cd eval/explorations/2026-05-17-coalescing-upper-bound && ln -sf 02_cpsat_coalesce.py cpsat_coalesce.py
```

- [ ] **Step 3: Run the smoke test**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis && .venv/bin/python -m pytest eval/explorations/2026-05-17-coalescing-upper-bound/tests/test_cpsat_coalesce.py -v
```

Expected: PASS.

If the smoke test fails because `twisted_analysis.topology.Topology(slice=(4,))` is invalid (1-D topology), use a 2×2 torus instead: `Topology(slice=(2, 2))` (n=4) and update `_build_tiny_routing` to use simple 2D routing. Verify the existing topology API in `twisted_analysis/topology.py` first.

- [ ] **Step 4: Commit**

```bash
git add eval/explorations/2026-05-17-coalescing-upper-bound/02_cpsat_coalesce.py eval/explorations/2026-05-17-coalescing-upper-bound/cpsat_coalesce.py eval/explorations/2026-05-17-coalescing-upper-bound/tests/test_cpsat_coalesce.py
git commit -m "feat(coalesce): CP-SAT coalesce-minimization model + smoke test"
```

---

### Task 6: Run optimization on real makespan-78 problem (Phase 2)

**Files:**
- Create: `eval/explorations/2026-05-17-coalescing-upper-bound/02_run_coalesce_upper_bound.py`

- [ ] **Step 1: Write the Phase 2 runner**

Create `02_run_coalesce_upper_bound.py`:

```python
"""Phase 2: run cpsat_coalesce on the real loaded 8x4x4 problem with t_upper=78.

Outputs `02_results.json` with the best-found coalescing factor.
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from twisted_analysis.io.routing_table import load_routing_table
from twisted_analysis.topology import Topology
from cpsat_coalesce import cpsat_coalesce
from descriptor_counter import count_dma_descriptors


ROUTING_PATH = REPO / "fixtures" / "routing_table_8x4x4_twist.json"
OUT_PATH = Path(__file__).resolve().parent / "02_results.json"
SCHEDULE_OUT = Path(__file__).resolve().parent / "02_coalesce_schedule.json"


def main(time_limit_s: int = 3600, n_workers: int = 8) -> None:
    topology = Topology(slice=(8, 4, 4))
    table = load_routing_table(ROUTING_PATH)

    t0 = time.time()
    entries, reported = cpsat_coalesce(
        topology, table,
        t_upper=78,
        time_limit_s=time_limit_s,
        n_workers=n_workers,
        solver_msg=True,
    )
    runtime = time.time() - t0

    if entries is None:
        result = {
            "status": "no_incumbent",
            "t_upper": 78,
            "time_limit_s": time_limit_s,
            "runtime_s": round(runtime, 1),
        }
        OUT_PATH.write_text(json.dumps(result, indent=2))
        print("No incumbent found within time budget.")
        return

    uncoalesced, coalesced_post_hoc = count_dma_descriptors(entries)
    assert coalesced_post_hoc == reported, (
        f"sanity check failed: post-hoc={coalesced_post_hoc} vs reported={reported}"
    )
    factor = uncoalesced / coalesced_post_hoc
    makespan = max(e["round"] + (len(e["path"]) - 1) for e in entries)

    SCHEDULE_OUT.write_text(json.dumps(entries, indent=2))

    result = {
        "status": "ok",
        "t_upper": 78,
        "achieved_makespan": makespan,
        "time_limit_s": time_limit_s,
        "runtime_s": round(runtime, 1),
        "uncoalesced_descriptors": uncoalesced,
        "coalesced_descriptors": coalesced_post_hoc,
        "coalescing_factor_upper_bound": round(factor, 3),
        "break_even_factor": round(uncoalesced / 16256, 3),
        "headroom_above_break_even": round(factor - (uncoalesced / 16256), 3),
    }
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the Phase 2 probe**

Run via a bash script for reproducibility:

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis && PATH=".venv/bin:$PATH" .venv/bin/python -u eval/explorations/2026-05-17-coalescing-upper-bound/02_run_coalesce_upper_bound.py 2>&1 | tee eval/explorations/2026-05-17-coalescing-upper-bound/02_solver.log
```

Expected: solver runs up to 1 hour; on completion, prints the final JSON result and saves `02_results.json` + `02_coalesce_schedule.json`. If it hits the time limit without an incumbent, the script reports `status: no_incumbent` — that itself is data (the model is too hard at this scale, fall back to Phase 1's number as a lower bound on the upper bound).

- [ ] **Step 3: Verify output files**

```bash
ls eval/explorations/2026-05-17-coalescing-upper-bound/02_*.json && head -30 eval/explorations/2026-05-17-coalescing-upper-bound/02_results.json
```

Expected: `02_results.json` and `02_coalesce_schedule.json` exist; JSON has `coalescing_factor_upper_bound`.

- [ ] **Step 4: Commit results (but not the schedule JSON if it is large; check size first)**

```bash
ls -lh eval/explorations/2026-05-17-coalescing-upper-bound/02_*.json
```

If `02_coalesce_schedule.json` is < 5 MB, commit it. Otherwise, add it to `.gitignore` and commit only the small summary:

```bash
git add eval/explorations/2026-05-17-coalescing-upper-bound/02_run_coalesce_upper_bound.py eval/explorations/2026-05-17-coalescing-upper-bound/02_results.json eval/explorations/2026-05-17-coalescing-upper-bound/02_solver.log
# If small enough, also add 02_coalesce_schedule.json
git commit -m "feat(coalesce): Phase 2 CP-SAT upper-bound run on 8x4x4 makespan 78"
```

---

### Task 7: Final RESULTS + recommendation

**Files:**
- Modify: `eval/explorations/2026-05-17-coalescing-upper-bound/RESULTS.md`

- [ ] **Step 1: Append Phase 2 section to RESULTS.md**

Open `RESULTS.md` and append:

```markdown
## Phase 2: CP-SAT upper bound on coalescing under fixed routing + makespan ≤ 78

| Metric | Value |
|---|---:|
| Achieved makespan | [fill from 02_results.json] |
| Solver runtime | [fill] s |
| Uncoalesced descriptors | [fill] |
| Coalesced descriptors (upper bound) | [fill] |
| **Coalescing factor (upper bound)** | **[fill]** |
| Break-even factor for kernel switch | [fill] |
| Headroom above break-even | [fill] |

(Schedule saved to `02_coalesce_schedule.json` for downstream comparison.)

## Recommendation

[Use the Phase-2-decision rules from the plan to write one of:]

- **OPTION 3 IS DEAD.** Upper bound < 2.0 means even the optimal re-scheduling
  cannot recover the multi-hop kernel overhead. Per-DMA cost reduction must
  come from other levers: TPU profiler trace, scalar-overhead reduction,
  packet-size sweep at 2^13 / 2^14 / 2^16.

- **OPTION 3 IS MARGINAL.** Upper bound in [2.0, 3.0) means break-even
  territory. The engineering cost of switching to per-hop kernel structure
  + cross-round coalescing is unlikely to pay back for an expected gain of
  < 1.5× DMA reduction. Recommend deprioritizing in favor of profiling-led
  optimizations.

- **OPTION 3 HAS HEADROOM.** Upper bound ≥ 3.0 means cross-round same-edge
  coalescing (Option B from prior analysis) can plausibly close the gap to
  P2P and beyond. Recommend prototyping a per-hop Pallas kernel with Option B
  coalescing on a single heavy edge first, measuring on TPU, and scaling
  out only if the prototype shows >5% throughput gain.
```

(Fill in all bracketed values from `02_results.json` before committing.)

- [ ] **Step 2: Commit**

```bash
git add eval/explorations/2026-05-17-coalescing-upper-bound/RESULTS.md
git commit -m "docs(coalesce): Phase 2 RESULTS + final Option 3 recommendation"
```

---

## Self-review notes

**Spec coverage check:**
- Goal (compute coalescing upper bound under fixed routing + makespan ≤ 78): covered by Tasks 5–6.
- Phase 1 cheap diagnostic: covered by Tasks 2–4.
- Decision gate after Phase 1: covered by Task 4 Step 4.
- Decision recommendation after Phase 2: covered by Task 7 Step 1.
- Routing fixed (no routing redesign): the model uses the loaded routing table as-is; no flow paths are mutated.
- No multi-dest DMA assumption: coalescing only fuses DMAs on the *same physical edge* (no fan-out).

**Risks the implementer should flag (not pre-emptively work around):**
- Phase 2 model size: 1.27M y-vars + ~60k a-vars + ~60k b-vars + channeling constraints. If CP-SAT cannot find an incumbent in 1 hour, the bash command's logged status will say `no_incumbent`; treat that as the upper-bound being at least the Phase 1 number (because Phase 1's existing schedule is itself feasible at makespan 78). Document this in RESULTS.md if it occurs.
- If `02_coalesce_schedule.json` exceeds GitHub's 100 MB hard limit, do not commit it (the summary `02_results.json` is what matters for the decision).
