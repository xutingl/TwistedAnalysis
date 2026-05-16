# Beating P2P on Loaded 8×4×4: Scheduler Exploration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find a scheduler that produces a sim-makespan ≤ 83 (and ideally ≤ 80) on `slice=(8,4,4)` with the loaded routing at `fixtures/routing_table_8x4x4_twist.json`, beating the reference P2P kernel's 134541 gbps. Document every probe — what was tried, what its makespan was, runtime, and lessons learned — under `eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/`.

**Architecture.** Six probes, cheapest first, each writing structured JSON results into the exploration folder and updating a single rolling `RESULTS.md` table. Probes are layered: each new probe starts from the best schedule the previous probes found and reports whether it beat that. Three new schedulers (`cpsat_literal`, `lp_rounding`, `local_search`) land in `twisted_analysis/schedules/` so they're reusable by the existing kernel generator. The exploration folder holds run scripts and result JSONs but no production code. Final task generates a Pallas kernel from the best schedule found and updates `pallas_kernel/README.md` so the result is discoverable.

**Target arithmetic.** Reference: P2P kernel at 134541 gbps. Existing orbit_greedy at 132758 gbps = makespan 85. Linear scaling: P2P-equivalent makespan = 132758×85/134541 ≈ 83.87. So **makespan ≤ 83** is required to definitively beat P2P. Ordering sweep already found `orbit_greedy_full[lpt]` at makespan **84** (borderline). LB is **75**. The 75→83 gap is the search budget.

**Tech Stack.** Python 3.11+, existing `twisted_analysis` package, `pulp` (already installed for `ilp_literal`), `ortools` (new — `uv pip install ortools`), `numpy`. No new dependencies beyond ortools.

---

## File Structure

**New schedulers (in `twisted_analysis/schedules/`):**

- `cpsat_literal.py` — CP-SAT (OR-Tools) feasibility model on the literal flow set; parameterized by `t_upper` and `time_limit_s`. ~150 LOC.
- `lp_rounding.py` — LP-relaxation of the literal_ilp model + randomized rounding (Raghavan-Thompson) + feasibility repair via literal_greedy fallback. ~180 LOC.
- `local_search.py` — Local-search repair: start from a feasible schedule, repeatedly try moves (shift one flow's `round` by ±1; swap two flows' rounds) that strictly reduce the max edge-time occupancy. ~140 LOC.

**Schedule I/O additions (in `twisted_analysis/io/schedule.py`):**

- `schedule_from_cpsat_literal`, `schedule_from_lp_rounding`, `schedule_from_local_search` — thin adapters; each registered in `_SCHEDULER_DISPATCH`.

**Kernel generator extension (in `pallas_kernel/gen_orbit_greedy_kernel.py`):**

- Add `cpsat_literal`, `lp_rounding`, `local_search` to the `--scheduler` choices and dispatch through `schedule_from_algorithm`.

**Tests (in `tests/`):**

- `test_cpsat_literal.py` — TDD; instance: (2,4) ILP routing; assert makespan == 3, 0 violations.
- `test_lp_rounding.py` — TDD; instance: (2,4) ILP routing; assert feasible (0 violations), makespan ≤ literal_greedy's.
- `test_local_search.py` — TDD; instance: (2,2,4) loaded; assert makespan doesn't increase and stays feasible.

**Exploration folder (new, `eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/`):**

- `README.md` — problem statement, goal, list of probes with links, summary of conclusions (filled at the end).
- `RESULTS.md` — rolling table of every probe's (scheduler, params, makespan, violations, runtime); updated after every probe.
- `01_ordering_sweep.py` — deterministic-orderings sweep on `orbit_greedy_full` and `literal_greedy`.
- `01_ordering_sweep_results.json` — output.
- `02_random_orbit_shuffle.py` — random orbit-orderings on `orbit_greedy_full` (N=1000 seeds).
- `02_random_orbit_shuffle_results.json` — output.
- `03_cpsat_probe.py` — CP-SAT at decreasing `t_upper ∈ {84, 83, 82, 81, 80, 78, 76}`, 30 min/probe.
- `03_cpsat_results.json` — output.
- `04_lp_rounding_probe.py` — LP-relaxation once, then 100 randomized-rounding trials.
- `04_lp_rounding_results.json` — output.
- `05_local_search_probe.py` — local search from best-so-far, 10 min budget.
- `05_local_search_results.json` — output.
- `best_schedule.json` — the best schedule found across all probes.
- `pallas_kernel.py` (or `pallas_kernel_8x4x4_best.py`) — generated kernel for the best schedule.

---

## Execution Notes

- All commands assume CWD `/home/xutingl/collective_comm/TwistedAnalysis`.
- Use `.venv/bin/python` (not `uv run`) per project memory.
- Each probe script is self-contained: imports the scheduler, loads the routing fixture, runs, verifies, writes JSON. The exploration folder owns runtime artifacts; the schedulers live in production code so they're reusable.
- Long-running probes (CP-SAT, LP-rounding, local-search) should NOT run inside test files — they live in the exploration scripts.
- Before each new probe, read the current best from `eval/explorations/.../RESULTS.md` and only persist the new probe's schedule as `best_schedule.json` if it strictly improves.

---

### Task 1: Bootstrap exploration folder + install ortools

**Files:**
- Create: `eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/README.md`
- Create: `eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/RESULTS.md`

- [ ] **Step 1: Create exploration directory**

```bash
mkdir -p eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4
```

- [ ] **Step 2: Write README.md skeleton**

`eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/README.md`:

```markdown
# Beating the P2P kernel on loaded 8×4×4

## Problem

The reference Pallas point-to-point AllToAll kernel measures **134541 gbps** on TPU
v5e with `slice=(8,4,4)` under the routing table at
`fixtures/routing_table_8x4x4_twist.json` ("loaded" routing — externally produced,
likely escape-VC + OCS aware). Our `orbit_greedy_full[lpt_tail_asc]` schedule with
sim-makespan **85** measures **132758 gbps** — 1.3% slower than P2P.

By linear scaling (throughput ∝ 1/makespan in steady state), we need
**sim-makespan ≤ 83** to definitively beat P2P, and ideally lower since the
simulator omits per-step barrier latency, HBM contention, and VC arbitration.

Physical-edge LB on this routing is **75** (max edge load over the AllToAll
workload). Whether LB is achievable on this non-translation-equivariant routing
is open — literal ILP with CBC is intractable at N=128 (1.37M binary variables).

## Goal

Find a schedule with sim-makespan ≤ 83 (and ideally ≤ 80) on the loaded routing
through algorithmic search. Each probe is documented in `RESULTS.md` with its
makespan, violation count, and runtime.

## Probes (cheapest first)

1. [01_ordering_sweep.py](01_ordering_sweep.py) — Deterministic orderings on existing greedies.
2. [02_random_orbit_shuffle.py](02_random_orbit_shuffle.py) — Random orbit orderings on `orbit_greedy_full`.
3. [03_cpsat_probe.py](03_cpsat_probe.py) — Google OR-Tools CP-SAT at decreasing `t_upper`.
4. [04_lp_rounding_probe.py](04_lp_rounding_probe.py) — LP relaxation + randomized rounding.
5. [05_local_search_probe.py](05_local_search_probe.py) — Local-search repair on best-found.

## Conclusions

(Filled at the end of the exploration — see [RESULTS.md](RESULTS.md) for the rolling table.)
```

- [ ] **Step 3: Write RESULTS.md skeleton**

`eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/RESULTS.md`:

```markdown
# Probe Results: Beating P2P on Loaded 8×4×4

**Target:** sim-makespan ≤ 83 (P2P equivalent), LB = 75. Current best to beat: 84 (`orbit_greedy_full[lpt]`).

| Probe | Scheduler | Params | Makespan | Violations | Runtime | Result file |
|---|---|---|---:|---:|---:|---|
| baseline | orbit_greedy_full | lpt_tail_asc | 85 | 0 | 14s | `fixtures/schedule_8x4x4_loaded_orbit_greedy_full_lpt_tail_asc.json` |
| baseline | orbit_greedy_full | lpt | 84 | 0 | 14s | (this exploration, Task 2) |
| baseline | literal_greedy | lpt | 87 | 0 | 0.4s | `fixtures/schedule_8x4x4_loaded_literal_greedy_lpt.json` |
```

(Subsequent rows are added as each probe runs.)

- [ ] **Step 4: Install ortools**

```bash
uv pip install ortools
```

Expected: ortools installed; verify with `.venv/bin/python -c "from ortools.sat.python import cp_model; print(cp_model.__name__)"` (should print `ortools.sat.python.cp_model`).

- [ ] **Step 5: Commit**

```bash
git add eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/
git commit -m "exploration: bootstrap 8x4x4 loaded scheduler search

Adds skeleton README + RESULTS.md for the multi-probe search documented
in docs/superpowers/plans/2026-05-15-beating-p2p-loaded-8x4x4.md. Goal:
sim-makespan <= 83 on loaded 8x4x4 to beat P2P at 134541 gbps."
```

---

### Task 2: Phase 1 — Deterministic ordering sweep on existing greedies

**Files:**
- Create: `eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/01_ordering_sweep.py`
- Create: `eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/01_ordering_sweep_results.json`
- Modify: `eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/RESULTS.md`

- [ ] **Step 1: Write the sweep script**

`eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/01_ordering_sweep.py`:

```python
"""Phase 1: deterministic orderings on orbit_greedy_full and literal_greedy.

Runs every valid (scheduler, order) combination on loaded 8x4x4 routing and
writes one JSON row per combination."""
from __future__ import annotations
import json
import time
from pathlib import Path

from twisted_analysis.topology import Topology
from twisted_analysis.io.routing_table import load_routing_table
from twisted_analysis.io.schedule import (
    schedule_from_orbit_greedy_full,
    schedule_from_literal_greedy,
)
from twisted_analysis.schedules.verify import schedule_makespan, verify_capacity

ROUTING = "fixtures/routing_table_8x4x4_twist.json"
SLICE = (8, 4, 4)
OUT = Path(__file__).with_suffix("") .parent / "01_ordering_sweep_results.json"


def run():
    topology = Topology(slice=SLICE)
    table = load_routing_table(ROUTING)
    rows = []
    for order in ("lpt_tail_asc", "lpt", "spt", "tail_asc"):
        t0 = time.time()
        sch = schedule_from_orbit_greedy_full(topology, table, order=order)
        dt = time.time() - t0
        v = verify_capacity(sch)
        rows.append({
            "scheduler": "orbit_greedy_full",
            "order": order,
            "makespan": schedule_makespan(sch),
            "violations": len(v),
            "runtime_s": round(dt, 3),
            "n_flows": len(sch),
        })
    for order in ("lpt", "spt", "natural"):
        t0 = time.time()
        sch = schedule_from_literal_greedy(topology, table, order=order)
        dt = time.time() - t0
        v = verify_capacity(sch)
        rows.append({
            "scheduler": "literal_greedy",
            "order": order,
            "makespan": schedule_makespan(sch),
            "violations": len(v),
            "runtime_s": round(dt, 3),
            "n_flows": len(sch),
        })
    rows.sort(key=lambda r: r["makespan"])
    OUT.write_text(json.dumps(rows, indent=2))
    for r in rows:
        print(f"  {r['scheduler']:20s} {r['order']:14s}"
              f"  makespan={r['makespan']:3d}  viol={r['violations']:3d}"
              f"  t={r['runtime_s']:5.1f}s")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Run the sweep**

```bash
.venv/bin/python eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/01_ordering_sweep.py
```

Expected output (orderings, may differ slightly): 7 rows; minimum makespan should be **84** (`orbit_greedy_full[lpt]`); max around 109 (`literal_greedy[natural]`).

- [ ] **Step 3: Update RESULTS.md with new rows**

Append new rows to `eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/RESULTS.md` under a new "Phase 1: ordering sweep" subheading. Use the actual numbers from `01_ordering_sweep_results.json`. If a new best (≤ 83) appears, also save it as `best_schedule.json` (regenerate that JSON via the sweep code).

- [ ] **Step 4: Commit**

```bash
git add eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/01_ordering_sweep.py \
        eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/01_ordering_sweep_results.json \
        eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/RESULTS.md
git commit -m "exploration[phase1]: deterministic ordering sweep on loaded 8x4x4

orbit_greedy_full[lpt] reaches makespan 84; lpt_tail_asc gives 85; spt/tail_asc
worse. literal_greedy across all orderings stays at 87 or worse. Baseline to
beat for Phase 2."
```

---

### Task 3: Phase 1b — Random orbit-shuffle exploration

**Files:**
- Create: `eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/02_random_orbit_shuffle.py`
- Create: `eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/02_random_orbit_shuffle_results.json`
- Modify: `eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/RESULTS.md`

**Rationale.** `orbit_greedy_full`'s outcome depends on the orbit iteration order. The deterministic orderings (lpt/spt/tail_asc) test only a handful of orderings; random orderings explore the search space more broadly. Cheap probe (~14s per seed × 1000 seeds with progress checkpointing, but can short-circuit on first improvement).

- [ ] **Step 1: Write the random-shuffle probe**

`eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/02_random_orbit_shuffle.py`:

```python
"""Phase 1b: random orbit-orderings on orbit_greedy_full.

Calls the internal `compute_hop0_firing_times_full` style logic but with a
shuffled orbit order. Records the best ordering found and its makespan."""
from __future__ import annotations
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from twisted_analysis.topology import Topology
from twisted_analysis.io.routing_table import load_routing_table
from twisted_analysis.io.coords import flatten
from twisted_analysis.lp.orbit import compute_orbits
from twisted_analysis.schedules.orbit_greedy_full import _orbit_hop_edge_sets
from twisted_analysis.schedules.verify import verify_capacity, schedule_makespan
from twisted_analysis.io.schedule import save_schedule

ROUTING = "fixtures/routing_table_8x4x4_twist.json"
SLICE = (8, 4, 4)
N_SEEDS = 1000
OUT = Path(__file__).parent / "02_random_orbit_shuffle_results.json"
BEST = Path(__file__).parent / "02_best_random_shuffle_schedule.json"


def fire_with_ordering(per_orbit, ordering):
    """Greedy-fire orbits in the given order; return per-orbit hop-0 time."""
    edge_busy: dict[tuple[int, int], set[int]] = defaultdict(set)
    t_hop0: dict = {}
    for orbit_id in ordering:
        hops = per_orbit[orbit_id]
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


def assemble_schedule(t_hop0, orbits, table, slice_):
    entries = []
    for orbit_id, members in orbits.items():
        r = int(t_hop0[orbit_id])
        for (src, dst) in members:
            sf = flatten(src, slice_)
            df = flatten(dst, slice_)
            entries.append({"round": r, "src": sf, "dst": df,
                            "path": list(table[sf][df])})
    entries.sort(key=lambda e: (e["round"], e["src"]))
    return entries


def run():
    topology = Topology(slice=SLICE)
    table = load_routing_table(ROUTING)
    per_orbit = _orbit_hop_edge_sets(topology, table)
    orbit_ids = list(per_orbit.keys())
    orbits = compute_orbits(topology)

    best_makespan = None
    best_seed = None
    best_ordering = None
    history = []

    t_start = time.time()
    rng = random.Random(0)
    for seed in range(N_SEEDS):
        ordering = orbit_ids[:]
        rng.shuffle(ordering)
        t_hop0 = fire_with_ordering(per_orbit, ordering)
        sch = assemble_schedule(t_hop0, orbits, table, topology.slice)
        m = schedule_makespan(sch)
        if best_makespan is None or m < best_makespan:
            best_makespan = m
            best_seed = seed
            best_ordering = list(ordering)
            save_schedule(sch, BEST)
            print(f"  seed={seed} new best makespan={m}")
        history.append({"seed": seed, "makespan": m})

    result = {
        "best_makespan": best_makespan,
        "best_seed": best_seed,
        "n_seeds": N_SEEDS,
        "runtime_s": round(time.time() - t_start, 1),
        "histogram": _hist(history),
        "best_schedule_file": str(BEST.relative_to(BEST.parents[3])),
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(f"\nbest makespan {best_makespan} (seed {best_seed})")
    print(f"wrote {OUT}")


def _hist(history):
    counts = defaultdict(int)
    for h in history:
        counts[h["makespan"]] += 1
    return [{"makespan": m, "count": counts[m]} for m in sorted(counts)]


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Run the probe**

```bash
.venv/bin/python eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/02_random_orbit_shuffle.py
```

Expected runtime: ~4 hours at 14s/seed × 1000 (could be much less if Python overhead dominates). If too slow, reduce `N_SEEDS = 200`. Expected outcome: some seeds improve on 84; track best.

- [ ] **Step 3: If best ≤ 83, update best_schedule.json**

```bash
cp eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/02_best_random_shuffle_schedule.json \
   eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/best_schedule.json
```

- [ ] **Step 4: Update RESULTS.md**

Add a "Phase 1b: random orbit shuffle" subsection to RESULTS.md with the best seed, makespan, histogram summary, and runtime.

- [ ] **Step 5: Commit**

```bash
git add eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/02_random_orbit_shuffle.py \
        eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/02_random_orbit_shuffle_results.json \
        eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/02_best_random_shuffle_schedule.json \
        eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/best_schedule.json \
        eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/RESULTS.md
git commit -m "exploration[phase1b]: random orbit-shuffle sweep on loaded 8x4x4

<best makespan> across <N> seeds. <commentary>"
```

---

### Task 4: Add `cpsat_literal` scheduler (TDD)

**Files:**
- Create: `twisted_analysis/schedules/cpsat_literal.py`
- Create: `tests/test_cpsat_literal.py`
- Modify: `twisted_analysis/io/schedule.py` (register adapter)

- [ ] **Step 1: Write the failing test**

`tests/test_cpsat_literal.py`:

```python
"""CP-SAT literal scheduler: tests on small cells with known optima."""
from __future__ import annotations
from collections import Counter

import pytest

pytest.importorskip("ortools")

from twisted_analysis.topology import Topology, ILPRouter
from twisted_analysis.schedules.cpsat_literal import cpsat_literal
from twisted_analysis.schedules.verify import verify_capacity, schedule_makespan


def _table_from_ilp_router(slice_):
    t = Topology(slice=slice_)
    r = ILPRouter(t)
    table = [[None] * t.n_nodes for _ in range(t.n_nodes)]
    for s in range(t.n_nodes):
        for d in range(t.n_nodes):
            if s == d:
                table[s][d] = [s]
            else:
                table[s][d] = list(r.path(s, d))
    return t, table


def _physical_edge_lb(table, n):
    c: Counter = Counter()
    for s in range(n):
        for d in range(n):
            if s == d:
                continue
            path = table[s][d]
            for h in range(len(path) - 1):
                c[(path[h], path[h + 1])] += 1
    return max(c.values())


def test_cpsat_literal_2x4_ilp_lb_tight():
    """On (2,4) with ILP routing, LB is achievable at t_upper=LB."""
    t, table = _table_from_ilp_router((2, 4))
    lb = _physical_edge_lb(table, t.n_nodes)
    sch = cpsat_literal(t, table, t_upper=lb, time_limit_s=60)
    assert verify_capacity(sch) == []
    assert schedule_makespan(sch) <= lb


def test_cpsat_literal_infeasible_raises():
    """At t_upper < LB the model must be infeasible."""
    t, table = _table_from_ilp_router((2, 4))
    lb = _physical_edge_lb(table, t.n_nodes)
    with pytest.raises(RuntimeError, match="infeasible|no solution"):
        cpsat_literal(t, table, t_upper=lb - 1, time_limit_s=30)
```

- [ ] **Step 2: Run test to verify failure**

```bash
.venv/bin/python -m pytest tests/test_cpsat_literal.py -v
```

Expected: FAIL (`ModuleNotFoundError: No module named 'twisted_analysis.schedules.cpsat_literal'`).

- [ ] **Step 3: Write `cpsat_literal.py`**

`twisted_analysis/schedules/cpsat_literal.py`:

```python
"""CP-SAT feasibility / minimization on the literal flow set.

Variables (Boolean):
  y[f, s]  = 1 iff flow f is injected at time s.
              Domain of s: 0 <= s <= t_upper - L_f.

Constraints:
  - sum_s y[f, s] == 1                                    (exactly-one start)
  - for each (edge e, time tau): at-most-1 over (f, s)    (capacity)
       such that path(f) contains e at hop h with s + h == tau.

Objective (when minimize=True):
  - minimize the makespan M, where M >= s + L_f for the chosen y[f, s].

CP-SAT handles pseudo-Boolean and at-most-one constraints natively, which
is much more efficient than CBC on this structure. Use this when literal_ilp
(pulp + CBC) times out.
"""
from __future__ import annotations
from collections import defaultdict


def _flow_set(table: list[list[list[int]]], n: int):
    flows = []
    for s in range(n):
        for d in range(n):
            if s == d:
                continue
            flows.append((s, d, list(table[s][d])))
    return flows


def cpsat_literal(
    topology,
    table: list[list[list[int]]],
    *,
    t_upper: int,
    time_limit_s: int = 600,
    solver_msg: bool = False,
    n_workers: int = 8,
    minimize: bool = True,
) -> list[dict]:
    """Solve / feasibility-probe the literal scheduling problem with CP-SAT.

    Raises ImportError if ortools is not installed.
    Raises RuntimeError if the problem is infeasible at the given `t_upper`.
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise ImportError(
            "cpsat_literal requires ortools. Install: `uv pip install ortools`."
        ) from exc

    n = topology.n_nodes
    flows = _flow_set(table, n)

    model = cp_model.CpModel()

    y: dict[tuple[int, int], cp_model.IntVar] = {}
    for f_idx, (_s, _d, path) in enumerate(flows):
        L = len(path) - 1
        starts = list(range(0, t_upper - L + 1))
        if not starts:
            raise RuntimeError(
                f"t_upper={t_upper} too small: flow {f_idx} has L={L}, needs t_upper>=L+1"
            )
        var_list = []
        for s in starts:
            v = model.NewBoolVar(f"y_{f_idx}_{s}")
            y[(f_idx, s)] = v
            var_list.append(v)
        model.AddExactlyOne(var_list)

    # Edge capacity: for each (edge, tau), at most one (f, s) with path[f] containing edge at hop h, s+h=tau.
    edge_hops: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for f_idx, (_s, _d, path) in enumerate(flows):
        for h in range(len(path) - 1):
            edge_hops[(path[h], path[h + 1])].append((f_idx, h))

    for _edge, demands in edge_hops.items():
        for tau in range(t_upper):
            vars_here = []
            for f_idx, h in demands:
                s = tau - h
                if (f_idx, s) in y:
                    vars_here.append(y[(f_idx, s)])
            if len(vars_here) >= 2:
                model.AddAtMostOne(vars_here)

    if minimize:
        M = model.NewIntVar(0, t_upper, "M")
        for f_idx, (_s, _d, path) in enumerate(flows):
            L = len(path) - 1
            for s in range(0, t_upper - L + 1):
                model.Add(M >= s + L).OnlyEnforceIf(y[(f_idx, s)])
        model.Minimize(M)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = int(n_workers)
    if solver_msg:
        solver.parameters.log_search_progress = True

    status = solver.Solve(model)

    if status in (cp_model.INFEASIBLE,):
        raise RuntimeError(f"cpsat_literal: infeasible at t_upper={t_upper}")
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(
            f"cpsat_literal: solver returned status={status} (no incumbent found "
            f"within time_limit_s={time_limit_s})"
        )

    rounds: dict[tuple[int, int], int] = {}
    for f_idx, (src, dst, path) in enumerate(flows):
        L = len(path) - 1
        chosen = None
        for s in range(0, t_upper - L + 1):
            if solver.Value(y[(f_idx, s)]) == 1:
                chosen = s
                break
        if chosen is None:
            raise RuntimeError(f"cpsat_literal: no start picked for flow {f_idx}")
        rounds[(src, dst)] = chosen

    entries = []
    for s in range(n):
        for d in range(n):
            if s == d:
                continue
            entries.append({
                "round": rounds[(s, d)],
                "src": s, "dst": d,
                "path": list(table[s][d]),
            })
    entries.sort(key=lambda e: (e["round"], e["src"]))
    return entries
```

- [ ] **Step 4: Run test to verify passing**

```bash
.venv/bin/python -m pytest tests/test_cpsat_literal.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Add adapter in `twisted_analysis/io/schedule.py`**

Add after `schedule_from_ilp_literal` (around line 182):

```python
def schedule_from_cpsat_literal(
    topology: Topology,
    table: list[list[list[int]]],
    *,
    t_upper: int,
    time_limit_s: int = 600,
    n_workers: int = 8,
) -> list[dict]:
    """Adapter: cpsat_literal -> schedule entries."""
    from twisted_analysis.io.routing_table import validate_routing_table_shape
    from twisted_analysis.schedules.cpsat_literal import cpsat_literal

    validate_routing_table_shape(table, topology.n_nodes)
    return cpsat_literal(
        topology, table, t_upper=t_upper,
        time_limit_s=time_limit_s, n_workers=n_workers,
    )
```

And add to `_SCHEDULER_DISPATCH`:

```python
_SCHEDULER_DISPATCH = {
    "orbit_greedy": schedule_from_orbit_greedy,
    "orbit_greedy_full": schedule_from_orbit_greedy_full,
    "literal_greedy": schedule_from_literal_greedy,
    "ilp_literal": schedule_from_ilp_literal,
    "cpsat_literal": schedule_from_cpsat_literal,
}
```

Update the docstring of `schedule_from_algorithm` to list `cpsat_literal`.

- [ ] **Step 6: Run all related tests**

```bash
.venv/bin/python -m pytest tests/test_cpsat_literal.py tests/test_schedule_from_algorithm.py -v
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add twisted_analysis/schedules/cpsat_literal.py \
        twisted_analysis/io/schedule.py \
        tests/test_cpsat_literal.py
git commit -m "schedules: add cpsat_literal scheduler

CP-SAT (OR-Tools) literal-flow scheduler. Strictly stronger than CBC-based
ilp_literal on this structure (native at-most-one + parallel search). Used
to probe whether LB or near-LB makespans are reachable on routings where
CBC times out — including loaded 8x4x4."
```

---

### Task 5: Phase 2a — CP-SAT probe at decreasing `t_upper`

**Files:**
- Create: `eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/03_cpsat_probe.py`
- Create: `eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/03_cpsat_results.json`
- Modify: `eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/RESULTS.md`

- [ ] **Step 1: Write the probe**

`eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/03_cpsat_probe.py`:

```python
"""Phase 2a: CP-SAT at decreasing t_upper on loaded 8x4x4.

For each t_upper in {84, 83, 82, 81, 80, 78, 76}, run the CP-SAT solver
with a 1800s (30 min) wall-clock budget. Record:
  - status (FEASIBLE / INFEASIBLE / TIMEOUT)
  - actual makespan of the returned schedule (when FEASIBLE)
  - violation count (sanity, should be 0)
  - runtime

If a probe returns FEASIBLE with makespan < current best, save the schedule
to best_schedule.json."""
from __future__ import annotations
import json
import time
from pathlib import Path

from twisted_analysis.topology import Topology
from twisted_analysis.io.routing_table import load_routing_table
from twisted_analysis.io.schedule import save_schedule
from twisted_analysis.schedules.cpsat_literal import cpsat_literal
from twisted_analysis.schedules.verify import verify_capacity, schedule_makespan

ROUTING = "fixtures/routing_table_8x4x4_twist.json"
SLICE = (8, 4, 4)
T_UPPER_SCHEDULE = [84, 83, 82, 81, 80, 78, 76]
TIME_LIMIT_S = 1800
WORKERS = 8

OUT = Path(__file__).parent / "03_cpsat_results.json"
BEST_FROM_PHASE = Path(__file__).parent / "03_best_cpsat_schedule.json"
GLOBAL_BEST = Path(__file__).parent / "best_schedule.json"


def run():
    topology = Topology(slice=SLICE)
    table = load_routing_table(ROUTING)
    rows = []
    best_makespan = None
    for t_upper in T_UPPER_SCHEDULE:
        print(f"\n--- t_upper={t_upper}, budget={TIME_LIMIT_S}s ---", flush=True)
        t0 = time.time()
        try:
            sch = cpsat_literal(
                topology, table,
                t_upper=t_upper, time_limit_s=TIME_LIMIT_S,
                n_workers=WORKERS, solver_msg=True,
            )
            dt = time.time() - t0
            v = verify_capacity(sch)
            m = schedule_makespan(sch)
            row = {"t_upper": t_upper, "status": "FEASIBLE",
                   "makespan": m, "violations": len(v),
                   "runtime_s": round(dt, 1)}
            print(f"FEASIBLE: makespan={m}, viol={len(v)}, t={dt:.1f}s", flush=True)
            if best_makespan is None or m < best_makespan:
                best_makespan = m
                save_schedule(sch, BEST_FROM_PHASE)
                print(f"  saved schedule to {BEST_FROM_PHASE}", flush=True)
        except RuntimeError as e:
            dt = time.time() - t0
            msg = str(e)
            status = "INFEASIBLE" if "infeasible" in msg.lower() else "TIMEOUT"
            row = {"t_upper": t_upper, "status": status,
                   "makespan": None, "violations": None,
                   "runtime_s": round(dt, 1), "error": msg}
            print(f"{status}: {msg} (t={dt:.1f}s)", flush=True)
            if status == "INFEASIBLE":
                # no point probing tighter t_upper; break early
                rows.append(row)
                break
        rows.append(row)

    result = {
        "schedule": T_UPPER_SCHEDULE,
        "best_makespan": best_makespan,
        "best_schedule_file": str(BEST_FROM_PHASE) if best_makespan is not None else None,
        "rows": rows,
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Run the probe**

```bash
.venv/bin/python -u eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/03_cpsat_probe.py 2>&1 | tee eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/03_cpsat_probe.log
```

Expected runtime: up to 7 × 30 min = 3.5 hours. May terminate earlier on INFEASIBLE.

- [ ] **Step 3: If CP-SAT found a new best, promote to `best_schedule.json`**

```bash
# Check if 03_best_cpsat_schedule.json exists and was an improvement
.venv/bin/python -c "
import json
from pathlib import Path
from twisted_analysis.io.schedule import load_schedule
from twisted_analysis.schedules.verify import schedule_makespan
folder = Path('eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4')
cpsat = folder / '03_best_cpsat_schedule.json'
gbest = folder / 'best_schedule.json'
if not cpsat.exists():
    print('no cpsat schedule produced'); exit(0)
m_new = schedule_makespan(load_schedule(cpsat))
m_old = schedule_makespan(load_schedule(gbest)) if gbest.exists() else 10**9
print('cpsat makespan', m_new, 'vs current best', m_old)
if m_new < m_old:
    import shutil; shutil.copy(cpsat, gbest); print('promoted to best_schedule.json')
"
```

- [ ] **Step 4: Update RESULTS.md**

Add a "Phase 2a: CP-SAT probe" subsection summarizing each t_upper row.

- [ ] **Step 5: Commit**

```bash
git add eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/03_cpsat_probe.py \
        eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/03_cpsat_results.json \
        eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/03_cpsat_probe.log \
        eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/RESULTS.md \
        eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/best_schedule.json \
        eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/03_best_cpsat_schedule.json 2>/dev/null
git commit -m "exploration[phase2a]: CP-SAT probe at decreasing t_upper on loaded 8x4x4

<best makespan found by CP-SAT and at which t_upper, or TIMEOUT/INFEASIBLE outcomes>"
```

---

### Task 6: Add `lp_rounding` scheduler (TDD)

**Files:**
- Create: `twisted_analysis/schedules/lp_rounding.py`
- Create: `tests/test_lp_rounding.py`
- Modify: `twisted_analysis/io/schedule.py` (register adapter)

**Approach.** Solve the LP relaxation of the literal_ilp model (drop the binary constraint on `x[f, t]`, keep `0 <= x <= 1`). For each flow f, the LP yields a probability distribution over start times. Sample integer assignments via Raghavan-Thompson rounding: for each flow independently, draw a start time according to the LP's `x[f, ·]` distribution. After rounding, repair feasibility violations by displacing conflicting flows to a later time (greedy repair). Run N=100 trials; return the schedule with smallest makespan.

- [ ] **Step 1: Write the failing test**

`tests/test_lp_rounding.py`:

```python
"""LP-rounding scheduler: small-cell sanity tests."""
from __future__ import annotations
import pytest

pytest.importorskip("pulp")

from twisted_analysis.topology import Topology, ILPRouter
from twisted_analysis.schedules.lp_rounding import lp_rounding
from twisted_analysis.schedules.literal_greedy import literal_greedy
from twisted_analysis.schedules.verify import verify_capacity, schedule_makespan


def _table(slice_):
    t = Topology(slice=slice_)
    r = ILPRouter(t)
    n = t.n_nodes
    table = [[[s] if s == d else list(r.path(s, d)) for d in range(n)] for s in range(n)]
    return t, table


def test_lp_rounding_produces_feasible_2x4():
    t, table = _table((2, 4))
    sch = lp_rounding(t, table, t_upper=10, n_trials=20, seed=0)
    assert verify_capacity(sch) == []


def test_lp_rounding_beats_or_matches_literal_greedy_2x4():
    t, table = _table((2, 4))
    lg = schedule_makespan(literal_greedy(t, table, order="lpt"))
    sch = lp_rounding(t, table, t_upper=max(10, lg + 2), n_trials=50, seed=0)
    assert schedule_makespan(sch) <= lg + 1  # within 1 step of greedy in small cell
```

- [ ] **Step 2: Run the test to confirm failure**

```bash
.venv/bin/python -m pytest tests/test_lp_rounding.py -v
```

Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `lp_rounding.py`**

`twisted_analysis/schedules/lp_rounding.py`:

```python
"""LP relaxation of the literal scheduling ILP + randomized rounding.

Steps:
  1. Build the literal_ilp LP (no integrality constraint on x[f,t]).
  2. Solve once. Read out fractional values x_lp[f, t] ∈ [0, 1].
  3. For each trial: for each flow f, sample start time t from
     distribution x_lp[f, ·] (clipped+renormalized). Build raw schedule.
  4. Repair feasibility: greedily walk flows in increasing chosen start;
     if a flow conflicts on any edge-time, shift its start to the smallest
     later time with no conflict (this is the literal_greedy repair).
  5. Return the trial whose final makespan is smallest.

Polynomial-time. No LB-tightness guarantee, but in practice close to LP
bound (which equals the LP relaxation's optimum, a valid lower bound)."""
from __future__ import annotations
import random
from collections import defaultdict

from twisted_analysis.topology import Topology


def _flow_set(table, n):
    flows = []
    for s in range(n):
        for d in range(n):
            if s == d:
                continue
            flows.append((s, d, list(table[s][d])))
    return flows


def _solve_lp(flows, t_upper):
    """Return: dict (f_idx -> list[float] of length len(starts))."""
    import pulp
    prob = pulp.LpProblem("literal_lp_relaxation", pulp.LpMinimize)
    M = pulp.LpVariable("M", lowBound=0, upBound=t_upper, cat="Continuous")
    x = {}
    for f_idx, (_s, _d, path) in enumerate(flows):
        L = len(path) - 1
        starts = list(range(0, t_upper - L + 1))
        for s in starts:
            x[(f_idx, s)] = pulp.LpVariable(f"x_{f_idx}_{s}",
                                            lowBound=0, upBound=1,
                                            cat="Continuous")
        prob += pulp.lpSum(x[(f_idx, s)] for s in starts) == 1
        prob += M >= pulp.lpSum((s + L) * x[(f_idx, s)] for s in starts)

    edge_hops = defaultdict(list)
    for f_idx, (_s, _d, path) in enumerate(flows):
        for h in range(len(path) - 1):
            edge_hops[(path[h], path[h + 1])].append((f_idx, h))
    for _e, demands in edge_hops.items():
        for tau in range(t_upper):
            terms = []
            for f_idx, h in demands:
                s = tau - h
                if (f_idx, s) in x:
                    terms.append(x[(f_idx, s)])
            if len(terms) >= 2:
                prob += pulp.lpSum(terms) <= 1
    prob += M
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    out = {}
    for f_idx, (_s, _d, path) in enumerate(flows):
        L = len(path) - 1
        vals = [max(0.0, pulp.value(x[(f_idx, s)]) or 0.0)
                for s in range(0, t_upper - L + 1)]
        z = sum(vals)
        if z > 0:
            vals = [v / z for v in vals]
        else:
            vals = [1.0 / len(vals)] * len(vals) if vals else [1.0]
        out[f_idx] = vals
    return out, pulp.value(M)


def _sample_round(flows, x_lp, t_upper, rng):
    edge_busy: dict[tuple[int, int], set[int]] = defaultdict(set)
    rounds = {}
    # Sample start per flow.
    starts = [None] * len(flows)
    for f_idx, (_s, _d, path) in enumerate(flows):
        L = len(path) - 1
        probs = x_lp[f_idx]
        # rng.choices returns a list of 1 element.
        choice = rng.choices(range(len(probs)), weights=probs, k=1)[0]
        starts[f_idx] = choice
    # Repair: process in increasing chosen start, displace conflicts forward.
    order = sorted(range(len(flows)), key=lambda i: starts[i])
    for f_idx in order:
        path = flows[f_idx][2]
        L = len(path) - 1
        start = starts[f_idx]
        while True:
            conflict = False
            for h in range(L):
                if (start + h) in edge_busy[(path[h], path[h + 1])]:
                    conflict = True
                    break
            if not conflict:
                break
            start += 1
        for h in range(L):
            edge_busy[(path[h], path[h + 1])].add(start + h)
        rounds[f_idx] = start
    return rounds


def _makespan(flows, rounds):
    m = 0
    for f_idx, (_s, _d, path) in enumerate(flows):
        L = len(path) - 1
        f = rounds[f_idx] + L
        if f > m:
            m = f
    return m


def lp_rounding(
    topology: Topology,
    table: list[list[list[int]]],
    *,
    t_upper: int,
    n_trials: int = 100,
    seed: int = 0,
) -> list[dict]:
    """Solve LP relaxation, randomly round, repair, take best of n_trials.

    Raises ImportError if pulp is missing.
    """
    try:
        import pulp  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "lp_rounding requires pulp. Install: `uv pip install pulp`."
        ) from exc

    n = topology.n_nodes
    flows = _flow_set(table, n)
    x_lp, lp_m = _solve_lp(flows, t_upper)

    best_rounds = None
    best_m = None
    rng = random.Random(seed)
    for _ in range(n_trials):
        rounds = _sample_round(flows, x_lp, t_upper, rng)
        m = _makespan(flows, rounds)
        if best_m is None or m < best_m:
            best_m = m
            best_rounds = rounds

    entries = []
    for f_idx, (src, dst, path) in enumerate(flows):
        entries.append({
            "round": int(best_rounds[f_idx]),
            "src": src, "dst": dst,
            "path": list(path),
        })
    entries.sort(key=lambda e: (e["round"], e["src"]))
    return entries
```

- [ ] **Step 4: Run the test to verify passing**

```bash
.venv/bin/python -m pytest tests/test_lp_rounding.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Add adapter in `twisted_analysis/io/schedule.py`**

Append after `schedule_from_cpsat_literal`:

```python
def schedule_from_lp_rounding(
    topology: Topology,
    table: list[list[list[int]]],
    *,
    t_upper: int,
    n_trials: int = 100,
    seed: int = 0,
) -> list[dict]:
    """Adapter: lp_rounding -> schedule entries."""
    from twisted_analysis.io.routing_table import validate_routing_table_shape
    from twisted_analysis.schedules.lp_rounding import lp_rounding

    validate_routing_table_shape(table, topology.n_nodes)
    return lp_rounding(topology, table, t_upper=t_upper,
                       n_trials=n_trials, seed=seed)
```

And update `_SCHEDULER_DISPATCH`:

```python
    "lp_rounding": schedule_from_lp_rounding,
```

- [ ] **Step 6: Commit**

```bash
git add twisted_analysis/schedules/lp_rounding.py \
        twisted_analysis/io/schedule.py \
        tests/test_lp_rounding.py
git commit -m "schedules: add lp_rounding scheduler (LP relaxation + randomized rounding)

Solves LP relaxation of literal_ilp once, then runs n_trials of randomized
rounding + greedy repair. Polynomial-time alternative to CP-SAT for large
cells. Provably within O(log m) of LP bound; in practice within 1-2 steps.

Tests on (2,4) ILP pass; expected use case is loaded 8x4x4 where CP-SAT may
time out."
```

---

### Task 7: Phase 2b — LP-rounding probe on loaded 8×4×4

**Files:**
- Create: `eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/04_lp_rounding_probe.py`
- Create: `eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/04_lp_rounding_results.json`
- Modify: `eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/RESULTS.md`

- [ ] **Step 1: Write the probe**

`eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/04_lp_rounding_probe.py`:

```python
"""Phase 2b: LP-relaxation + randomized rounding on loaded 8x4x4."""
from __future__ import annotations
import json
import time
from pathlib import Path

from twisted_analysis.topology import Topology
from twisted_analysis.io.routing_table import load_routing_table
from twisted_analysis.io.schedule import save_schedule
from twisted_analysis.schedules.lp_rounding import lp_rounding
from twisted_analysis.schedules.verify import verify_capacity, schedule_makespan

ROUTING = "fixtures/routing_table_8x4x4_twist.json"
SLICE = (8, 4, 4)
T_UPPER = 95  # generous; LP just sees this as a horizon
N_TRIALS = 200
SEED = 0
OUT = Path(__file__).parent / "04_lp_rounding_results.json"
BEST_PHASE = Path(__file__).parent / "04_best_lp_rounding_schedule.json"


def run():
    topology = Topology(slice=SLICE)
    table = load_routing_table(ROUTING)
    t0 = time.time()
    sch = lp_rounding(topology, table,
                      t_upper=T_UPPER, n_trials=N_TRIALS, seed=SEED)
    dt = time.time() - t0
    m = schedule_makespan(sch)
    v = verify_capacity(sch)
    save_schedule(sch, BEST_PHASE)
    result = {
        "t_upper": T_UPPER,
        "n_trials": N_TRIALS,
        "best_makespan": m,
        "violations": len(v),
        "runtime_s": round(dt, 1),
        "schedule_file": str(BEST_PHASE),
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(f"LP-rounding best of {N_TRIALS} trials: makespan={m}, "
          f"viol={len(v)}, t={dt:.1f}s")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Run the probe**

```bash
.venv/bin/python -u eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/04_lp_rounding_probe.py 2>&1 | tee eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/04_lp_rounding_probe.log
```

Expected runtime: 5-30 minutes (LP solve ~minutes, 200 trials of sampling+repair ~seconds each).

- [ ] **Step 3: Promote to global best if better**

```bash
.venv/bin/python -c "
from pathlib import Path
from twisted_analysis.io.schedule import load_schedule
from twisted_analysis.schedules.verify import schedule_makespan
import shutil
folder = Path('eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4')
new = folder / '04_best_lp_rounding_schedule.json'
gbest = folder / 'best_schedule.json'
m_new = schedule_makespan(load_schedule(new))
m_old = schedule_makespan(load_schedule(gbest)) if gbest.exists() else 10**9
print('lp_rounding makespan', m_new, 'vs current best', m_old)
if m_new < m_old:
    shutil.copy(new, gbest); print('promoted')
"
```

- [ ] **Step 4: Update RESULTS.md**

Add a "Phase 2b: LP-rounding" subsection.

- [ ] **Step 5: Commit**

```bash
git add eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/04_lp_rounding_probe.py \
        eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/04_lp_rounding_results.json \
        eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/04_lp_rounding_probe.log \
        eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/04_best_lp_rounding_schedule.json \
        eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/best_schedule.json \
        eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/RESULTS.md
git commit -m "exploration[phase2b]: LP-rounding probe on loaded 8x4x4

<best makespan, n_trials, runtime>"
```

---

### Task 8: Add `local_search` scheduler (TDD)

**Files:**
- Create: `twisted_analysis/schedules/local_search.py`
- Create: `tests/test_local_search.py`
- Modify: `twisted_analysis/io/schedule.py` (register adapter)

**Approach.** Hill-climbing on round assignments. Define a single move type: for each pair of flows (f1, f2) where flow f1's last hop is at the bottleneck time, try shifting f1's round by -1 (earlier) and verify feasibility. If a shift is feasible AND strictly reduces the makespan (or reduces conflicts at the bottleneck without increasing makespan), accept. Iterate until no improving move found.

- [ ] **Step 1: Write the failing test**

`tests/test_local_search.py`:

```python
"""Local-search scheduler: feasibility + monotone improvement tests."""
from __future__ import annotations
import pytest

from twisted_analysis.topology import Topology, ILPRouter
from twisted_analysis.schedules.local_search import local_search_repair
from twisted_analysis.schedules.literal_greedy import literal_greedy
from twisted_analysis.schedules.verify import verify_capacity, schedule_makespan


def _table(slice_):
    t = Topology(slice=slice_)
    r = ILPRouter(t)
    n = t.n_nodes
    table = [[[s] if s == d else list(r.path(s, d)) for d in range(n)] for s in range(n)]
    return t, table


def test_local_search_preserves_feasibility_2x4():
    t, table = _table((2, 4))
    seed = literal_greedy(t, table, order="lpt")
    out = local_search_repair(t, table, seed, max_iters=50)
    assert verify_capacity(out) == []


def test_local_search_does_not_worsen_makespan_2x4():
    t, table = _table((2, 4))
    seed = literal_greedy(t, table, order="lpt")
    m0 = schedule_makespan(seed)
    out = local_search_repair(t, table, seed, max_iters=200)
    assert schedule_makespan(out) <= m0
```

- [ ] **Step 2: Run the test to confirm failure**

```bash
.venv/bin/python -m pytest tests/test_local_search.py -v
```

Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write `local_search.py`**

`twisted_analysis/schedules/local_search.py`:

```python
"""Local-search repair scheduler.

Given a seed schedule (any feasible list of {round, src, dst, path} entries),
repeatedly try the cheapest improving move:
  - For each flow f currently finishing at time M (the makespan):
      try shifting f's round by -k for k = 1, 2, ... until either the shift
      is feasible (no edge-time conflict) or we exhaust the search.
  - If a shift reduces M, accept and continue.

When no shift reduces M, terminate. Polynomial per iteration: O(F * M * D)
where F = flows, M = current makespan, D = max path length.

This is a refinement step — call after a base scheduler produces a feasible
schedule (orbit_greedy_full, lp_rounding, etc.)."""
from __future__ import annotations
from collections import defaultdict
from copy import deepcopy

from twisted_analysis.topology import Topology


def _occupancy(schedule):
    """Return {(edge, time): flow_idx or sentinel}; assumes valid schedule."""
    occ: dict[tuple[tuple[int, int], int], int] = {}
    for f_idx, e in enumerate(schedule):
        r = e["round"]
        path = e["path"]
        for h in range(len(path) - 1):
            occ[((path[h], path[h + 1]), r + h)] = f_idx
    return occ


def _makespan(schedule):
    m = 0
    for e in schedule:
        L = len(e["path"]) - 1
        f = e["round"] + L
        if f > m:
            m = f
    return m


def _try_shift(schedule, occ, f_idx, new_round):
    """Return new occ if shifting flow f_idx to new_round is feasible, else None."""
    if new_round < 0:
        return None
    e = schedule[f_idx]
    old_round = e["round"]
    if new_round == old_round:
        return occ
    path = e["path"]
    # Remove old slots from occ.
    candidate = dict(occ)
    for h in range(len(path) - 1):
        key = ((path[h], path[h + 1]), old_round + h)
        if candidate.get(key) == f_idx:
            del candidate[key]
    # Check new slots free.
    for h in range(len(path) - 1):
        key = ((path[h], path[h + 1]), new_round + h)
        if key in candidate:
            return None
    # Apply.
    for h in range(len(path) - 1):
        key = ((path[h], path[h + 1]), new_round + h)
        candidate[key] = f_idx
    return candidate


def local_search_repair(
    topology: Topology,
    table: list[list[list[int]]],  # accepted for signature symmetry, unused
    seed_schedule: list[dict],
    *,
    max_iters: int = 1000,
) -> list[dict]:
    """Repeatedly apply earliest-shift moves to reduce makespan.

    Returns a feasible schedule with makespan <= seed_schedule's.
    """
    schedule = [dict(e) for e in seed_schedule]
    occ = _occupancy(schedule)
    M = _makespan(schedule)

    for _ in range(max_iters):
        improved = False
        # Identify flows finishing at M.
        late_flows = []
        for f_idx, e in enumerate(schedule):
            L = len(e["path"]) - 1
            if e["round"] + L == M:
                late_flows.append(f_idx)
        # Try to shift each by -1, -2, ...
        for f_idx in late_flows:
            old_r = schedule[f_idx]["round"]
            best_shift = None
            for new_r in range(old_r - 1, -1, -1):
                cand_occ = _try_shift(schedule, occ, f_idx, new_r)
                if cand_occ is not None:
                    best_shift = (new_r, cand_occ)
                else:
                    break  # earlier slots usually only become harder; bail
            if best_shift is not None:
                new_r, new_occ = best_shift
                schedule[f_idx]["round"] = new_r
                occ = new_occ
                improved = True
                break
        if not improved:
            break
        M = _makespan(schedule)

    schedule.sort(key=lambda e: (e["round"], e["src"]))
    return schedule
```

- [ ] **Step 4: Run the test to verify passing**

```bash
.venv/bin/python -m pytest tests/test_local_search.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Add adapter in `twisted_analysis/io/schedule.py`**

```python
def schedule_from_local_search(
    topology: Topology,
    table: list[list[list[int]]],
    *,
    seed_schedule: list[dict],
    max_iters: int = 1000,
) -> list[dict]:
    """Adapter: local_search_repair on a seed schedule."""
    from twisted_analysis.io.routing_table import validate_routing_table_shape
    from twisted_analysis.schedules.local_search import local_search_repair

    validate_routing_table_shape(table, topology.n_nodes)
    return local_search_repair(topology, table, seed_schedule, max_iters=max_iters)
```

Add to `_SCHEDULER_DISPATCH`:

```python
    "local_search": schedule_from_local_search,
```

- [ ] **Step 6: Commit**

```bash
git add twisted_analysis/schedules/local_search.py \
        twisted_analysis/io/schedule.py \
        tests/test_local_search.py
git commit -m "schedules: add local_search_repair scheduler

Hill-climbing post-processor on a feasible seed schedule: shifts the
makespan-defining flows earlier when feasible. Polynomial per iteration.
No LB guarantee but cheap to chain after any greedy or LP-rounding output."
```

---

### Task 9: Phase 3 — Local-search repair from current best

**Files:**
- Create: `eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/05_local_search_probe.py`
- Create: `eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/05_local_search_results.json`
- Modify: `eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/RESULTS.md`

- [ ] **Step 1: Write the probe**

`eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/05_local_search_probe.py`:

```python
"""Phase 3: run local_search_repair starting from each schedule produced in
earlier phases and from each baseline. Record per-seed improvement."""
from __future__ import annotations
import json
import time
from pathlib import Path

from twisted_analysis.topology import Topology
from twisted_analysis.io.routing_table import load_routing_table
from twisted_analysis.io.schedule import load_schedule, save_schedule
from twisted_analysis.schedules.local_search import local_search_repair
from twisted_analysis.schedules.verify import schedule_makespan, verify_capacity

ROUTING = "fixtures/routing_table_8x4x4_twist.json"
SLICE = (8, 4, 4)
FOLDER = Path(__file__).parent
OUT = FOLDER / "05_local_search_results.json"

SEED_PATHS = [
    # path-relative to repo root or absolute
    Path("fixtures/schedule_8x4x4_loaded_orbit_greedy_full_lpt_tail_asc.json"),
    Path("fixtures/schedule_8x4x4_loaded_literal_greedy_lpt.json"),
    FOLDER / "02_best_random_shuffle_schedule.json",
    FOLDER / "03_best_cpsat_schedule.json",
    FOLDER / "04_best_lp_rounding_schedule.json",
]


def run():
    topology = Topology(slice=SLICE)
    table = load_routing_table(ROUTING)
    rows = []
    best_makespan = None
    best_seed_path = None
    for seed_path in SEED_PATHS:
        if not seed_path.exists():
            rows.append({"seed": str(seed_path), "status": "MISSING"}); continue
        seed = load_schedule(seed_path)
        m_in = schedule_makespan(seed)
        t0 = time.time()
        out = local_search_repair(topology, table, seed, max_iters=2000)
        dt = time.time() - t0
        m_out = schedule_makespan(out)
        v = verify_capacity(out)
        row = {
            "seed": str(seed_path),
            "makespan_in": m_in,
            "makespan_out": m_out,
            "delta": m_in - m_out,
            "violations": len(v),
            "runtime_s": round(dt, 1),
        }
        rows.append(row)
        out_path = FOLDER / f"05_localsearch_from_{seed_path.stem}.json"
        save_schedule(out, out_path)
        print(f"  {seed_path.name}: {m_in} -> {m_out} (Δ={m_in - m_out}), "
              f"viol={len(v)}, t={dt:.1f}s")
        if best_makespan is None or m_out < best_makespan:
            best_makespan = m_out
            best_seed_path = str(out_path)
    OUT.write_text(json.dumps({"rows": rows,
                               "best_makespan": best_makespan,
                               "best_schedule_file": best_seed_path},
                              indent=2))
    print(f"\nbest after local-search: makespan={best_makespan}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Run the probe**

```bash
.venv/bin/python -u eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/05_local_search_probe.py 2>&1 | tee eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/05_local_search_probe.log
```

Expected runtime: ~5-30 minutes total across all seeds.

- [ ] **Step 3: Promote to global best if better**

```bash
.venv/bin/python -c "
from pathlib import Path
import json, shutil
from twisted_analysis.io.schedule import load_schedule
from twisted_analysis.schedules.verify import schedule_makespan
folder = Path('eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4')
res = json.loads((folder / '05_local_search_results.json').read_text())
new = Path(res['best_schedule_file']) if res['best_schedule_file'] else None
gbest = folder / 'best_schedule.json'
m_new = schedule_makespan(load_schedule(new)) if new else 10**9
m_old = schedule_makespan(load_schedule(gbest)) if gbest.exists() else 10**9
print('localsearch best', m_new, 'vs current best', m_old)
if new and m_new < m_old:
    shutil.copy(new, gbest); print('promoted')
"
```

- [ ] **Step 4: Update RESULTS.md**

Add a "Phase 3: local-search repair" subsection with per-seed deltas.

- [ ] **Step 5: Commit**

```bash
git add eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/05_local_search_probe.py \
        eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/05_local_search_results.json \
        eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/05_local_search_probe.log \
        eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/05_localsearch_*.json \
        eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/best_schedule.json \
        eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/RESULTS.md
git commit -m "exploration[phase3]: local-search repair on every seed schedule

<best post-repair makespan, deltas per seed>"
```

---

### Task 10: Generate Pallas kernel for the best schedule

**Files:**
- Create: `eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/best_pallas_kernel.py` (generated by CLI)
- Modify: `pallas_kernel/README.md` (add a row for the new best, if better than baseline)

- [ ] **Step 1: Determine the best schedule's source scheduler**

```bash
.venv/bin/python -c "
from pathlib import Path
from twisted_analysis.io.schedule import load_schedule
from twisted_analysis.schedules.verify import schedule_makespan, verify_capacity
p = Path('eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/best_schedule.json')
sch = load_schedule(p)
print('makespan:', schedule_makespan(sch))
print('violations:', len(verify_capacity(sch)))
print('n_entries:', len(sch))
"
```

This is just verification; the source-scheduler identity is recorded in RESULTS.md.

- [ ] **Step 2: Generate kernel for the best schedule**

Reuse the existing `gen_orbit_greedy_kernel.py` CLI by pointing it at the best scheduler+params combination that produced `best_schedule.json`. (Per-step-barrier is only valid for orbit-aligned schedules; the CLI auto-refuses if the schedule is per-flow.) Example invocations depending on which probe won:

```bash
# If orbit_greedy_full[lpt] won (sim-makespan 84):
.venv/bin/python -m pallas_kernel.gen_orbit_greedy_kernel \
    --slice 8 4 4 \
    --routing-table fixtures/routing_table_8x4x4_twist.json \
    --scheduler orbit_greedy_full \
    --order lpt \
    --schedule-out eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/best_schedule.json \
    --out eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/best_pallas_kernel.py
```

For probes not exposed via the existing `--scheduler` dispatch (e.g. a winning shuffled-orbit ordering or a local-search-repaired schedule), a slightly different path is required: write the best schedule to `best_schedule.json`, then modify the CLI invocation to read from disk rather than re-run the scheduler. As a simpler alternative, add a `--schedule-in` flag to the CLI in a follow-up task. For now, regenerate using the winning algorithm name + params; if that's not in the dispatch, document it in RESULTS.md as "kernel generation deferred — schedule available at best_schedule.json."

For probes that produce per-flow schedules (`cpsat_literal`, `lp_rounding`, `local_search`), invoke without `--per-step-barrier`. Adjust the CLI invocation to match the winning scheduler.

- [ ] **Step 3: Sanity-check the generated kernel exists**

```bash
test -f eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/best_pallas_kernel.py \
    && wc -l eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/best_pallas_kernel.py
```

Expected: kernel file present, hundreds of lines (one per orbit fire / per flow).

- [ ] **Step 4: Update `pallas_kernel/README.md`**

In the routing × scheduler performance matrix (introduced in commit `e25bc48`), append a row for the new best with: routing=loaded, scheduler=<winner>, sim-makespan=<best>, gap-to-LB=<best - 75>, location of kernel file. If no probe beat 84, note that the existing best stands.

- [ ] **Step 5: Commit**

```bash
git add eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/best_pallas_kernel.py \
        eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/best_schedule.json \
        pallas_kernel/README.md
git commit -m "exploration: emit Pallas kernel for best loaded-8x4x4 schedule

Best sim-makespan: <X> (scheduler=<Y>, gap-to-LB=<Z>). Kernel ready for TPU
v5e measurement; comparison with P2P (134541 gbps) pending hardware run."
```

---

### Task 11: Final summary + cross-link

**Files:**
- Modify: `eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/README.md` (fill Conclusions)
- Modify: `eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/RESULTS.md` (add final summary header)
- Modify: `README.md` (root — add a brief link to this exploration)
- Modify: `docs/orbit_greedy_optimality.md` (cross-reference if a probe revealed something new about LB tightness)

- [ ] **Step 1: Fill in `README.md` Conclusions section**

Update the "Conclusions" section in `eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/README.md` to summarize:

1. **Best makespan found**, from which probe, with how much improvement over the 85 baseline.
2. **Whether sim-makespan ≤ 83 was achieved** (target to beat P2P by linear scaling).
3. **What each phase taught us**: ordering matters (lpt > lpt_tail_asc by 1 step); CP-SAT (in)tractable at N=128 (status per t_upper); LP-rounding (in)tractable; local-search yields negligible / measurable improvement.
4. **Open questions**: is LB=75 achievable? (status: feasible / infeasible by CP-SAT, or still open).
5. **Hardware-vs-simulator caveat reminder**: even with sim-makespan ≤ 83, the Pallas kernel may not beat P2P because the simulator omits per-step barriers, HBM contention, and VC arbitration. The next-step is a TPU v5e measurement.

- [ ] **Step 2: Finalize `RESULTS.md`**

Add a top-level "Summary" section above the per-phase tables: scheduler that won, makespan, runtime, sim-vs-P2P estimate.

- [ ] **Step 3: Add link in root `README.md`**

Add a brief entry near the scheduler matrix (the section added in commit `e25bc48`):

```markdown
For an empirical search over scheduling algorithms specifically targeting
the loaded 8×4×4 routing (the routing the deployed Pallas kernel uses),
see [eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/](eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/).
```

- [ ] **Step 4: Run full test suite to catch regressions**

```bash
.venv/bin/python -m pytest -x -q
```

Expected: all PASS (or only pre-existing slow-marked skips).

- [ ] **Step 5: Final commit**

```bash
git add eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/README.md \
        eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/RESULTS.md \
        README.md \
        docs/orbit_greedy_optimality.md 2>/dev/null
git commit -m "exploration: final summary for loaded-8x4x4 scheduler search

Summary: <one-line outcome>. Cross-linked from root README; conclusions
recorded in eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/README.md."
```

---

## End-of-plan notes

**Order of attack (recap).**
1. (cheap) ordering sweep — already knows makespan=84 is reachable.
2. (cheap) random orbit shuffle — see if shuffling beats lpt.
3. (medium) CP-SAT probe — tight feasibility check, scales beyond CBC.
4. (medium) LP-rounding — polynomial alternative if CP-SAT times out.
5. (cheap) local-search repair on every best-so-far.

**Stop-early triggers.**
- If Phase 1b finds a seed with makespan ≤ 80, skip Phase 2a/2b and go straight to Phase 3.
- If Phase 2a returns INFEASIBLE at some t_upper, all later probes know LB+k>=t_upper is the floor.

**What this plan does NOT include.**
- Production-quality CLI surface for `lp_rounding`'s seed parameter.
- TPU v5e hardware measurement of the best kernel — that's a follow-up (separate session, requires v5e access).
- Generalization to other routings (e.g., 2×4×4 ILP). The schedulers added are general-purpose, so a follow-up plan can run them on other cells with minimal effort.

**Self-test before declaring victory.**
After Task 11's `pytest -x -q`, confirm:
1. All new tests pass.
2. The kernel generator's existing test (`test_gen_orbit_greedy_kernel_pipeline.py`) still passes — the dispatch additions in `schedule.py` must not break it.
3. `eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/RESULTS.md` accurately reflects which probe won.
