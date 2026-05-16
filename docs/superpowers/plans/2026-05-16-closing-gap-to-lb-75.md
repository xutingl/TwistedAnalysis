# Closing the makespan gap to LB=75 on loaded 8×4×4 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Push the best-known schedule for the loaded 8×4×4 twisted-torus routing from sim-makespan 80 toward the physical-edge lower bound LB=75 by combining (A) CP-SAT warm-started from the current best with extended compute budget and (B) Large-Neighborhood-Search (LNS) using CP-SAT as the subproblem solver.

**Architecture:** Refactor `cpsat_literal` into a single underlying solve primitive that accepts both **warm-start hints** (initial variable values via `model.AddHint`) and **fixed assignments** (flows whose round is pinned by the caller). The standalone scheduler uses the first; LNS uses both — it pins all flows not in the destroy set and hints from the current incumbent. Add an LNS module with three destroy strategies (time-window, random-subset, makespan-flows) and an improve-only acceptance rule. Run two probes back-to-back over multiple days: warm-started CP-SAT at decreasing `t_upper` and LNS with rotating destroy strategies. If either yields makespan < 80, promote to fixtures + `cns_schedules` and regenerate the Pallas kernel.

**Tech Stack:** Python 3, OR-Tools CP-SAT (`ortools.sat.python.cp_model`), existing `twisted_analysis` package (Topology, routing tables, schedule I/O, verification), pytest. Probes run with `.venv/bin/python -u`.

---

## File Structure

**Modify:**
- `twisted_analysis/schedules/cpsat_literal.py` — Add `warm_start_schedule` and `fixed_assignments` kwargs; restructure variable creation to handle pinned flows.
- `twisted_analysis/io/schedule.py` — Plumb `warm_start_schedule` through `schedule_from_cpsat_literal`; add `schedule_from_lns_cpsat`; register `lns_cpsat` in `_SCHEDULER_DISPATCH`; update `schedule_from_algorithm` docstring.
- `tests/test_cpsat_literal.py` — Add tests for warm-start hint acceptance and fixed-assignment behavior.
- `README.md` (repo root) — Add row for `lns_cpsat` in the scheduler matrix and link the new exploration folder.

**Create:**
- `twisted_analysis/schedules/lns_cpsat.py` — LNS driver: destroy strategies + acceptance loop, calls into refactored `cpsat_literal`.
- `tests/test_lns_cpsat.py` — TDD: validate each destroy strategy + improve-only acceptance on small topology.
- `eval/explorations/2026-05-16-closing-gap-to-lb-75/README.md` — Problem, goal, planned probes.
- `eval/explorations/2026-05-16-closing-gap-to-lb-75/RESULTS.md` — Rolling per-probe table.
- `eval/explorations/2026-05-16-closing-gap-to-lb-75/01_cpsat_warm_start_probe.py` — Warm-start CP-SAT at `t_upper ∈ {79, 78, 77, 76}`, 4h each.
- `eval/explorations/2026-05-16-closing-gap-to-lb-75/02_lns_probe.py` — LNS driver, rotates strategies, logs per-iteration improvements.

**Generated outputs (not committed until they exist):**
- `eval/explorations/2026-05-16-closing-gap-to-lb-75/01_cpsat_warm_start_results.json`
- `eval/explorations/2026-05-16-closing-gap-to-lb-75/01_best_warm_start_schedule.json` (only if < 80)
- `eval/explorations/2026-05-16-closing-gap-to-lb-75/02_lns_results.json`
- `eval/explorations/2026-05-16-closing-gap-to-lb-75/02_best_lns_schedule.json` (only if < 80)
- `eval/explorations/2026-05-16-closing-gap-to-lb-75/best_schedule.json` (overall winner across both probes)
- `eval/explorations/2026-05-16-closing-gap-to-lb-75/best_pallas_kernel.py` (regenerated from overall winner if < 80)
- `fixtures/schedule_8x4x4_loaded_<scheduler>.json` + `fixtures/cns_schedules/schedule_<name>_4x4x8_twisted.json` (conditional promotion).

---

### Task 1: Refactor `cpsat_literal` to support warm-start hints and fixed assignments

**Files:**
- Modify: `twisted_analysis/schedules/cpsat_literal.py`
- Test: `tests/test_cpsat_literal.py`

The current `cpsat_literal(topology, table, *, t_upper, time_limit_s, solver_msg, n_workers, minimize)` builds variables `y[f, s]` for every flow/start pair, then enforces ExactlyOne per flow and AtMostOne per (edge, tau). We add two optional kwargs:

- `warm_start_schedule: list[dict] | None` — a previously-computed schedule (same format as `save_schedule` output). For each entry `(src, dst, round)` we call `model.AddHint(y[(f_idx, round)], 1)`, skipping any entry whose `round > t_upper - L_f` (variable doesn't exist at this `t_upper`).
- `fixed_assignments: dict[tuple[int, int], int] | None` — `(src, dst) -> required_round`. For each fixed flow we create only the single variable at the required round and pin it to True; we skip the ExactlyOne (it's a unit constraint). Edge-capacity constraints still see the variable via `(f_idx, s) in y`.

- [ ] **Step 1: Add the failing test for warm-start acceptance**

Append to `tests/test_cpsat_literal.py`:

```python
def test_cpsat_literal_warm_start_accepts_seed():
    """When warm-started with an optimal seed at the same t_upper, CP-SAT
    must return a feasible schedule (the hint should be respected when
    feasible, and the search at most matches the hint)."""
    t, table = _table_from_ilp_router((2, 4))
    lb = _physical_edge_lb(table, t.n_nodes)
    seed = cpsat_literal(t, table, t_upper=lb, time_limit_s=60)
    # Warm-start the next solve from the seed at the same t_upper.
    sch = cpsat_literal(t, table, t_upper=lb, time_limit_s=60,
                        warm_start_schedule=seed)
    assert verify_capacity(sch) == []
    assert schedule_makespan(sch) <= lb


def test_cpsat_literal_fixed_assignments_pins_flows():
    """fixed_assignments must force named flows to their required round
    (and the rest of the schedule must remain feasible)."""
    t, table = _table_from_ilp_router((2, 4))
    lb = _physical_edge_lb(table, t.n_nodes)
    seed = cpsat_literal(t, table, t_upper=lb, time_limit_s=60)
    # Pin every flow to its seed round; the model should accept and return it.
    fixed = {(e["src"], e["dst"]): e["round"] for e in seed}
    sch = cpsat_literal(t, table, t_upper=lb, time_limit_s=60,
                        fixed_assignments=fixed)
    by_key = {(e["src"], e["dst"]): e["round"] for e in sch}
    for k, r in fixed.items():
        assert by_key[k] == r, f"flow {k}: expected round {r}, got {by_key[k]}"


def test_cpsat_literal_fixed_assignments_infeasible_combination_raises():
    """If pinned flows conflict (two flows on same edge at same time), raise."""
    t, table = _table_from_ilp_router((2, 4))
    lb = _physical_edge_lb(table, t.n_nodes)
    # Find any two flows that share at least one physical edge at any hop.
    from collections import defaultdict
    edge_hops: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for s in range(t.n_nodes):
        for d in range(t.n_nodes):
            if s == d:
                continue
            path = table[s][d]
            for h in range(len(path) - 1):
                edge_hops[(path[h], path[h + 1])].append((s, d, h))
    # Pick a shared edge with at least 2 demands.
    shared = next(((e, demands) for e, demands in edge_hops.items()
                   if len(demands) >= 2), None)
    if shared is None:
        import pytest as _p
        _p.skip("no shared edge on this topology — skip conflict test")
    _edge, demands = shared
    (s1, d1, h1), (s2, d2, h2), *_ = demands
    # Pin both at the same edge-time: round_i = tau - h_i for tau = 0.
    tau = max(h1, h2)
    fixed = {(s1, d1): tau - h1, (s2, d2): tau - h2}
    import pytest as _p
    with _p.raises(RuntimeError, match="infeasible|no solution"):
        cpsat_literal(t, table, t_upper=lb, time_limit_s=30,
                      fixed_assignments=fixed)
```

- [ ] **Step 2: Run tests to verify they fail (kwargs not yet implemented)**

Run: `cd /home/xutingl/collective_comm/TwistedAnalysis && .venv/bin/python -m pytest tests/test_cpsat_literal.py -v -x`

Expected: 3 new tests FAIL with `TypeError: cpsat_literal() got an unexpected keyword argument 'warm_start_schedule'` (and same for `fixed_assignments`). The pre-existing 2 tests should still PASS.

- [ ] **Step 3: Update `cpsat_literal` signature and variable construction**

Replace the function in `twisted_analysis/schedules/cpsat_literal.py` (preserving the module docstring, imports, and `_flow_set`) with:

```python
def cpsat_literal(
    topology,
    table: list[list[list[int]]],
    *,
    t_upper: int,
    time_limit_s: int = 600,
    solver_msg: bool = False,
    n_workers: int = 8,
    minimize: bool = True,
    warm_start_schedule: list[dict] | None = None,
    fixed_assignments: dict[tuple[int, int], int] | None = None,
) -> list[dict]:
    """Solve / feasibility-probe the literal scheduling problem with CP-SAT.

    Optional kwargs:
      warm_start_schedule: a previously-computed schedule (list of
        {round, src, dst, path} dicts). Each entry becomes an `AddHint` on
        the corresponding y-variable when that variable exists at this
        `t_upper`. Entries whose round exceeds `t_upper - L_f` are silently
        skipped.
      fixed_assignments: dict (src, dst) -> required round. For each fixed
        flow only the variable at the required round is created (and
        pinned to True). Other y-variables for that flow do not exist;
        edge-capacity constraints still account for the fixed flow.

    Raises ImportError if ortools is not installed.
    Raises RuntimeError if infeasible at the given `t_upper` (including
    when fixed_assignments make the model infeasible).
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise ImportError(
            "cpsat_literal requires ortools. Install: `uv pip install ortools`."
        ) from exc

    n = topology.n_nodes
    flows = _flow_set(table, n)

    # Build a quick lookup (src, dst) -> f_idx for fixed_assignments and warm_start.
    flow_key_to_idx = {(s, d): i for i, (s, d, _) in enumerate(flows)}

    fixed_idx: dict[int, int] = {}
    if fixed_assignments:
        for (src, dst), required in fixed_assignments.items():
            if (src, dst) not in flow_key_to_idx:
                raise ValueError(
                    f"fixed_assignments: flow ({src},{dst}) not in flow set"
                )
            fixed_idx[flow_key_to_idx[(src, dst)]] = int(required)

    model = cp_model.CpModel()
    y: dict[tuple[int, int], cp_model.IntVar] = {}

    for f_idx, (_s, _d, path) in enumerate(flows):
        L = len(path) - 1
        if f_idx in fixed_idx:
            required = fixed_idx[f_idx]
            if not (0 <= required <= t_upper - L):
                raise RuntimeError(
                    f"fixed_assignments: flow {f_idx} required round {required} "
                    f"out of feasible range [0, {t_upper - L}] at t_upper={t_upper}"
                )
            v = model.NewBoolVar(f"y_{f_idx}_{required}_fixed")
            model.Add(v == 1)
            y[(f_idx, required)] = v
            continue
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
            if f_idx in fixed_idx:
                # Single var case: makespan must accommodate this single assignment.
                required = fixed_idx[f_idx]
                model.Add(M >= required + L).OnlyEnforceIf(y[(f_idx, required)])
            else:
                for s in range(0, t_upper - L + 1):
                    model.Add(M >= s + L).OnlyEnforceIf(y[(f_idx, s)])
        model.Minimize(M)

    # Warm-start hints from a prior schedule (after all variables exist).
    if warm_start_schedule is not None:
        for e in warm_start_schedule:
            key = (e["src"], e["dst"])
            f_idx = flow_key_to_idx.get(key)
            if f_idx is None:
                continue  # foreign flow — ignore
            s = int(e["round"])
            if (f_idx, s) in y:
                model.AddHint(y[(f_idx, s)], 1)

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
        if f_idx in fixed_idx:
            chosen = fixed_idx[f_idx]
        else:
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

- [ ] **Step 4: Run all tests to verify they pass**

Run: `cd /home/xutingl/collective_comm/TwistedAnalysis && .venv/bin/python -m pytest tests/test_cpsat_literal.py -v`

Expected: all 5 tests PASS (2 pre-existing + 3 new). If the `infeasible_combination` test errors with "no shared edge", that's a `skip`, not a failure — still counts as PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
git add twisted_analysis/schedules/cpsat_literal.py tests/test_cpsat_literal.py
git commit -m "cpsat_literal: add warm_start_schedule and fixed_assignments kwargs"
```

---

### Task 2: Plumb warm-start through `schedule_from_cpsat_literal`

**Files:**
- Modify: `twisted_analysis/io/schedule.py:184-200` (`schedule_from_cpsat_literal`)

The adapter needs to accept `warm_start_schedule` (and pass it through). We do **not** expose `fixed_assignments` through the adapter — that's an LNS-internal mechanism, used directly via `cpsat_literal`.

- [ ] **Step 1: Update the adapter signature**

In `twisted_analysis/io/schedule.py`, replace the `schedule_from_cpsat_literal` function with:

```python
def schedule_from_cpsat_literal(
    topology: Topology,
    table: list[list[list[int]]],
    *,
    t_upper: int,
    time_limit_s: int = 600,
    n_workers: int = 8,
    warm_start_schedule: list[dict] | None = None,
) -> list[dict]:
    """Adapter: cpsat_literal -> schedule entries.

    `warm_start_schedule`: when provided, prior schedule entries are fed
    to CP-SAT as variable hints (see cpsat_literal docstring).
    """
    from twisted_analysis.io.routing_table import validate_routing_table_shape
    from twisted_analysis.schedules.cpsat_literal import cpsat_literal

    validate_routing_table_shape(table, topology.n_nodes)
    return cpsat_literal(
        topology, table, t_upper=t_upper,
        time_limit_s=time_limit_s, n_workers=n_workers,
        warm_start_schedule=warm_start_schedule,
    )
```

- [ ] **Step 2: Run existing schedule-dispatch tests**

Run: `cd /home/xutingl/collective_comm/TwistedAnalysis && .venv/bin/python -m pytest tests/test_cpsat_literal.py -v`

Expected: all 5 tests PASS (no regressions). The adapter change is additive — existing callers don't pass `warm_start_schedule`, so behavior is unchanged.

- [ ] **Step 3: Commit**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
git add twisted_analysis/io/schedule.py
git commit -m "schedule_from_cpsat_literal: thread warm_start_schedule kwarg"
```

---

### Task 3: Create `lns_cpsat.py` with three destroy strategies and improve-only acceptance

**Files:**
- Create: `twisted_analysis/schedules/lns_cpsat.py`
- Test: `tests/test_lns_cpsat.py`

LNS API:

```python
def lns_cpsat_repair(
    topology,
    table,
    seed_schedule: list[dict],
    *,
    n_iters: int = 100,
    per_subproblem_budget_s: int = 300,
    destroy_strategies: tuple[str, ...] = ("time_window", "random_subset", "makespan_flows"),
    destroy_size_frac: float = 0.05,
    rng_seed: int = 0,
    n_workers: int = 8,
    log_fn=None,
) -> list[dict]:
    ...
```

At each iteration: rotate to the next destroy strategy; compute the destroy set `D` of flow keys; build `fixed_assignments` from non-D flows in the current incumbent; call `cpsat_literal(..., t_upper=M-1, fixed_assignments=..., warm_start_schedule=current)`; on FEASIBLE with new makespan ≤ M, replace incumbent; on TIMEOUT or INFEASIBLE, keep incumbent. After `n_iters` iterations or when the makespan equals the physical-edge LB, return.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_lns_cpsat.py`:

```python
"""LNS CP-SAT repair: tests on a small cell."""
from __future__ import annotations
import tempfile, os
from collections import Counter

import pytest

pytest.importorskip("ortools")

from twisted_analysis.topology import Topology, ILPRouter
from twisted_analysis.io.routing_table import save_routing_table, load_routing_table
from twisted_analysis.schedules.cpsat_literal import cpsat_literal
from twisted_analysis.schedules.lns_cpsat import lns_cpsat_repair
from twisted_analysis.schedules.verify import verify_capacity, schedule_makespan


def _table_from_ilp_router(slice_):
    t = Topology(slice=slice_)
    r = ILPRouter(t)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name
    try:
        save_routing_table(t, r, tmp_path)
        table = load_routing_table(tmp_path)
    finally:
        os.unlink(tmp_path)
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


def test_lns_returns_feasible_schedule_for_each_strategy():
    """Every destroy strategy must produce a capacity-feasible schedule."""
    t, table = _table_from_ilp_router((2, 4))
    lb = _physical_edge_lb(table, t.n_nodes)
    seed = cpsat_literal(t, table, t_upper=lb + 2, time_limit_s=60)
    for strat in ("time_window", "random_subset", "makespan_flows"):
        sch = lns_cpsat_repair(t, table, seed, n_iters=3,
                               per_subproblem_budget_s=30,
                               destroy_strategies=(strat,))
        assert verify_capacity(sch) == [], f"{strat}: violations"
        assert schedule_makespan(sch) <= schedule_makespan(seed), \
            f"{strat}: makespan increased ({schedule_makespan(sch)} > {schedule_makespan(seed)})"


def test_lns_improve_only_never_increases_makespan():
    """Across many iterations, the makespan must be monotone non-increasing."""
    t, table = _table_from_ilp_router((2, 4))
    lb = _physical_edge_lb(table, t.n_nodes)
    seed = cpsat_literal(t, table, t_upper=lb + 3, time_limit_s=60)
    history: list[int] = []
    def log(it, info):
        history.append(info["current_makespan"])
    sch = lns_cpsat_repair(t, table, seed, n_iters=10,
                           per_subproblem_budget_s=20,
                           rng_seed=42, log_fn=log)
    assert verify_capacity(sch) == []
    for i in range(1, len(history)):
        assert history[i] <= history[i - 1], \
            f"makespan increased at iter {i}: {history[i-1]} -> {history[i]}"


def test_lns_at_lb_stops_immediately():
    """When seeded with an LB-tight schedule, LNS must not increase makespan."""
    t, table = _table_from_ilp_router((2, 4))
    lb = _physical_edge_lb(table, t.n_nodes)
    seed = cpsat_literal(t, table, t_upper=lb, time_limit_s=60)
    assert schedule_makespan(seed) <= lb
    sch = lns_cpsat_repair(t, table, seed, n_iters=5,
                           per_subproblem_budget_s=10)
    assert verify_capacity(sch) == []
    assert schedule_makespan(sch) <= lb
```

- [ ] **Step 2: Run tests to verify they fail (module doesn't exist yet)**

Run: `cd /home/xutingl/collective_comm/TwistedAnalysis && .venv/bin/python -m pytest tests/test_lns_cpsat.py -v -x`

Expected: collection errors with `ModuleNotFoundError: No module named 'twisted_analysis.schedules.lns_cpsat'`.

- [ ] **Step 3: Implement the LNS module**

Create `twisted_analysis/schedules/lns_cpsat.py`:

```python
"""Large-Neighborhood Search (LNS) repair on top of cpsat_literal.

Iteratively destroys a subset D of flow assignments from a seed schedule,
fixes the remaining flows in place, and asks CP-SAT to re-optimize the
destroy set with t_upper = current_makespan - 1. On FEASIBLE we accept
the strictly-better incumbent; on TIMEOUT/INFEASIBLE we keep the old one
and try a different destroy on the next iteration.

Destroy strategies:
  - "time_window":   pick all flows in the trailing `k` rounds of the
                     current incumbent.
  - "random_subset": pick K random flow keys (uniform).
  - "makespan_flows": find the bottleneck physical edge in the trailing
                     `k` rounds (most heavily used) and pick all flows
                     whose path contains that edge.

`destroy_size_frac` controls how big the destroy set is (fraction of N(N-1)).
"""
from __future__ import annotations
import random
from collections import Counter, defaultdict

from twisted_analysis.schedules.cpsat_literal import cpsat_literal
from twisted_analysis.schedules.verify import schedule_makespan


def _flows_by_round(schedule: list[dict]) -> dict[int, list[tuple[int, int]]]:
    out: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for e in schedule:
        out[int(e["round"])].append((int(e["src"]), int(e["dst"])))
    return out


def _physical_edge_lb(table, n):
    c: Counter = Counter()
    for s in range(n):
        for d in range(n):
            if s == d:
                continue
            path = table[s][d]
            for h in range(len(path) - 1):
                c[(path[h], path[h + 1])] += 1
    return max(c.values()) if c else 0


def _makespan_defining_flows(schedule):
    """Return the flow keys whose (round + L) equals the current makespan.

    These flows MUST be in the destroy set: otherwise the LNS subproblem
    (which sets t_upper = M-1 and pins all non-destroyed flows) is
    immediately infeasible because the makespan-defining flow's pinned
    round + L exceeds the new t_upper.
    """
    M = max(int(e["round"]) + (len(e["path"]) - 1) for e in schedule)
    return {(int(e["src"]), int(e["dst"])) for e in schedule
            if int(e["round"]) + (len(e["path"]) - 1) >= M}


def _destroy_time_window(schedule, k_rounds):
    M = max(int(e["round"]) for e in schedule)
    cutoff = max(0, M - k_rounds + 1)
    return {(int(e["src"]), int(e["dst"])) for e in schedule
            if int(e["round"]) >= cutoff}


def _destroy_random_subset(schedule, K, rng):
    keys = [(int(e["src"]), int(e["dst"])) for e in schedule]
    rng.shuffle(keys)
    return set(keys[:K])


def _destroy_makespan_flows(schedule, k_rounds):
    """Find the bottleneck edge in the trailing window and destroy every
    flow that traverses it."""
    M = max(int(e["round"]) for e in schedule)
    cutoff = max(0, M - k_rounds + 1)
    # Count (edge, _) loads in the trailing window: each flow contributes
    # one slot per physical edge it uses, but only count flows whose
    # assigned round falls in the window. (This is a heuristic — we want
    # to pick a *contention-relevant* edge, not just the heaviest overall.)
    edge_load: Counter = Counter()
    flows_on_edge: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for e in schedule:
        if int(e["round"]) < cutoff:
            continue
        path = e["path"]
        for h in range(len(path) - 1):
            edge_load[(path[h], path[h + 1])] += 1
            flows_on_edge[(path[h], path[h + 1])].append(
                (int(e["src"]), int(e["dst"]))
            )
    if not edge_load:
        # Fallback: window empty (degenerate seed) — just use all flows.
        return {(int(e["src"]), int(e["dst"])) for e in schedule}
    bottleneck = edge_load.most_common(1)[0][0]
    return set(flows_on_edge[bottleneck])


def lns_cpsat_repair(
    topology,
    table: list[list[list[int]]],
    seed_schedule: list[dict],
    *,
    n_iters: int = 100,
    per_subproblem_budget_s: int = 300,
    destroy_strategies: tuple[str, ...] = (
        "time_window", "random_subset", "makespan_flows",
    ),
    destroy_size_frac: float = 0.05,
    rng_seed: int = 0,
    n_workers: int = 8,
    log_fn=None,
) -> list[dict]:
    n = topology.n_nodes
    lb = _physical_edge_lb(table, n)
    rng = random.Random(rng_seed)

    incumbent: dict[tuple[int, int], int] = {
        (int(e["src"]), int(e["dst"])): int(e["round"]) for e in seed_schedule
    }
    # Verifier-style: makespan = max(round + L) per flow.
    incumbent_makespan = max(
        r + (len(table[s][d]) - 1) for (s, d), r in incumbent.items()
    )

    def _schedule_from_incumbent() -> list[dict]:
        out = []
        for (s, d), r in incumbent.items():
            out.append({
                "round": r, "src": s, "dst": d,
                "path": list(table[s][d]),
            })
        out.sort(key=lambda e: (e["round"], e["src"]))
        return out

    K = max(1, int(destroy_size_frac * (n * (n - 1))))
    k_rounds = max(1, int(0.10 * max(1, incumbent_makespan)))

    for it in range(n_iters):
        if incumbent_makespan <= lb:
            break  # at LB — cannot improve further
        strat = destroy_strategies[it % len(destroy_strategies)]
        cur_sched = _schedule_from_incumbent()
        if strat == "time_window":
            D = _destroy_time_window(cur_sched, k_rounds)
        elif strat == "random_subset":
            D = _destroy_random_subset(cur_sched, K, rng)
        elif strat == "makespan_flows":
            D = _destroy_makespan_flows(cur_sched, k_rounds)
        else:
            raise ValueError(f"unknown destroy strategy: {strat!r}")
        # Always union the makespan-defining flow(s) into D, otherwise the
        # subproblem at t_upper = M-1 is trivially infeasible (those pinned
        # flows already violate the new bound).
        D = D | _makespan_defining_flows(cur_sched)
        if not D:
            continue
        fixed = {k: r for k, r in incumbent.items() if k not in D}
        if log_fn is not None:
            log_fn(it, {
                "strategy": strat, "destroy_size": len(D),
                "fixed_size": len(fixed),
                "current_makespan": incumbent_makespan,
                "target_t_upper": incumbent_makespan - 1,
            })
        try:
            new = cpsat_literal(
                topology, table,
                t_upper=incumbent_makespan - 1,
                time_limit_s=per_subproblem_budget_s,
                n_workers=n_workers,
                fixed_assignments=fixed,
                warm_start_schedule=cur_sched,
            )
        except RuntimeError:
            continue
        new_makespan = schedule_makespan(new)
        if new_makespan < incumbent_makespan:
            incumbent = {(int(e["src"]), int(e["dst"])): int(e["round"]) for e in new}
            incumbent_makespan = new_makespan

    return _schedule_from_incumbent()
```

- [ ] **Step 4: Run LNS tests to verify they pass**

Run: `cd /home/xutingl/collective_comm/TwistedAnalysis && .venv/bin/python -m pytest tests/test_lns_cpsat.py -v`

Expected: all 3 tests PASS. If `test_lns_improve_only_never_increases_makespan` fails because no iteration ever fired (all destroys hit the same flow set), the test still asserts monotonicity, which holds vacuously — so it should pass.

- [ ] **Step 5: Commit**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
git add twisted_analysis/schedules/lns_cpsat.py tests/test_lns_cpsat.py
git commit -m "schedules: add lns_cpsat (LNS repair using cpsat_literal subsolver)"
```

---

### Task 4: Register `lns_cpsat` adapter in `io/schedule.py`

**Files:**
- Modify: `twisted_analysis/io/schedule.py` (add `schedule_from_lns_cpsat`, register in `_SCHEDULER_DISPATCH`, extend `schedule_from_algorithm` docstring)

- [ ] **Step 1: Add the adapter and register it**

In `twisted_analysis/io/schedule.py`, after the `schedule_from_local_search` function and before the `_SCHEDULER_DISPATCH = {...}` block, add:

```python
def schedule_from_lns_cpsat(
    topology: Topology,
    table: list[list[list[int]]],
    *,
    seed_schedule: list[dict],
    n_iters: int = 100,
    per_subproblem_budget_s: int = 300,
    destroy_size_frac: float = 0.05,
    rng_seed: int = 0,
    n_workers: int = 8,
) -> list[dict]:
    """Adapter: lns_cpsat_repair on a seed schedule.

    Iteratively destroys part of `seed_schedule` and re-solves the
    subproblem with CP-SAT, accepting strictly-better incumbents.
    """
    from twisted_analysis.io.routing_table import validate_routing_table_shape
    from twisted_analysis.schedules.lns_cpsat import lns_cpsat_repair

    validate_routing_table_shape(table, topology.n_nodes)
    return lns_cpsat_repair(
        topology, table, seed_schedule,
        n_iters=n_iters,
        per_subproblem_budget_s=per_subproblem_budget_s,
        destroy_size_frac=destroy_size_frac,
        rng_seed=rng_seed,
        n_workers=n_workers,
    )
```

Then update the `_SCHEDULER_DISPATCH` dict (currently ending at line 243) to add the new entry:

```python
_SCHEDULER_DISPATCH = {
    "orbit_greedy": schedule_from_orbit_greedy,
    "orbit_greedy_full": schedule_from_orbit_greedy_full,
    "literal_greedy": schedule_from_literal_greedy,
    "ilp_literal": schedule_from_ilp_literal,
    "cpsat_literal": schedule_from_cpsat_literal,
    "lp_rounding": schedule_from_lp_rounding,
    "local_search": schedule_from_local_search,
    "lns_cpsat": schedule_from_lns_cpsat,
}
```

And in the `schedule_from_algorithm` docstring (currently at lines 254-272), append after the `"local_search"` bullet:

```
      - "lns_cpsat":         Large-Neighborhood-Search repair using CP-SAT as the
        subproblem solver. Each iteration destroys a subset of the seed schedule
        (time-window, random subset, or makespan-bottleneck flows), fixes the
        rest in place, and asks CP-SAT to re-optimize. Accepts strictly-better
        incumbents. Requires `seed_schedule` kwarg; tunables `n_iters` (default
        100), `per_subproblem_budget_s` (default 300), `destroy_size_frac`
        (default 0.05), `rng_seed` (default 0), `n_workers` (default 8).
```

- [ ] **Step 2: Add a dispatch round-trip test**

Append to `tests/test_lns_cpsat.py`:

```python
def test_lns_cpsat_via_dispatch():
    """`schedule_from_algorithm('lns_cpsat', ...)` must dispatch correctly."""
    from twisted_analysis.io.schedule import schedule_from_algorithm
    t, table = _table_from_ilp_router((2, 4))
    lb = _physical_edge_lb(table, t.n_nodes)
    seed = cpsat_literal(t, table, t_upper=lb + 2, time_limit_s=30)
    sch = schedule_from_algorithm(
        "lns_cpsat", t, table,
        seed_schedule=seed, n_iters=2, per_subproblem_budget_s=15,
    )
    assert verify_capacity(sch) == []
    assert schedule_makespan(sch) <= schedule_makespan(seed)
```

- [ ] **Step 3: Run the dispatch test**

Run: `cd /home/xutingl/collective_comm/TwistedAnalysis && .venv/bin/python -m pytest tests/test_lns_cpsat.py::test_lns_cpsat_via_dispatch -v`

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
git add twisted_analysis/io/schedule.py tests/test_lns_cpsat.py
git commit -m "io/schedule: register lns_cpsat in dispatch"
```

---

### Task 5: Create exploration folder + README + RESULTS scaffold

**Files:**
- Create: `eval/explorations/2026-05-16-closing-gap-to-lb-75/README.md`
- Create: `eval/explorations/2026-05-16-closing-gap-to-lb-75/RESULTS.md`

- [ ] **Step 1: Create the folder**

```bash
mkdir -p /home/xutingl/collective_comm/TwistedAnalysis/eval/explorations/2026-05-16-closing-gap-to-lb-75
```

- [ ] **Step 2: Write README.md**

Create `eval/explorations/2026-05-16-closing-gap-to-lb-75/README.md` with:

```markdown
# Closing the makespan gap to LB=75 on loaded 8×4×4

## Problem

The 2026-05-15 exploration brought the best-known schedule on the loaded
8×4×4 routing from `orbit_greedy_full[lpt_tail_asc]` makespan 85 down to
`cpsat_literal` makespan **80** at `t_upper=80`, projected to ~141055 gbps
vs P2P's measured 134541 gbps (+4.8%). The physical-edge lower bound is
**LB=75**. CP-SAT timed out at `t_upper ∈ {76, 78}` with no incumbent —
evidence the search is hard at those bounds, not proof of infeasibility.

This exploration tries to close the 80 → 75 gap with two complementary
methods.

## Goal

Find a capacity-feasible schedule for the loaded 8×4×4 routing with
sim-makespan strictly below 80 (ideally 75 = LB). If 75 is unreachable,
quantify how close we can get.

## Approach (2 probes)

1. [01_cpsat_warm_start_probe.py](01_cpsat_warm_start_probe.py) —
   CP-SAT at `t_upper ∈ {79, 78, 77, 76}` with **4h per probe**,
   warm-started from `fixtures/schedule_8x4x4_loaded_cpsat_literal.json`.
   Hypothesis: the 80 → 76 search space was hard cold because CP-SAT
   restarted from scratch at each `t_upper`. Warm-starting from a
   feasible incumbent at 80 (most variables hinted) should let CP-SAT
   focus on the few rounds at the makespan tail.

2. [02_lns_probe.py](02_lns_probe.py) — Large-Neighborhood Search:
   destroy 5–10% of the seed schedule (time-window / random-subset /
   makespan-bottleneck flows) and ask CP-SAT to re-optimize the
   subproblem with the rest pinned. Each subproblem is much smaller
   than the full N(N-1)=16256 model. Hypothesis: the previous-exploration
   local-search dead-end (every seed at a local optimum under shift-earlier
   moves) is escapable by simultaneously re-assigning a connected subset.

## Compute budget

Multi-day. Probe 1: 4 × 4h ≈ 16h sequentially. Probe 2: 100 iterations ×
~5 min per subproblem ≈ 8h. Both run sequentially on the same machine to
avoid CPU contention (CP-SAT uses 8 workers each).

## Outcome

(Filled in after probes complete.)
```

- [ ] **Step 3: Write RESULTS.md (scaffold)**

Create `eval/explorations/2026-05-16-closing-gap-to-lb-75/RESULTS.md` with:

```markdown
# Results: Closing the gap to LB=75

**Baseline (incoming):** `cpsat_literal` makespan **80** at `t_upper=80`,
30-min budget, no warm-start. Schedule:
`fixtures/schedule_8x4x4_loaded_cpsat_literal.json`.

**LB:** 75 (max physical-edge load).

## Probe 1: CP-SAT warm-started from makespan-80

Schedule: `t_upper ∈ {79, 78, 77, 76}`, 14400s (4h) budget per probe,
8 workers, warm-start from `fixtures/schedule_8x4x4_loaded_cpsat_literal.json`.

| t_upper | status | makespan | violations | runtime |
|---:|---|---:|---:|---:|
| (pending) | | | | |

## Probe 2: LNS with CP-SAT subsolver

Driver: `lns_cpsat_repair`, `n_iters=100`, `per_subproblem_budget_s=300`,
`destroy_size_frac=0.05`, strategies rotated time_window / random_subset /
makespan_flows.

| iter | strategy | destroy_size | new_makespan | accepted | runtime |
|---:|---|---:|---:|:---:|---:|
| (pending) | | | | | |

## Summary

(Filled in after probes complete.)
```

- [ ] **Step 4: Commit**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
git add eval/explorations/2026-05-16-closing-gap-to-lb-75/
git commit -m "exploration: scaffold 2026-05-16-closing-gap-to-lb-75 (README + RESULTS)"
```

---

### Task 6: CP-SAT warm-start probe script

**Files:**
- Create: `eval/explorations/2026-05-16-closing-gap-to-lb-75/01_cpsat_warm_start_probe.py`

- [ ] **Step 1: Write the probe script**

Create `eval/explorations/2026-05-16-closing-gap-to-lb-75/01_cpsat_warm_start_probe.py`:

```python
"""Probe 1: warm-start CP-SAT from the makespan-80 incumbent.

For each t_upper in {79, 78, 77, 76}, run cpsat_literal with a 4-hour
(14400s) budget, warm-started from the previous best incumbent (initially
fixtures/schedule_8x4x4_loaded_cpsat_literal.json; updated to whatever
FEASIBLE incumbent the prior probe produced).

Saves per-t_upper rows to 01_cpsat_warm_start_results.json. The best
schedule (overall lowest makespan) is saved to 01_best_warm_start_schedule.json.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

from twisted_analysis.topology import Topology
from twisted_analysis.io.routing_table import load_routing_table
from twisted_analysis.io.schedule import save_schedule, load_schedule
from twisted_analysis.schedules.cpsat_literal import cpsat_literal
from twisted_analysis.schedules.verify import verify_capacity, schedule_makespan

ROUTING = "fixtures/routing_table_8x4x4_twist.json"
SLICE = (8, 4, 4)
T_UPPER_SCHEDULE = [79, 78, 77, 76]
TIME_LIMIT_S = 14400  # 4 hours
WORKERS = 8

SEED_SCHEDULE = "fixtures/schedule_8x4x4_loaded_cpsat_literal.json"
OUT = Path(__file__).parent / "01_cpsat_warm_start_results.json"
BEST = Path(__file__).parent / "01_best_warm_start_schedule.json"


def run():
    topology = Topology(slice=SLICE)
    table = load_routing_table(ROUTING)
    seed = load_schedule(SEED_SCHEDULE)
    seed_makespan = schedule_makespan(seed)
    print(f"Seed: {SEED_SCHEDULE}, makespan={seed_makespan}, entries={len(seed)}",
          flush=True)

    rows = []
    best_makespan = seed_makespan
    current_seed = seed
    for t_upper in T_UPPER_SCHEDULE:
        print(f"\n--- t_upper={t_upper}, budget={TIME_LIMIT_S}s, "
              f"warm-start from makespan={schedule_makespan(current_seed)} ---",
              flush=True)
        t0 = time.time()
        try:
            sch = cpsat_literal(
                topology, table,
                t_upper=t_upper, time_limit_s=TIME_LIMIT_S,
                n_workers=WORKERS, solver_msg=True,
                warm_start_schedule=current_seed,
            )
            dt = time.time() - t0
            v = verify_capacity(sch)
            m = schedule_makespan(sch)
            row = {"t_upper": t_upper, "status": "FEASIBLE",
                   "makespan": m, "violations": len(v),
                   "runtime_s": round(dt, 1)}
            print(f"FEASIBLE: makespan={m}, viol={len(v)}, t={dt:.1f}s",
                  flush=True)
            if m < best_makespan:
                best_makespan = m
                save_schedule(sch, BEST)
                print(f"  saved new best to {BEST}", flush=True)
                current_seed = sch  # chain warm-start
        except RuntimeError as e:
            dt = time.time() - t0
            msg = str(e)
            status = "INFEASIBLE" if "infeasible" in msg.lower() else "TIMEOUT"
            row = {"t_upper": t_upper, "status": status,
                   "makespan": None, "violations": None,
                   "runtime_s": round(dt, 1), "error": msg}
            print(f"{status}: {msg} (t={dt:.1f}s)", flush=True)
            if status == "INFEASIBLE":
                rows.append(row)
                break
        rows.append(row)

    result = {
        "schedule": T_UPPER_SCHEDULE,
        "seed_schedule_file": SEED_SCHEDULE,
        "seed_makespan": seed_makespan,
        "best_makespan": best_makespan,
        "best_schedule_file": str(BEST) if best_makespan < seed_makespan else None,
        "rows": rows,
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Smoke-test the script with a tiny budget (do NOT run the full probe yet)**

Temporarily edit the script: change `T_UPPER_SCHEDULE = [79, 78, 77, 76]` → `T_UPPER_SCHEDULE = [80]` and `TIME_LIMIT_S = 14400` → `TIME_LIMIT_S = 30`. Run:

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis && PATH=".venv/bin:$PATH" .venv/bin/python -u eval/explorations/2026-05-16-closing-gap-to-lb-75/01_cpsat_warm_start_probe.py
```

Expected: prints `Seed: ... makespan=80`, attempts t_upper=80 with 30s budget, either lands at makespan 80 (FEASIBLE, hint accepted) or TIMEOUT. Both outcomes confirm the script wires up correctly. Then revert the edits:

```bash
git checkout -- eval/explorations/2026-05-16-closing-gap-to-lb-75/01_cpsat_warm_start_probe.py
# Also delete the smoke-test results files so they don't get committed:
rm -f eval/explorations/2026-05-16-closing-gap-to-lb-75/01_cpsat_warm_start_results.json \
      eval/explorations/2026-05-16-closing-gap-to-lb-75/01_best_warm_start_schedule.json
```

- [ ] **Step 3: Commit the script**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
git add eval/explorations/2026-05-16-closing-gap-to-lb-75/01_cpsat_warm_start_probe.py
git commit -m "exploration: add cpsat warm-start probe script (script only)"
```

---

### Task 7: LNS probe script

**Files:**
- Create: `eval/explorations/2026-05-16-closing-gap-to-lb-75/02_lns_probe.py`

- [ ] **Step 1: Write the LNS probe script**

Create `eval/explorations/2026-05-16-closing-gap-to-lb-75/02_lns_probe.py`:

```python
"""Probe 2: LNS with CP-SAT subsolver, seeded from the makespan-80 incumbent.

Runs lns_cpsat_repair for 100 iterations with a 300s (5 min) per-subproblem
budget. Strategies rotate time_window / random_subset / makespan_flows.
Logs every iteration (strategy, destroy size, current makespan, whether
accepted) to 02_lns_results.json. The final schedule (which is the best
incumbent across all iterations) is saved to 02_best_lns_schedule.json
if its makespan is strictly below the seed's.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

from twisted_analysis.topology import Topology
from twisted_analysis.io.routing_table import load_routing_table
from twisted_analysis.io.schedule import save_schedule, load_schedule
from twisted_analysis.schedules.lns_cpsat import lns_cpsat_repair
from twisted_analysis.schedules.verify import verify_capacity, schedule_makespan

ROUTING = "fixtures/routing_table_8x4x4_twist.json"
SLICE = (8, 4, 4)
SEED_SCHEDULE = "fixtures/schedule_8x4x4_loaded_cpsat_literal.json"

N_ITERS = 100
PER_SUBPROBLEM_S = 300
DESTROY_FRAC = 0.05
RNG_SEED = 20260516
WORKERS = 8
STRATEGIES = ("time_window", "random_subset", "makespan_flows")

OUT = Path(__file__).parent / "02_lns_results.json"
BEST = Path(__file__).parent / "02_best_lns_schedule.json"


def run():
    topology = Topology(slice=SLICE)
    table = load_routing_table(ROUTING)
    seed = load_schedule(SEED_SCHEDULE)
    seed_makespan = schedule_makespan(seed)
    print(f"Seed: {SEED_SCHEDULE}, makespan={seed_makespan}", flush=True)

    iter_log: list[dict] = []
    t_start = time.time()

    def log(it: int, info: dict):
        t = round(time.time() - t_start, 1)
        info["iter"] = it
        info["elapsed_s"] = t
        iter_log.append(dict(info))
        print(f"  iter {it:3d}: strat={info['strategy']:<16s} "
              f"|D|={info['destroy_size']:>5d} "
              f"cur_M={info['current_makespan']:>3d} "
              f"target={info['target_t_upper']:>3d} "
              f"t={t:>7.1f}s",
              flush=True)

    print(f"--- LNS: n_iters={N_ITERS}, per_subproblem={PER_SUBPROBLEM_S}s, "
          f"destroy_frac={DESTROY_FRAC}, strategies={STRATEGIES} ---",
          flush=True)
    sch = lns_cpsat_repair(
        topology, table, seed,
        n_iters=N_ITERS,
        per_subproblem_budget_s=PER_SUBPROBLEM_S,
        destroy_strategies=STRATEGIES,
        destroy_size_frac=DESTROY_FRAC,
        rng_seed=RNG_SEED,
        n_workers=WORKERS,
        log_fn=log,
    )
    elapsed = round(time.time() - t_start, 1)
    v = verify_capacity(sch)
    m = schedule_makespan(sch)
    print(f"\nFinal: makespan={m}, viol={len(v)}, total={elapsed}s", flush=True)

    if m < seed_makespan:
        save_schedule(sch, BEST)
        print(f"  saved new best to {BEST}", flush=True)

    result = {
        "seed_schedule_file": SEED_SCHEDULE,
        "seed_makespan": seed_makespan,
        "final_makespan": m,
        "violations": len(v),
        "total_runtime_s": elapsed,
        "best_schedule_file": str(BEST) if m < seed_makespan else None,
        "params": {
            "n_iters": N_ITERS, "per_subproblem_budget_s": PER_SUBPROBLEM_S,
            "destroy_size_frac": DESTROY_FRAC, "rng_seed": RNG_SEED,
            "strategies": list(STRATEGIES), "n_workers": WORKERS,
        },
        "iter_log": iter_log,
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Smoke-test with tiny budget (do NOT run full probe)**

Temporarily edit the script: change `N_ITERS = 100` → `N_ITERS = 2` and `PER_SUBPROBLEM_S = 300` → `PER_SUBPROBLEM_S = 20`. Run:

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis && PATH=".venv/bin:$PATH" .venv/bin/python -u eval/explorations/2026-05-16-closing-gap-to-lb-75/02_lns_probe.py
```

Expected: 2 iter lines printed, both with `cur_M=80` (initially), final makespan ≤ 80, no exceptions. Restore the edits and clean up smoke-test outputs:

```bash
git checkout -- eval/explorations/2026-05-16-closing-gap-to-lb-75/02_lns_probe.py
rm -f eval/explorations/2026-05-16-closing-gap-to-lb-75/02_lns_results.json \
      eval/explorations/2026-05-16-closing-gap-to-lb-75/02_best_lns_schedule.json
```

- [ ] **Step 3: Commit the script**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
git add eval/explorations/2026-05-16-closing-gap-to-lb-75/02_lns_probe.py
git commit -m "exploration: add LNS probe script (script only)"
```

---

### Task 8: Execute both probes (multi-day, sequential)

**Files (run, then commit logs/results):**
- `01_cpsat_warm_start_results.json` + `01_best_warm_start_schedule.json` (conditional)
- `02_lns_results.json` + `02_best_lns_schedule.json` (conditional)

CP-SAT uses 8 workers per probe; running both in parallel would oversubscribe CPU. Run sequentially. Probe 1 ≈ 16h wall-clock (4 × 4h); Probe 2 ≈ 8h. Total ≈ 24h. Use `run_in_background` for each.

- [ ] **Step 1: Launch probe 1 in the background**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis && PATH=".venv/bin:$PATH" .venv/bin/python -u eval/explorations/2026-05-16-closing-gap-to-lb-75/01_cpsat_warm_start_probe.py 2>&1 | tee eval/explorations/2026-05-16-closing-gap-to-lb-75/01_cpsat_warm_start_log.txt
```

Run with `run_in_background=true` in the Bash tool. The script logs the per-probe status to stdout (via `solver_msg=True`); the tee captures it.

- [ ] **Step 2: Wait for probe 1 to finish**

The implementer (or controller) waits for completion notification. Inspect `01_cpsat_warm_start_results.json`. If `best_makespan < 80`, the new best schedule is at `01_best_warm_start_schedule.json` — use it as the seed for probe 2. Otherwise, probe 2 keeps the same seed (`fixtures/schedule_8x4x4_loaded_cpsat_literal.json`).

If probe 1 produced a new best, edit `02_lns_probe.py`:

```python
SEED_SCHEDULE = "eval/explorations/2026-05-16-closing-gap-to-lb-75/01_best_warm_start_schedule.json"
```

(Otherwise leave the original `SEED_SCHEDULE`.)

- [ ] **Step 3: Launch probe 2 in the background**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis && PATH=".venv/bin:$PATH" .venv/bin/python -u eval/explorations/2026-05-16-closing-gap-to-lb-75/02_lns_probe.py 2>&1 | tee eval/explorations/2026-05-16-closing-gap-to-lb-75/02_lns_log.txt
```

Run with `run_in_background=true`. Wait for completion notification.

- [ ] **Step 4: Sanity-check results, commit logs + results**

After both probes finish:

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
cat eval/explorations/2026-05-16-closing-gap-to-lb-75/01_cpsat_warm_start_results.json | head -30
cat eval/explorations/2026-05-16-closing-gap-to-lb-75/02_lns_results.json | head -30
git add eval/explorations/2026-05-16-closing-gap-to-lb-75/01_cpsat_warm_start_results.json \
        eval/explorations/2026-05-16-closing-gap-to-lb-75/01_cpsat_warm_start_log.txt \
        eval/explorations/2026-05-16-closing-gap-to-lb-75/02_lns_results.json \
        eval/explorations/2026-05-16-closing-gap-to-lb-75/02_lns_log.txt
# Conditionally include best-schedule files (only if they were saved):
[ -f eval/explorations/2026-05-16-closing-gap-to-lb-75/01_best_warm_start_schedule.json ] && \
  git add eval/explorations/2026-05-16-closing-gap-to-lb-75/01_best_warm_start_schedule.json
[ -f eval/explorations/2026-05-16-closing-gap-to-lb-75/02_best_lns_schedule.json ] && \
  git add eval/explorations/2026-05-16-closing-gap-to-lb-75/02_best_lns_schedule.json
# If probe 2's seed line was edited, revert that edit before committing:
git checkout -- eval/explorations/2026-05-16-closing-gap-to-lb-75/02_lns_probe.py
git commit -m "exploration: probe 1 + probe 2 results (warm-start CP-SAT, LNS)"
```

---

### Task 9: Document results + conditional fixture promotion

**Files:**
- Modify: `eval/explorations/2026-05-16-closing-gap-to-lb-75/RESULTS.md`
- Modify: `eval/explorations/2026-05-16-closing-gap-to-lb-75/README.md` (fill in Outcome section)
- Conditional: `fixtures/schedule_8x4x4_loaded_<scheduler>.json`,
  `fixtures/cns_schedules/schedule_<name>_4x4x8_twisted.json`,
  `fixtures/cns_schedules/readme.md`,
  `eval/explorations/2026-05-16-closing-gap-to-lb-75/best_schedule.json`,
  `eval/explorations/2026-05-16-closing-gap-to-lb-75/best_pallas_kernel.py`

- [ ] **Step 1: Compute the overall best and update RESULTS.md**

Determine the overall winner across both probes (lowest makespan among `01_cpsat_warm_start_results.json::best_makespan` and `02_lns_results.json::final_makespan`, vs the incoming 80).

In `eval/explorations/2026-05-16-closing-gap-to-lb-75/RESULTS.md`, replace the `(pending)` rows with the actual data from the two results JSONs. For Probe 2, summarize the iter log: how many accepted iterations, lowest makespan reached, runtime, per-strategy contribution counts. Add a final "Summary" section with a table:

```markdown
## Summary — Probe winners

| Probe | Best makespan | Beats seed (80)? |
|---|---:|:---:|
| Probe 1 warm-start CP-SAT | <X> | <yes/no> |
| Probe 2 LNS              | <Y> | <yes/no> |

**Overall winner: <min(X, Y)>** at <method>. <Beats P2P by N%?>
```

- [ ] **Step 2: Fill in README.md "Outcome" section**

In `eval/explorations/2026-05-16-closing-gap-to-lb-75/README.md`, replace the `(Filled in after probes complete.)` placeholder with:

- The overall makespan winner and method
- Throughput projection: linear scaling from `132758 gbps @ makespan 85` → `132758 × 85 / winner_makespan` gbps; vs P2P 134541 gbps.
- Whether LB=75 was reached
- What each probe taught us (numbered bullets like the 2026-05-15 README)
- Open future work (e.g., still-untried HiGHS LP-rounding, longer LNS, structural decomposition)

- [ ] **Step 3: Commit docs**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
git add eval/explorations/2026-05-16-closing-gap-to-lb-75/RESULTS.md \
        eval/explorations/2026-05-16-closing-gap-to-lb-75/README.md
git commit -m "exploration: document closing-gap-to-lb-75 results"
```

- [ ] **Step 4: If overall best < 80 — save best_schedule.json and regenerate Pallas kernel**

Determine the overall winner schedule path (`01_best_warm_start_schedule.json` or `02_best_lns_schedule.json`, whichever has lower makespan). Save it as `best_schedule.json` in the exploration folder and regenerate the Pallas kernel:

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
# Replace <WINNER> with the actual filename:
cp eval/explorations/2026-05-16-closing-gap-to-lb-75/<WINNER>.json \
   eval/explorations/2026-05-16-closing-gap-to-lb-75/best_schedule.json
.venv/bin/python -u pallas_kernel/gen_orbit_greedy_kernel.py \
   --schedule-in eval/explorations/2026-05-16-closing-gap-to-lb-75/best_schedule.json \
   --out eval/explorations/2026-05-16-closing-gap-to-lb-75/best_pallas_kernel.py
```

Expected: kernel generated; its docstring header says "loaded-from best_schedule.json"; the file has the same `pl.pallas_call` structure as `eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/best_pallas_kernel.py`. If gen fails because `--schedule-in` doesn't accept this path format, debug at that point (this is the same flag added in the prior exploration).

- [ ] **Step 5: If overall best < 80 — promote to fixtures + cns_schedules**

Decide on a scheduler-name suffix for the fixture (e.g., if probe 1 won via warm-start CP-SAT, use `cpsat_literal_warm`; if probe 2 won, use `lns_cpsat`). Then:

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
# Replace <SUFFIX> and <CNS_NAME> with the chosen names:
cp eval/explorations/2026-05-16-closing-gap-to-lb-75/best_schedule.json \
   fixtures/schedule_8x4x4_loaded_<SUFFIX>.json
cp eval/explorations/2026-05-16-closing-gap-to-lb-75/best_schedule.json \
   fixtures/cns_schedules/schedule_<CNS_NAME>_4x4x8_twisted.json
```

Edit `fixtures/cns_schedules/readme.md` — add a new row at the top of the table (above the current `cpsatliteral` row) for the new winner; mark it as the new "Recommended for production measurement runs"; demote the prior `cpsatliteral` recommendation to "previous best" in the table description. Update the "ILP-optimal not provided" section to note that the prior `80 → 75` gap has been narrowed (or closed) by this exploration; cite the folder path.

- [ ] **Step 6: Commit fixture promotion (conditional)**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
git add eval/explorations/2026-05-16-closing-gap-to-lb-75/best_schedule.json \
        eval/explorations/2026-05-16-closing-gap-to-lb-75/best_pallas_kernel.py \
        fixtures/schedule_8x4x4_loaded_<SUFFIX>.json \
        fixtures/cns_schedules/schedule_<CNS_NAME>_4x4x8_twisted.json \
        fixtures/cns_schedules/readme.md
git commit -m "fixtures: promote new best (<METHOD>, makespan <M>) to fixtures + cns_schedules"
```

- [ ] **Step 7: If overall best == 80 (no improvement) — still document the negative result**

Skip Steps 4–6. The exploration documents that the 80 → 75 gap was NOT closed by either method at this compute budget. Update the README "Outcome" to call this out clearly and propose specific next attempts (longer LNS, larger destroy fraction, HiGHS LP-rounding, structural decomposition). No fixture changes.

---

### Task 10: Update root README scheduler matrix

**Files:**
- Modify: `README.md` (repo root)

- [ ] **Step 1: Add `lns_cpsat` row to scheduler matrix**

Open `README.md` at the repo root. Find the scheduler-matrix table (rows for `orbit_greedy`, `orbit_greedy_full`, `literal_greedy`, `ilp_literal`, `cpsat_literal`, `lp_rounding`, `local_search`). Add one row:

```markdown
| `lns_cpsat`         | Large-Neighborhood Search repair. Each iteration destroys 5–10% of a seed schedule (time-window / random-subset / makespan-bottleneck-edge) and asks CP-SAT to re-optimize. Accepts strictly-better incumbents. Requires `seed_schedule`. Best when greedy / single-shot CP-SAT lands at a local optimum. |
```

Then below the table, in the "Exploration TL;DR" section (or equivalent), add a one-liner linking the new exploration:

```markdown
- [2026-05-16: Closing the gap to LB=75 on loaded 8×4×4](eval/explorations/2026-05-16-closing-gap-to-lb-75/README.md) — warm-start CP-SAT + LNS, makespan <RESULT> (target LB=75).
```

Replace `<RESULT>` with the actual final makespan.

- [ ] **Step 2: Commit**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
git add README.md
git commit -m "README: register lns_cpsat scheduler and link 2026-05-16 exploration"
```

---

## Post-implementation hand-off

After Task 10, the work is complete. Tests pass; results committed; fixtures conditionally promoted. The implementer-or-controller may then invoke `superpowers:finishing-a-development-branch` to merge / PR / hand off.

If the overall best schedule did **not** beat makespan 80, the natural follow-ups (do not include in this plan — file them as new explorations):

- Probe 3: HiGHS-backed LP-rounding (`lp_rounding` with HiGHS as the LP backend, not CBC)
- Probe 4: column-generation / Dantzig-Wolfe reformulation
- Probe 5: simulated annealing with chain-swap move set
- Probe 6: LP relaxation lower bound (compute the LP makespan bound to see how much room remains to LB=75)
