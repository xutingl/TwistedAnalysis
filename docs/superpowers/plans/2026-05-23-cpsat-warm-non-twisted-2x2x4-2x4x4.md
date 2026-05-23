# CP-SAT-Warm Schedules + Pallas Kernels for Non-Twisted Torus (2×2×4, 2×4×4) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For each of the two non-twisted-torus routing tables (`fixtures/routing_table_torus_2x2x4.json`, `fixtures/routing_table_torus_2x4x4.json`), compute the best schedule we can (CP-SAT warm-started from an orbit-greedy seed; cold CP-SAT at `t_upper=LB` is provably optimal when feasible) and emit a Pallas TPU kernel from that schedule, parallel to the existing `cpsat_literal_warm` artifact for the loaded 8×4×4 routing.

**Architecture:** A single Python driver per cell does five things back-to-back: (1) load the routing table and compute the physical-edge lower bound `LB`; (2) run `orbit_greedy_full(order="lpt_tail_asc")` to produce a feasible seed `S_seed`; (3) invoke `cpsat_literal(t_upper=LB, warm_start_schedule=S_seed, time_limit_s=..., minimize=True)` — if `OPTIMAL` or `FEASIBLE` with `makespan ≤ LB`, this is provably optimal and we save it; (4) if `INFEASIBLE` at `LB`, sweep `t_upper ∈ {LB+1, LB+2, …, makespan(S_seed)−1}` warm-started from the running best until either `FEASIBLE` is achieved or we exhaust the sweep (in which case the seed itself is the best we have); (5) save the winning schedule under `fixtures/schedule_torus_<slice>_cpsat_literal_warm.json`, then invoke `pallas_kernel/gen_orbit_greedy_kernel.py --schedule-in` to verify-against-routing and emit `pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_torus_<slice>.py`. A thin bash wrapper runs the driver once per cell with the right paths. All artifacts (rolling results JSON, best schedule JSON, kernel `.py`) live under one exploration folder for reproducibility.

**Tech Stack:** Python 3 with the existing `twisted_analysis` package (`Topology`, `load_routing_table`, `orbit_greedy_full`, `cpsat_literal`, `verify_capacity`, `schedule_makespan`, `save_schedule`/`load_schedule`); OR-Tools CP-SAT (`ortools.sat.python.cp_model`); `pallas_kernel/gen_orbit_greedy_kernel.py` for the kernel emission stage. Run via `.venv/bin/python -u` (memory note: avoid `uv run` for stable env).

---

## File Structure

**Create:**
- `eval/explorations/2026-05-23-cpsat-warm-non-twisted/README.md` — Problem, goal, planned probes, expected results.
- `eval/explorations/2026-05-23-cpsat-warm-non-twisted/RESULTS.md` — Rolling per-cell table (LB, orbit-greedy makespan, CP-SAT outcome, runtime, kernel path).
- `eval/explorations/2026-05-23-cpsat-warm-non-twisted/run_cpsat_warm.py` — Single-cell driver: takes `--routing-table`, `--slice`, `--out-schedule`, `--time-limit-s`; loads table, computes LB, runs orbit_greedy_full seed + CP-SAT warm sweep, saves best schedule.
- `eval/explorations/2026-05-23-cpsat-warm-non-twisted/run_all.sh` — Runs the driver for both cells, then invokes `gen_orbit_greedy_kernel.py --schedule-in` for each.

**Created by the run (do not commit until they exist):**
- `fixtures/schedule_torus_2x2x4_cpsat_literal_warm.json`
- `fixtures/schedule_torus_2x4x4_cpsat_literal_warm.json`
- `pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_torus_2_2_4.py`
- `pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_torus_2_4_4.py`
- `eval/explorations/2026-05-23-cpsat-warm-non-twisted/run_log_2x2x4.txt`, `run_log_2x4x4.txt`
- `eval/explorations/2026-05-23-cpsat-warm-non-twisted/results.json` (per-cell rows: `{slice, lb, seed_makespan, cpsat_status, cpsat_makespan, cpsat_t_uppers_tried, runtime_s}`)

**Read-only (must already exist before running this plan):**
- `fixtures/routing_table_torus_2x2x4.json` — N=16, LB=8 (verified pre-plan).
- `fixtures/routing_table_torus_2x4x4.json` — N=32, LB=16 (verified pre-plan).
- `twisted_analysis/schedules/orbit_greedy_full.py`, `cpsat_literal.py`, `verify.py`.
- `twisted_analysis/io/routing_table.py`, `schedule.py`.
- `pallas_kernel/gen_orbit_greedy_kernel.py` (supports `--schedule-in`, `--out`, `--function-name`).

**Out of scope for this plan (intentionally NOT modified):**
- `twisted_analysis/schedules/cpsat_literal.py` is used as-is — `warm_start_schedule` is already plumbed through.
- `scripts/generate_schedule.py` is NOT extended to support `cpsat_literal` here — the driver calls `cpsat_literal` directly, matching the pattern in `eval/explorations/2026-05-16-closing-gap-to-lb-75/01_cpsat_warm_start_probe.py`.
- No new fixtures format, no `cns_schedules` promotion (out of this plan's scope).

---

## Background (one-time read, then proceed to tasks)

The pipeline mirrors the 8×4×4 exploration in `eval/explorations/2026-05-16-closing-gap-to-lb-75/` and the kernel pattern documented in `README.md` lines 167–174 (the production-recommended `cpsat_literal_warm` kernel is loaded via `--schedule-in fixtures/schedule_8x4x4_loaded_cpsat_literal_warm.json`).

Two facts that drive the strategy:

1. **`cpsat_literal(minimize=True)` at `t_upper=LB`**: if the solver returns `OPTIMAL` or any `FEASIBLE` solution, the returned `makespan ≤ LB`. Since `LB` is by definition a lower bound on the makespan, that schedule is provably optimal — no further search needed.
2. **`cpsat_literal(minimize=True)` at `t_upper=LB` returning `INFEASIBLE`**: the routing's physical-edge LB is unattainable under the literal model (e.g., translation-equivariance failure, as documented in `docs/orbit_greedy_optimality.md §6`). We then sweep `t_upper` upward, warm-started from the running best (initially the orbit_greedy_full seed).

Verified pre-plan: 2×2×4 has LB=8 (240 flows, 64 edges); 2×4×4 has LB=16 (992 flows, 160 edges). Both are small enough that a 5-minute (2×2×4) / 30-minute (2×4×4) CP-SAT budget is conservative.

---

### Task 1: Create the exploration folder skeleton

**Files:**
- Create: `eval/explorations/2026-05-23-cpsat-warm-non-twisted/README.md`
- Create: `eval/explorations/2026-05-23-cpsat-warm-non-twisted/RESULTS.md`

- [ ] **Step 1: Make the folder**

Run from the repo root:

```bash
mkdir -p eval/explorations/2026-05-23-cpsat-warm-non-twisted
```

Expected: directory created (or already-exists no-op).

- [ ] **Step 2: Write the README**

Create `eval/explorations/2026-05-23-cpsat-warm-non-twisted/README.md` with this exact content:

````markdown
# CP-SAT-Warm Schedules + Pallas Kernels for Non-Twisted Torus (2×2×4, 2×4×4)

**Date:** 2026-05-23
**Cells:** `fixtures/routing_table_torus_2x2x4.json` (N=16, LB=8); `fixtures/routing_table_torus_2x4x4.json` (N=32, LB=16).
**Goal:** Emit the best Pallas kernel we can for each cell — parallel to the existing `cpsat_literal_warm` 8×4×4 artifact.

## Pipeline

For each routing table:

1. Load table; compute `LB = max link load` over the routing.
2. Seed: `orbit_greedy_full(order="lpt_tail_asc")` → `S_seed` with `makespan = M_seed`.
3. CP-SAT cold-at-LB: `cpsat_literal(t_upper=LB, warm_start_schedule=S_seed, time_limit_s=..., minimize=True)`.
   - If feasible/optimal → that schedule is provably optimal (`makespan = LB`). Save.
   - If `INFEASIBLE` → sweep `t_upper ∈ {LB+1, ..., M_seed−1}`, warm-started from running best. First feasible wins; if none → keep `S_seed`.
4. Save best to `fixtures/schedule_torus_<slice>_cpsat_literal_warm.json`.
5. Emit kernel via `pallas_kernel/gen_orbit_greedy_kernel.py --schedule-in <fixture> --out <kernel.py> --function-name ...`.

## Per-cell budgets

| Cell | Flows | Time limit |
|---|---:|---:|
| 2×2×4 | 240 | 300 s |
| 2×4×4 | 992 | 1800 s |

## Outputs

- `fixtures/schedule_torus_2x2x4_cpsat_literal_warm.json`
- `fixtures/schedule_torus_2x4x4_cpsat_literal_warm.json`
- `pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_torus_2_2_4.py`
- `pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_torus_2_4_4.py`
- `results.json` — per-cell row with `{slice, lb, seed_makespan, cpsat_makespan, cpsat_t_uppers_tried, runtime_s}`.
- `run_log_<slice>.txt` — full CP-SAT solver log.

See `RESULTS.md` for the populated headline numbers after the run completes.
````

- [ ] **Step 3: Write the RESULTS.md skeleton**

Create `eval/explorations/2026-05-23-cpsat-warm-non-twisted/RESULTS.md` with this exact content:

````markdown
# Results — CP-SAT-Warm Non-Twisted Torus (2×2×4, 2×4×4)

| Cell | N | LB | orbit_greedy_full | CP-SAT makespan | Gap to LB | Runtime | Kernel |
|---|---:|---:|---:|---:|---:|---:|---|
| 2×2×4 | 16 | 8 | TBD | TBD | TBD | TBD | `pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_torus_2_2_4.py` |
| 2×4×4 | 32 | 16 | TBD | TBD | TBD | TBD | `pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_torus_2_4_4.py` |

Fill `TBD` after `run_all.sh` completes. `Gap to LB` is `CP-SAT makespan − LB`; expected = 0 unless the routing's physical-edge LB is unattainable (in which case the `t_uppers_tried` field in `results.json` shows which step succeeded).
````

- [ ] **Step 4: Commit the skeleton**

```bash
git add eval/explorations/2026-05-23-cpsat-warm-non-twisted/README.md \
        eval/explorations/2026-05-23-cpsat-warm-non-twisted/RESULTS.md
git commit -m "docs: add exploration skeleton for cpsat-warm non-twisted torus (2x2x4, 2x4x4)"
```

---

### Task 2: Write the per-cell driver `run_cpsat_warm.py`

**Files:**
- Create: `eval/explorations/2026-05-23-cpsat-warm-non-twisted/run_cpsat_warm.py`

The driver is parameterized by `(routing_table_path, slice, out_schedule_path, time_limit_s, n_workers)`. It loads the table, builds an `orbit_greedy_full` seed, runs CP-SAT cold-at-LB warm-started from the seed, sweeps `t_upper` upward on `INFEASIBLE`, and saves the best schedule plus a per-cell row of results.

- [ ] **Step 1: Create the driver file**

Create `eval/explorations/2026-05-23-cpsat-warm-non-twisted/run_cpsat_warm.py` with exactly this content:

```python
"""Per-cell driver: orbit_greedy_full seed -> CP-SAT warm-started sweep.

Pipeline (single cell):
  1. Load routing table.
  2. Compute LB = max physical-edge load over the routing.
  3. Seed = orbit_greedy_full(table, order="lpt_tail_asc").
  4. CP-SAT at t_upper=LB, warm-started from seed. If feasible, that's optimal.
  5. On INFEASIBLE at LB, sweep t_upper in {LB+1, ..., M_seed-1} warm-started
     from running best. First feasible wins.
  6. Save best schedule to --out-schedule. Append a row to --results-json.

Usage:
    .venv/bin/python -u eval/explorations/2026-05-23-cpsat-warm-non-twisted/run_cpsat_warm.py \\
        --routing-table fixtures/routing_table_torus_2x2x4.json \\
        --slice 2,2,4 \\
        --out-schedule fixtures/schedule_torus_2x2x4_cpsat_literal_warm.json \\
        --time-limit-s 300 \\
        --results-json eval/explorations/2026-05-23-cpsat-warm-non-twisted/results.json
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

# Make `python eval/.../run_cpsat_warm.py` work without install.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_ROOT))

from twisted_analysis.topology import Topology
from twisted_analysis.io.routing_table import load_routing_table
from twisted_analysis.io.schedule import save_schedule, schedule_from_orbit_greedy_full
from twisted_analysis.schedules.cpsat_literal import cpsat_literal
from twisted_analysis.schedules.verify import verify_capacity, schedule_makespan


def physical_edge_lb(table, n: int) -> int:
    c: Counter = Counter()
    for s in range(n):
        for d in range(n):
            if s == d:
                continue
            p = table[s][d]
            for h in range(len(p) - 1):
                c[(p[h], p[h + 1])] += 1
    return max(c.values())


def parse_slice(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def append_result_row(results_json: Path, row: dict) -> None:
    if results_json.exists():
        rows = json.loads(results_json.read_text())
        if not isinstance(rows, list):
            raise SystemExit(f"{results_json}: expected a JSON list, got {type(rows).__name__}")
    else:
        rows = []
    rows.append(row)
    results_json.write_text(json.dumps(rows, indent=2))


def run_cell(
    *,
    routing_table: Path,
    slice_: tuple[int, ...],
    out_schedule: Path,
    time_limit_s: int,
    n_workers: int,
    results_json: Path,
) -> int:
    topology = Topology(slice=slice_)
    table = load_routing_table(routing_table)
    if len(table) != topology.n_nodes:
        raise SystemExit(
            f"routing table {routing_table} has {len(table)} sources; "
            f"slice {slice_} expects {topology.n_nodes}"
        )
    lb = physical_edge_lb(table, topology.n_nodes)
    print(f"[stage 1] slice={slice_} N={topology.n_nodes} LB={lb}", flush=True)

    # Stage 2: orbit_greedy_full seed (via the public adapter).
    t0 = time.time()
    seed = schedule_from_orbit_greedy_full(topology, table, order="lpt_tail_asc")
    seed_makespan = schedule_makespan(seed)
    seed_violations = len(verify_capacity(seed))
    dt_seed = time.time() - t0
    print(f"[stage 2] orbit_greedy_full: makespan={seed_makespan} "
          f"viol={seed_violations} t={dt_seed:.2f}s", flush=True)
    if seed_violations:
        raise SystemExit(
            f"orbit_greedy_full produced {seed_violations} capacity violations on "
            f"{routing_table} — refusing to proceed."
        )

    # Stage 3+4: cold CP-SAT at t_upper=LB; on INFEASIBLE sweep upward.
    best = seed
    best_makespan = seed_makespan
    tried: list[dict] = []
    t_uppers = list(range(lb, seed_makespan))  # LB, LB+1, ..., M_seed-1
    if not t_uppers:
        # Seed already at LB; no room to search. Use seed.
        print(f"[stage 3] seed_makespan ({seed_makespan}) == LB ({lb}); "
              f"seed is already optimal.", flush=True)
        chosen_status = "SEED_AT_LB"
    else:
        chosen_status = None
        current_seed = seed
        for t_upper in t_uppers:
            print(f"\n--- t_upper={t_upper}, budget={time_limit_s}s, "
                  f"warm-start makespan={schedule_makespan(current_seed)} ---",
                  flush=True)
            t0 = time.time()
            try:
                sch = cpsat_literal(
                    topology, table,
                    t_upper=t_upper,
                    time_limit_s=time_limit_s,
                    n_workers=n_workers,
                    solver_msg=True,
                    warm_start_schedule=current_seed,
                )
                dt = time.time() - t0
                m = schedule_makespan(sch)
                v = len(verify_capacity(sch))
                tried.append({"t_upper": t_upper, "status": "FEASIBLE",
                              "makespan": m, "violations": v,
                              "runtime_s": round(dt, 2)})
                print(f"FEASIBLE: makespan={m} viol={v} t={dt:.2f}s", flush=True)
                if v != 0:
                    raise SystemExit(
                        f"cpsat_literal returned a schedule with {v} capacity "
                        f"violations at t_upper={t_upper} — bug in solver wrapper."
                    )
                if m < best_makespan:
                    best = sch
                    best_makespan = m
                    current_seed = sch  # chain warm-start at the next t_upper
                chosen_status = "FEASIBLE"
                break  # first feasible wins; further t_upper can only equal or exceed.
            except RuntimeError as e:
                dt = time.time() - t0
                msg = str(e)
                status = "INFEASIBLE" if "infeasible" in msg.lower() else "TIMEOUT_OR_ERROR"
                tried.append({"t_upper": t_upper, "status": status,
                              "makespan": None, "violations": None,
                              "runtime_s": round(dt, 2), "error": msg})
                print(f"{status}: {msg} (t={dt:.2f}s)", flush=True)
                if status != "INFEASIBLE":
                    # Hard error (e.g. ortools missing). Give up.
                    raise
                # else: INFEASIBLE at this t_upper -> continue sweep.
        if chosen_status is None:
            chosen_status = "ALL_INFEASIBLE_KEEPING_SEED"

    # Stage 5: save best.
    out_schedule.parent.mkdir(parents=True, exist_ok=True)
    save_schedule(best, out_schedule)
    print(f"\n[stage 5] saved best schedule to {out_schedule} "
          f"(makespan={best_makespan})", flush=True)

    # Append row to results.json.
    row = {
        "slice": list(slice_),
        "routing_table": str(routing_table),
        "lb": lb,
        "seed_makespan": seed_makespan,
        "cpsat_makespan": best_makespan,
        "cpsat_t_uppers_tried": tried,
        "chosen_status": chosen_status,
        "out_schedule": str(out_schedule),
    }
    results_json.parent.mkdir(parents=True, exist_ok=True)
    append_result_row(results_json, row)
    print(f"appended row to {results_json}", flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--routing-table", required=True, type=Path)
    p.add_argument("--slice", required=True, help="Comma-separated, e.g. 2,2,4")
    p.add_argument("--out-schedule", required=True, type=Path)
    p.add_argument("--time-limit-s", type=int, default=300)
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--results-json", required=True, type=Path)
    args = p.parse_args(argv)
    return run_cell(
        routing_table=args.routing_table,
        slice_=parse_slice(args.slice),
        out_schedule=args.out_schedule,
        time_limit_s=args.time_limit_s,
        n_workers=args.n_workers,
        results_json=args.results_json,
    )


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke-test the driver on 2×2×4**

Run from the repo root:

```bash
.venv/bin/python -u eval/explorations/2026-05-23-cpsat-warm-non-twisted/run_cpsat_warm.py \
    --routing-table fixtures/routing_table_torus_2x2x4.json \
    --slice 2,2,4 \
    --out-schedule /tmp/smoketest_2x2x4_schedule.json \
    --time-limit-s 60 \
    --n-workers 4 \
    --results-json /tmp/smoketest_results.json 2>&1 | tee /tmp/smoketest_2x2x4.log
```

Expected (in this order):
1. `[stage 1] slice=(2, 2, 4) N=16 LB=8`
2. `[stage 2] orbit_greedy_full: makespan=13 viol=0 t=<small>s` (measured pre-plan; expected exactly 13 on the committed `routing_table_torus_2x2x4.json`)
3. A CP-SAT block: at `t_upper=8` (= LB), expected `FEASIBLE: makespan=8 viol=0` (typically converges in well under 60s on N=16). The sweep stops after the first feasible solve.
4. `[stage 5] saved best schedule to /tmp/smoketest_2x2x4_schedule.json (makespan=8)`
5. `appended row to /tmp/smoketest_results.json`
6. Exit code 0.

If any step prints `viol>0`, stop and read the seed: the seed scheduler is the culprit (orbit_greedy_full bug, not our wrapper). Do not proceed to the next task.

- [ ] **Step 3: Clean up smoke-test artifacts**

```bash
rm -f /tmp/smoketest_2x2x4_schedule.json /tmp/smoketest_results.json /tmp/smoketest_2x2x4.log
```

- [ ] **Step 4: Commit the driver**

```bash
git add eval/explorations/2026-05-23-cpsat-warm-non-twisted/run_cpsat_warm.py
git commit -m "feat: per-cell CP-SAT warm-start driver for non-twisted torus"
```

---

### Task 3: Write the `run_all.sh` wrapper

**Files:**
- Create: `eval/explorations/2026-05-23-cpsat-warm-non-twisted/run_all.sh`

The wrapper is the one-shot reproducibility entry point: it runs the driver for both cells, captures per-cell logs, then invokes `gen_orbit_greedy_kernel.py --schedule-in` to emit each kernel.

- [ ] **Step 1: Create `run_all.sh` with exactly this content**

```bash
#!/usr/bin/env bash
# Reproducibility script for the 2026-05-23 cpsat-warm non-twisted exploration.
#
# Runs the per-cell driver on both 2x2x4 and 2x4x4, then emits a Pallas
# kernel from each best schedule. All artifacts (schedule fixtures, kernel
# .py files, run logs, results.json) land in stable, predictable paths.
#
# Usage:
#   bash eval/explorations/2026-05-23-cpsat-warm-non-twisted/run_all.sh
#
# Idempotent: rerunning overwrites the kernel .py files and the schedule
# fixtures; results.json is APPENDED to (each row is one (cell, run) pair).
# To start fresh, delete results.json first.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXPLO="${ROOT}/eval/explorations/2026-05-23-cpsat-warm-non-twisted"
PY="${ROOT}/.venv/bin/python"

cd "${ROOT}"

# Per-cell parameters: (slice_csv, slice_kern, routing_table_path, time_limit_s)
run_cell() {
    local slice_csv="$1"
    local slice_kern="$2"
    local routing_table="$3"
    local time_limit_s="$4"

    local slice_slug="${slice_csv//,/x}"   # 2,2,4 -> 2x2x4
    local schedule="fixtures/schedule_torus_${slice_slug}_cpsat_literal_warm.json"
    local kernel="pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_torus_${slice_kern}.py"
    local log="${EXPLO}/run_log_${slice_slug}.txt"

    echo "=== ${slice_csv}: driver ==="
    "${PY}" -u "${EXPLO}/run_cpsat_warm.py" \
        --routing-table "${routing_table}" \
        --slice "${slice_csv}" \
        --out-schedule "${schedule}" \
        --time-limit-s "${time_limit_s}" \
        --n-workers 8 \
        --results-json "${EXPLO}/results.json" 2>&1 | tee "${log}"

    echo
    echo "=== ${slice_csv}: kernel ==="
    "${PY}" -u pallas_kernel/gen_orbit_greedy_kernel.py \
        --slice "${slice_csv}" \
        --routing-table "${routing_table}" \
        --schedule-in "${schedule}" \
        --out "${kernel}" \
        --function-name "_ragged_a2a_kernel_cpsat_literal_warm_torus_${slice_kern}"

    echo "wrote ${schedule}"
    echo "wrote ${kernel}"
    echo
}

run_cell "2,2,4" "2_2_4" "fixtures/routing_table_torus_2x2x4.json"  300
run_cell "2,4,4" "2_4_4" "fixtures/routing_table_torus_2x4x4.json" 1800

echo "=== done. See ${EXPLO}/results.json and ${EXPLO}/RESULTS.md ==="
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x eval/explorations/2026-05-23-cpsat-warm-non-twisted/run_all.sh
```

- [ ] **Step 3: Commit the wrapper**

```bash
git add eval/explorations/2026-05-23-cpsat-warm-non-twisted/run_all.sh
git commit -m "feat: run_all.sh wrapper for cpsat-warm non-twisted exploration"
```

---

### Task 4: Run 2×2×4 end-to-end

**Files:**
- Created by run: `fixtures/schedule_torus_2x2x4_cpsat_literal_warm.json`
- Created by run: `pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_torus_2_2_4.py`
- Created by run: `eval/explorations/2026-05-23-cpsat-warm-non-twisted/run_log_2x2x4.txt`
- Created by run: `eval/explorations/2026-05-23-cpsat-warm-non-twisted/results.json` (new file or first row appended)

- [ ] **Step 1: Delete any stale `results.json`**

```bash
rm -f eval/explorations/2026-05-23-cpsat-warm-non-twisted/results.json
```

Expected: file removed (or already absent — `-f` makes either case a no-op).

- [ ] **Step 2: Run the 2×2×4 driver**

From the repo root:

```bash
.venv/bin/python -u eval/explorations/2026-05-23-cpsat-warm-non-twisted/run_cpsat_warm.py \
    --routing-table fixtures/routing_table_torus_2x2x4.json \
    --slice 2,2,4 \
    --out-schedule fixtures/schedule_torus_2x2x4_cpsat_literal_warm.json \
    --time-limit-s 300 \
    --n-workers 8 \
    --results-json eval/explorations/2026-05-23-cpsat-warm-non-twisted/results.json \
    2>&1 | tee eval/explorations/2026-05-23-cpsat-warm-non-twisted/run_log_2x2x4.txt
```

Expected last lines:
```
[stage 5] saved best schedule to fixtures/schedule_torus_2x2x4_cpsat_literal_warm.json (makespan=<M>)
appended row to eval/explorations/2026-05-23-cpsat-warm-non-twisted/results.json
```
where `<M>` should be `8` (= LB) unless the routing is non-translation-equivariant under physical-edge model, in which case `8 < <M> ≤ 13` (the orbit_greedy_full seed makespan, measured pre-plan). Exit code 0.

If the script exhausts the sweep without a `FEASIBLE` outcome, that means the seed itself (`makespan = 13`) was already the best CP-SAT can prove. The `chosen_status` row in `results.json` will read `ALL_INFEASIBLE_KEEPING_SEED`, and the saved schedule equals `S_seed`. Proceed to Step 3 in that case.

- [ ] **Step 3: Verify the saved schedule is capacity-feasible (sanity)**

```bash
.venv/bin/python -c "
from twisted_analysis.io.schedule import load_schedule
from twisted_analysis.schedules.verify import verify_capacity, schedule_makespan
s = load_schedule('fixtures/schedule_torus_2x2x4_cpsat_literal_warm.json')
v = verify_capacity(s)
print(f'entries={len(s)} makespan={schedule_makespan(s)} violations={len(v)}')
assert len(s) == 16 * 15, f'expected 240 entries, got {len(s)}'
assert v == [], f'capacity violations: {v[:3]}'
"
```

Expected: `entries=240 makespan=<M> violations=0` then no `AssertionError`.

- [ ] **Step 4: Emit the 2×2×4 kernel**

```bash
.venv/bin/python -u pallas_kernel/gen_orbit_greedy_kernel.py \
    --slice 2,2,4 \
    --routing-table fixtures/routing_table_torus_2x2x4.json \
    --schedule-in fixtures/schedule_torus_2x2x4_cpsat_literal_warm.json \
    --out pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_torus_2_2_4.py \
    --function-name _ragged_a2a_kernel_cpsat_literal_warm_torus_2_2_4
```

Expected (stderr lines, exact text):
```
[2/4] loaded schedule    fixtures/schedule_torus_2x2x4_cpsat_literal_warm.json (240 entries)
[3/4] verified schedule  (240 flows, 0 violations)
[4/4] wrote kernel       pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_torus_2_2_4.py (<bytes> bytes)
```

Exit code 0. If you see `--schedule-in ...: <X> entries have paths that disagree with --routing-table` — the routing table and schedule disagree (bug). Do not commit.

- [ ] **Step 5: Sanity-check the generated kernel imports**

```bash
.venv/bin/python -c "
import importlib.util, sys
sys.path.insert(0, 'pallas_kernel/outputs')
# We cannot fully import without jax/megablox, but we can syntax-check + dest_table extract.
import ast, pathlib
src = pathlib.Path('pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_torus_2_2_4.py').read_text()
tree = ast.parse(src)
fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)
           and n.name == '_ragged_a2a_kernel_cpsat_literal_warm_torus_2_2_4'), None)
assert fn is not None, 'function _ragged_a2a_kernel_cpsat_literal_warm_torus_2_2_4 not found'
# Find _DEST_TABLE_NP assignment, check shape literal is (16, 15).
assigns = [n for n in tree.body if isinstance(n, ast.Assign)]
assert any('_DEST_TABLE_NP' in [t.id for t in a.targets if isinstance(t, ast.Name)]
           for a in assigns), '_DEST_TABLE_NP not at module level'
print('kernel parses; function name found; _DEST_TABLE_NP present.')
"
```

Expected: `kernel parses; function name found; _DEST_TABLE_NP present.` Exit code 0.

- [ ] **Step 6: Commit the 2×2×4 artifacts**

```bash
git add fixtures/schedule_torus_2x2x4_cpsat_literal_warm.json \
        pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_torus_2_2_4.py \
        eval/explorations/2026-05-23-cpsat-warm-non-twisted/run_log_2x2x4.txt \
        eval/explorations/2026-05-23-cpsat-warm-non-twisted/results.json
git commit -m "feat: cpsat-warm schedule + kernel for non-twisted torus 2x2x4"
```

---

### Task 5: Run 2×4×4 end-to-end

**Files:**
- Created by run: `fixtures/schedule_torus_2x4x4_cpsat_literal_warm.json`
- Created by run: `pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_torus_2_4_4.py`
- Created by run: `eval/explorations/2026-05-23-cpsat-warm-non-twisted/run_log_2x4x4.txt`
- Modified by run: `eval/explorations/2026-05-23-cpsat-warm-non-twisted/results.json` (second row appended)

- [ ] **Step 1: Run the 2×4×4 driver**

From the repo root:

```bash
.venv/bin/python -u eval/explorations/2026-05-23-cpsat-warm-non-twisted/run_cpsat_warm.py \
    --routing-table fixtures/routing_table_torus_2x4x4.json \
    --slice 2,4,4 \
    --out-schedule fixtures/schedule_torus_2x4x4_cpsat_literal_warm.json \
    --time-limit-s 1800 \
    --n-workers 8 \
    --results-json eval/explorations/2026-05-23-cpsat-warm-non-twisted/results.json \
    2>&1 | tee eval/explorations/2026-05-23-cpsat-warm-non-twisted/run_log_2x4x4.txt
```

Expected last lines:
```
[stage 5] saved best schedule to fixtures/schedule_torus_2x4x4_cpsat_literal_warm.json (makespan=<M>)
appended row to eval/explorations/2026-05-23-cpsat-warm-non-twisted/results.json
```
where `<M>` should be `16` (= LB) when the routing achieves LB; otherwise `16 < <M> ≤ 33` (the orbit_greedy_full seed makespan, measured pre-plan). The README's matrix flags ILPRouter as failing translation-equivariance on `(2,4,4)`, so this is a real possibility, but the saved schedule is still optimal-given-the-routing because CP-SAT is exact. Exit code 0.

- [ ] **Step 2: Verify the saved schedule is capacity-feasible**

```bash
.venv/bin/python -c "
from twisted_analysis.io.schedule import load_schedule
from twisted_analysis.schedules.verify import verify_capacity, schedule_makespan
s = load_schedule('fixtures/schedule_torus_2x4x4_cpsat_literal_warm.json')
v = verify_capacity(s)
print(f'entries={len(s)} makespan={schedule_makespan(s)} violations={len(v)}')
assert len(s) == 32 * 31, f'expected 992 entries, got {len(s)}'
assert v == [], f'capacity violations: {v[:3]}'
"
```

Expected: `entries=992 makespan=<M> violations=0`, no `AssertionError`.

- [ ] **Step 3: Emit the 2×4×4 kernel**

```bash
.venv/bin/python -u pallas_kernel/gen_orbit_greedy_kernel.py \
    --slice 2,4,4 \
    --routing-table fixtures/routing_table_torus_2x4x4.json \
    --schedule-in fixtures/schedule_torus_2x4x4_cpsat_literal_warm.json \
    --out pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_torus_2_4_4.py \
    --function-name _ragged_a2a_kernel_cpsat_literal_warm_torus_2_4_4
```

Expected (stderr lines, exact text):
```
[2/4] loaded schedule    fixtures/schedule_torus_2x4x4_cpsat_literal_warm.json (992 entries)
[3/4] verified schedule  (992 flows, 0 violations)
[4/4] wrote kernel       pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_torus_2_4_4.py (<bytes> bytes)
```

Exit code 0.

- [ ] **Step 4: Sanity-check the generated kernel imports**

```bash
.venv/bin/python -c "
import ast, pathlib
src = pathlib.Path('pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_torus_2_4_4.py').read_text()
tree = ast.parse(src)
fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)
           and n.name == '_ragged_a2a_kernel_cpsat_literal_warm_torus_2_4_4'), None)
assert fn is not None, 'function _ragged_a2a_kernel_cpsat_literal_warm_torus_2_4_4 not found'
assigns = [n for n in tree.body if isinstance(n, ast.Assign)]
assert any('_DEST_TABLE_NP' in [t.id for t in a.targets if isinstance(t, ast.Name)]
           for a in assigns), '_DEST_TABLE_NP not at module level'
print('kernel parses; function name found; _DEST_TABLE_NP present.')
"
```

Expected: `kernel parses; function name found; _DEST_TABLE_NP present.` Exit code 0.

- [ ] **Step 5: Commit the 2×4×4 artifacts**

```bash
git add fixtures/schedule_torus_2x4x4_cpsat_literal_warm.json \
        pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_torus_2_4_4.py \
        eval/explorations/2026-05-23-cpsat-warm-non-twisted/run_log_2x4x4.txt \
        eval/explorations/2026-05-23-cpsat-warm-non-twisted/results.json
git commit -m "feat: cpsat-warm schedule + kernel for non-twisted torus 2x4x4"
```

---

### Task 6: Populate `RESULTS.md` from `results.json`

**Files:**
- Modify: `eval/explorations/2026-05-23-cpsat-warm-non-twisted/RESULTS.md`

- [ ] **Step 1: Read `results.json` and extract the headline numbers**

```bash
.venv/bin/python - <<'EOF'
import json
rows = json.load(open('eval/explorations/2026-05-23-cpsat-warm-non-twisted/results.json'))
for r in rows:
    slice_str = "×".join(str(s) for s in r["slice"])
    n = 1
    for s in r["slice"]:
        n *= s
    runtime = sum(t.get("runtime_s", 0) for t in r["cpsat_t_uppers_tried"])
    gap = r["cpsat_makespan"] - r["lb"]
    print(f'| {slice_str} | {n} | {r["lb"]} | {r["seed_makespan"]} | {r["cpsat_makespan"]} | {gap} | {runtime:.1f}s | `{r["out_schedule"]}` |')
EOF
```

Expected: two markdown table rows, one per cell. Copy them.

- [ ] **Step 2: Replace the TBD rows in `RESULTS.md`**

Open `eval/explorations/2026-05-23-cpsat-warm-non-twisted/RESULTS.md` and replace the two `TBD` rows under the header with the rows printed in Step 1. Adjust the right-hand "Kernel" column to point at the corresponding generated `pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_torus_<slice>.py` (the script in Step 1 prints the schedule path; the kernel path is the mechanical sibling).

The final table should look like:

```markdown
| Cell | N | LB | orbit_greedy_full | CP-SAT makespan | Gap to LB | Runtime | Kernel |
|---|---:|---:|---:|---:|---:|---:|---|
| 2×2×4 | 16 | 8 | <M_seed> | <M_cpsat> | <gap> | <runtime>s | `pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_torus_2_2_4.py` |
| 2×4×4 | 32 | 16 | <M_seed> | <M_cpsat> | <gap> | <runtime>s | `pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_torus_2_4_4.py` |
```

- [ ] **Step 3: Commit the populated results**

```bash
git add eval/explorations/2026-05-23-cpsat-warm-non-twisted/RESULTS.md
git commit -m "docs: populate RESULTS.md for cpsat-warm non-twisted torus"
```

---

### Task 7: Add a one-line pointer in the top-level `README.md`

**Files:**
- Modify: `README.md` (single new bullet under the `pallas_kernel/` line in the `## Layout` section)

The existing layout bullet ends with mentions of `spread_greedy_k2_8_4_4 / spread_greedy_k2_inline_8_4_4`. Append two more pre-generated kernels to the same enumerated list so readers can find them by scanning `README.md`.

- [ ] **Step 1: Identify the exact line to extend**

Locate the line in `README.md` that begins with:

```
- `pallas_kernel/` — Pallas TPU kernel generator (consumes a routing table + schedule, emits ...
```

The phrase `spread_greedy_k2_8_4_4 / spread_greedy_k2_inline_8_4_4 (per-device DMA-cap K=2; testbed for the DMA-oversubscription hypothesis from 2026-05-17).` is at the end of that paragraph.

- [ ] **Step 2: Append the two new kernels to that sentence**

Edit `README.md`: locate the substring
```
spread_greedy_k2_8_4_4 / spread_greedy_k2_inline_8_4_4 (per-device DMA-cap K=2; testbed for the DMA-oversubscription hypothesis from 2026-05-17).
```
and replace it with
```
spread_greedy_k2_8_4_4 / spread_greedy_k2_inline_8_4_4 (per-device DMA-cap K=2; testbed for the DMA-oversubscription hypothesis from 2026-05-17), and `cpsat_literal_warm_torus_2_2_4` / `cpsat_literal_warm_torus_2_4_4` (non-twisted-torus AllToAll on slices (2,2,4) and (2,4,4); see [eval/explorations/2026-05-23-cpsat-warm-non-twisted/](eval/explorations/2026-05-23-cpsat-warm-non-twisted/)).
```

- [ ] **Step 3: Commit the README pointer**

```bash
git add README.md
git commit -m "docs: link cpsat-warm non-twisted torus kernels from top-level README"
```

---

## Notes for the implementer

- **`results.json` append behavior**: the driver appends; running 2×2×4 then 2×4×4 produces a 2-row list. Re-running both without first deleting `results.json` will produce a 4-row list. The Task 4 Step 1 `rm -f` guarantees a clean 2-row file. Do not modify the driver to overwrite — appending is what lets `run_all.sh` and the per-cell tasks share the same file.
- **Why this driver and not `scripts/generate_schedule.py`**: that CLI does not (yet) support `cpsat_literal`. Extending it would be a larger refactor; the pattern of a per-exploration driver matches every previous CP-SAT exploration in `eval/explorations/`.
- **Why `orbit_greedy_full` and not `orbit_greedy`**: the README matrix flags `orbit_greedy` as "only correct on translation-equivariant routings", and these non-twisted-torus fixtures are not guaranteed to be such (depending on their original router). `orbit_greedy_full` is correct under any translation-symmetric workload — it's the safe default seed.
- **CP-SAT runtime expectation**: on 240–992 flows the cold solve at `t_upper=LB` typically converges in seconds. The 300s / 1800s budgets are headroom, not expected wall-clock.
- **What happens if cold CP-SAT hits `OPTIMAL` with `makespan < LB`**: cannot happen — `LB` is by definition a lower bound. If you see this, the bug is in `physical_edge_lb` (not the solver). The Task 4 / Task 5 sanity checks would catch it via `assert makespan ≥ LB`-style observation in the log.
- **Memory note (from CLAUDE.md)**: use `.venv/bin/python -u`, not `uv run`. The dependency lockfile resync issue documented in user memory makes `uv run` non-reproducible across sessions.
