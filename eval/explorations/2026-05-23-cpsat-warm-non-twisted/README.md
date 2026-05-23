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
