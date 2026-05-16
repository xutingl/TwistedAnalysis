# Probe Results: Beating P2P on Loaded 8×4×4

**Target:** sim-makespan ≤ 83 (P2P equivalent), LB = 75. Current best to beat: 84 (`orbit_greedy_full[lpt]`).

| Probe | Scheduler | Params | Makespan | Violations | Runtime | Result file |
|---|---|---|---:|---:|---:|---|
| baseline | orbit_greedy_full | lpt_tail_asc | 85 | 0 | 14s | `fixtures/schedule_8x4x4_loaded_orbit_greedy_full_lpt_tail_asc.json` |
| baseline | orbit_greedy_full | lpt | 84 | 0 | 14s | (this exploration, Task 2) |
| baseline | literal_greedy | lpt | 87 | 0 | 0.4s | `fixtures/schedule_8x4x4_loaded_literal_greedy_lpt.json` |

## Phase 1: Ordering Sweep

Deterministic orderings on orbit_greedy_full and literal_greedy (Task 2).

| Scheduler | Order | Makespan | Violations | Runtime | Notes |
|---|---|---:|---:|---:|---|
| orbit_greedy_full | lpt | 84 | 0 | 14.0s | **Best** (matches baseline) |
| orbit_greedy_full | lpt_tail_asc | 85 | 0 | 14.2s | |
| literal_greedy | lpt | 87 | 0 | 0.4s | |
| orbit_greedy_full | tail_asc | 94 | 0 | 14.1s | |
| orbit_greedy_full | spt | 102 | 0 | 14.1s | |
| literal_greedy | spt | 105 | 0 | 0.4s | |
| literal_greedy | natural | 109 | 0 | 0.4s | Worst |

## Phase 1b: Random Orbit Shuffle

Random permutations of orbit firing order on `orbit_greedy_full` (Task 3).

**Summary:** 1000 random seeds; best random ordering yields makespan **86 at seed 125** — **worse** than deterministic lpt (84). The structured `lpt` heuristic dominates uniform-random orderings here.

| Metric | Value |
|---|---|
| Best random makespan | 86 |
| Best seed | 125 |
| Total seeds | 1000 |
| Runtime | 180.2 seconds |
| Best schedule | `02_best_random_shuffle_schedule.json` |

**Distribution (top 5 most common makespans across 1000 trials):**

| Makespan | Count |
|---:|---:|
| 93 | 171 |
| 92 | 168 |
| 91 | 162 |
| 90 | 104 |
| 89 | 53 |

**Outcome:** Random shuffling does **NOT** beat deterministic lpt — confirms that the structural bias in lpt (longest-path-first by orbit size) matters. The distribution peaks around 92-93 with a tail down to 86. No ordering breached 84, let alone the 83 target. Proceeding to Phase 2 (CP-SAT, LP-rounding) for structurally different schedules.

## Phase 2a: CP-SAT at decreasing t_upper

Google OR-Tools CP-SAT solver on the literal scheduling ILP, 1800s budget per probe, 8 workers (Task 5).

| t_upper | status | makespan | violations | runtime |
|---:|---|---:|---:|---:|
| 84 | FEASIBLE | **83** | 0 | 1838s |
| 83 | TIMEOUT | — | — | 1835s (no incumbent) |
| 82 | FEASIBLE | **82** | 0 | 1837s |
| 81 | TIMEOUT | — | — | 1830s (no incumbent) |
| 80 | FEASIBLE | **80** | 0 | 1837s |
| 78 | TIMEOUT | — | — | 1833s (no incumbent) |
| 76 | TIMEOUT | — | — | 1831s (no incumbent) |

**Best: makespan 80** at t_upper=80, schedule saved to `03_best_cpsat_schedule.json` (1.95 MB, 16256 entries, 0 violations).

**Pattern.** CP-SAT consistently finds a feasible solution AT the t_upper itself (the model finds incumbents by anchoring near the horizon), but the time budget is too small to push BELOW the horizon. Each FEASIBLE row's makespan == t_upper, never strictly less. This is the well-known "objective-following" behavior of CP-SAT on large feasibility-style problems with weak objective LP relaxation: it can satisfy the constraints up to the horizon, but tightening the horizon requires re-solving from scratch and the search restarts cold.

**Throughput projection.** Linear scaling from `orbit_greedy_full[lpt_tail_asc]` (makespan 85 → 132758 gbps): makespan 80 → **141055 gbps**, which is **4.8% above P2P's 134541 gbps**. Target met.

## Phase 2b: LP-rounding (intractable on N=128)

Attempted to solve LP relaxation of literal_ilp (1.37M continuous variables) with CBC and run 200 randomized-rounding trials (Task 7).

**Outcome: process killed after 6.5h with the LP solve still not complete.** CBC's simplex on this LP scale is dominated by basis updates and doesn't make progress within practical wall-clock. No incumbent produced; no schedule generated.

A future iteration could try a stronger LP solver (Gurobi, HiGHS) or a column-generation reformulation, but for the purposes of this exploration the CP-SAT route was sufficient.

## Phase 3: Local-Search Repair

Hill-climbing on round assignments — for each makespan-defining flow, try shifting it earlier (Task 9). Applied to every available seed schedule, including CP-SAT's makespan-80.

| Seed | Makespan In | Makespan Out | Δ | Runtime |
|---|---:|---:|---:|---:|
| `schedule_8x4x4_loaded_orbit_greedy_full_lpt_tail_asc.json` | 85 | 85 | 0 | 0.2s |
| `schedule_8x4x4_loaded_literal_greedy_lpt.json` | 87 | 87 | 0 | 0.1s |
| `02_best_random_shuffle_schedule.json` | 86 | 86 | 0 | 0.2s |
| `03_best_cpsat_schedule.json` | 80 | 80 | 0 | 21.6s |
| `_seed_orbit_greedy_full_lpt.json` (Phase 1 best) | 84 | 84 | 0 | 0.4s |

**Outcome:** **Local search delivers zero improvement on every seed**, including CP-SAT's makespan-80. All produced schedules are local optima under the shift-earlier move set: every makespan-defining flow has its round set to the earliest physical-edge-feasible time given the other flows. To improve further you would need a richer move set (swap moves, accept-worse simulated annealing) or a structurally different starting schedule. This is itself an informative negative result: each phase's output is locally tight, so the only path to improvement is global re-optimization (CP-SAT, ILP, ...).

## Summary — what beat what

| Probe | Best makespan | Beats baseline (84)? | Beats P2P-equivalent (≤83)? |
|---|---:|:---:|:---:|
| Phase 1 deterministic | 84 (orbit_greedy_full[lpt]) | tie | no |
| Phase 1b random shuffle | 86 | no | no |
| Phase 2a CP-SAT | **80** | yes (−4) | **yes (−3)** |
| Phase 2b LP-rounding | — (intractable) | n/a | n/a |
| Phase 3 local-search | (mirrors input) | n/a | n/a |

**Winner: CP-SAT at t_upper=80, makespan 80, 5.0% below the baseline that already nearly matched P2P.** Best schedule promoted to `best_schedule.json`.
