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

## Phase 3 (run early, before Phase 2): Local-Search Repair

Hill-climbing on round assignments — for each makespan-defining flow, try shifting it earlier. Applied to every available seed schedule (Task 9).

| Seed | Makespan In | Makespan Out | Δ | Runtime |
|---|---:|---:|---:|---:|
| `schedule_8x4x4_loaded_orbit_greedy_full_lpt_tail_asc.json` | 85 | 85 | 0 | 0.3s |
| `schedule_8x4x4_loaded_literal_greedy_lpt.json` | 87 | 87 | 0 | 0.1s |
| `02_best_random_shuffle_schedule.json` | 86 | 86 | 0 | 0.3s |
| `_seed_orbit_greedy_full_lpt.json` (Phase 1 best) | 84 | 84 | 0 | 0.5s |

**Outcome:** **Local search delivers zero improvement on every seed.** All greedy-produced schedules are local optima under the shift-earlier move set: every makespan-defining flow has its round set to the earliest physical-edge-feasible time given the other flows. To improve, you would need a richer move set (swap moves, accept-worse simulated annealing) or a structurally different starting schedule. This is itself an informative negative result: the 84 → 75 (LB) gap is **structural**, not refinable by local moves.

CP-SAT probe (Phase 2a) results below will determine whether the structural gap can be closed.
