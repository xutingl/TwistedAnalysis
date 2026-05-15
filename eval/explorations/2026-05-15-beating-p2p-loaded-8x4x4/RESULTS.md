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
