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
