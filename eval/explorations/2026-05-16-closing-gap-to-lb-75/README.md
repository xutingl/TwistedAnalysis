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

## Approach (3 probes)

1. [01_cpsat_warm_start_probe.py](01_cpsat_warm_start_probe.py) —
   CP-SAT at `t_upper ∈ {79, 78, 77, 76}` with **4h per probe**,
   warm-started from `fixtures/schedule_8x4x4_loaded_cpsat_literal.json`.
   Hypothesis: the 80 → 76 search space was hard cold because CP-SAT
   restarted from scratch at each `t_upper`. Warm-starting from a
   feasible incumbent at 80 (most variables hinted) should let CP-SAT
   focus on the few rounds at the makespan tail.

2. [02_lns_probe.py](02_lns_probe.py) — Large-Neighborhood Search at
   `destroy_size_frac=0.05`, 5-min per subproblem, 100 iters: destroy
   5% of the seed schedule (time-window / random-subset /
   makespan-bottleneck flows) and ask CP-SAT to re-optimize the
   subproblem with the rest pinned. Hypothesis: the previous-exploration
   local-search dead-end (every seed at a local optimum under shift-earlier
   moves) is escapable by simultaneously re-assigning a connected subset.

3. [02b_lns_aggressive.py](02b_lns_aggressive.py) — Same LNS driver with
   `destroy_size_frac=0.30`, 10-min per subproblem, 24 iters. Added after
   Probe 2 showed every 5%-destroy subproblem was instantly INFEASIBLE.
   Hypothesis: bigger destroys give CP-SAT more degrees of freedom to find
   alternative schedules. Diagnostic: per-iteration `result` field
   classifies failures (infeasible vs timeout_no_incumbent vs
   pinning_conflict).

## Compute budget

Multi-day. Probe 1: 4 × 4h = 16h sequential. Probe 2: 100 iters × ~5 min
per subproblem (budget; actual was much faster). Probe 2b: 24 iters × ~10 min
per subproblem (budget; actual was much faster). Total compute spent: ~16h
for Probe 1 + a few minutes for both LNS probes (subproblems returned
INFEASIBLE in seconds rather than reaching the 5-/10-min budget).

## Outcome

**Best schedule: makespan 78** via warm-started CP-SAT at `t_upper=79`.
Promoted to fixtures as
[`fixtures/schedule_8x4x4_loaded_cpsat_literal_warm.json`](../../../fixtures/schedule_8x4x4_loaded_cpsat_literal_warm.json)
and
[`fixtures/cns_schedules/schedule_cpsatliteralwarm_4x4x8_twisted.json`](../../../fixtures/cns_schedules/schedule_cpsatliteralwarm_4x4x8_twisted.json).
Pallas kernel at [best_pallas_kernel.py](best_pallas_kernel.py); raw
schedule at [best_schedule.json](best_schedule.json). 16256 flows, 0
physical-edge capacity violations.

Projects to **~144607 gbps** (linear scaling from `orbit_greedy_full[lpt_tail_asc]`
at makespan 85 → 132758 gbps), **+7.5% above P2P's measured 134541 gbps**
on TPU v5e. The prior exploration's makespan 80 projected to +4.8%; this
exploration adds another **2.7 percentage points**.

**What each phase taught us:**

1. **Probe 1 (warm-start CP-SAT).** Warm-start strictly helped at the
   `t_upper=79` point not tried cold in the prior exploration: makespan
   78 was found in 4 h from a makespan-80 seed. At `t_upper ∈ {77, 76}`,
   even warm-start fails — CP-SAT cannot find any feasible incumbent in 4 h.
   (Bug fix in commit `ed566b2`: `CpSolverSolutionCallback` extracts
   UNKNOWN-status incumbents reliably; no impact on this exploration but
   correct per OR-Tools docs.)

2. **Probe 2 (LNS @5% destroy).** Every single subproblem proved
   INFEASIBLE in 2–4 seconds. At 5% destroy (813 flows), the pinned 95%
   already saturate the makespan-77 budget.

3. **Probe 2b (LNS @30% destroy, 10-min subproblems).** Same story even
   at 30% destroy. Every subproblem PROVES infeasibility in 3–11 seconds —
   not a timeout, an actual infeasibility proof. The makespan-78 schedule
   is structurally tight; the pinned majority dominates the edge-time
   budget at any reasonable destroy fraction.

**Is LB=75 achievable?** Open. The negative LNS findings suggest local
escape from makespan 78 is structurally blocked. Possible avenues:

- Cold CP-SAT at `t_upper ∈ {77, 76, 75}` with **much** longer budget
  (24–72 h per probe) and/or more workers.
- Multiple cold-restart CP-SAT with different RNG seeds; the parallel search
  trajectory at 4 h appears insufficient at `t_upper < 78`.
- LP-relaxation lower bound on the makespan (rather than the physical-edge
  LB of 75): the LP bound observed in CP-SAT logs hovered at 46 throughout,
  suggesting the LP relaxation is very weak and the true minimum may be
  significantly above 75. A stronger LP solver (HiGHS, Gurobi) on the LP
  relaxation would quantify this.
- Column generation / Dantzig-Wolfe reformulation.

**Hardware-vs-simulator caveat.** The simulator omits per-step barrier
latency, HBM bandwidth contention, VC arbitration, and OCS-layer
interference. A sim-makespan of 78 projects to ~144607 gbps under perfectly
linear scaling, but the actual TPU v5e measurement could land anywhere
between 135 and 150 Kgbps. Empirical TPU comparison vs P2P is the natural
next step (out of scope for this exploration).

See [RESULTS.md](RESULTS.md) for the full per-probe table.
