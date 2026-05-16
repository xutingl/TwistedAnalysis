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

**Result: CP-SAT found a schedule with sim-makespan 80** — 5 steps below the previous best (`orbit_greedy_full[lpt_tail_asc]` at 85, also matched at 84 by `orbit_greedy_full[lpt]`). By linear throughput scaling from 132758 gbps @ makespan 85, this projects to **~141055 gbps**, **4.8% above the P2P reference's 134541 gbps**. The generated Pallas kernel is at [best_pallas_kernel.py](best_pallas_kernel.py), verified to have 16256 flows and 0 capacity violations. The schedule itself is at [best_schedule.json](best_schedule.json).

**What each phase taught us:**

1. **Phase 1 (deterministic orderings).** `orbit_greedy_full[lpt]` reaches makespan 84 — one step better than the previously-shipped default `lpt_tail_asc`. The tiebreak rule that helps on translation-equivariant routings actively hurts on the non-equivariant loaded routing. **Cheap fix: switch the default `order` for loaded-routing kernels from `lpt_tail_asc` to `lpt`.**
2. **Phase 1b (random orbit shuffles).** 1000 random orbit-firing orderings on `orbit_greedy_full`; best = 86, distribution centered around 92-93. **Random shuffling does NOT beat deterministic `lpt`.** The structural bias of `lpt` (longest-orbit-first) matters; uniform random discards it.
3. **Phase 2a (CP-SAT).** Reached makespan 80 with 5/7 t_upper probes successful (the other 2 timed out cold). Pattern: CP-SAT consistently finds incumbents AT the horizon, not below — each tighter t_upper restarts cold and re-pays warm-up cost. Adding a warm-start hint (planned future work) would likely close the remaining 80 → 75 (LB) gap.
4. **Phase 2b (LP-rounding).** Intractable on N=128. CBC's simplex on the 1.37M-variable LP relaxation did not complete within 6.5h. Could be revisited with HiGHS or Gurobi, or via column generation.
5. **Phase 3 (local-search repair).** Zero improvement on every seed including CP-SAT's makespan-80. All produced schedules are locally tight under the "shift earlier" move set; only global re-optimization can help further.

**Is LB=75 achievable?** Still open. CP-SAT timed out at t_upper ∈ {76, 78} with no incumbent — that's evidence the search space is hard, not proof of infeasibility. Future work: warm-start CP-SAT with the makespan-80 schedule (the easiest way to break through), or commit serious CP-SAT compute (4-8 hour budget per probe) at t_upper=76.

**Hardware-vs-simulator caveat.** The simulator omits per-step barrier latency, HBM bandwidth contention, VC arbitration, and OCS-layer interference. A sim-makespan of 80 projects to ~141055 gbps under perfectly linear scaling, but the actual TPU v5e measurement of this kernel could land anywhere between 130 and 145 Kgbps. Empirical TPU comparison vs P2P is the natural next step (out of scope for this exploration).

See [RESULTS.md](RESULTS.md) for the full per-phase table.
