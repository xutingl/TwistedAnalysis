# Coalescing upper-bound diagnostic (Option D)

## Problem

The makespan-78 cpsat_literal_warm Pallas kernel measured 132764 gbps on TPU v5e,
~1.3% below the P2P reference (134541 gbps) despite a simulator projection of
+7.5%. Three TPU datapoints (inline kernel = 50% throughput; K=2 spread_greedy
matches cpsat_warm; packet_size=2^17 = -16%) localize the bottleneck to
per-DMA scalar setup cost, with VMEM/pipeline parallelism saturated at ~4
in-flight × 32 KB.

The current Pallas kernel issues one DMA descriptor per logical (src, dst)
flow = 16,256 descriptors total; TPU hardware routes multi-hop transparently.
The only way to reduce descriptor count under fixed routing + no multi-dest
DMA is **per-edge coalescing**: switch the kernel to per-hop DMA structure
(expanding to ~32k descriptors at avg_hop=2), then fuse same-edge DMAs across
flows (in-round) or across adjacent rounds (cross-round) back to a smaller
count. Coalescing must exceed factor 2.0 just to break even with the current
single-DMA-per-flow kernel.

## Goal

Compute the theoretical maximum coalescing factor achievable on the loaded
8×4×4 twisted routing under fixed routing + makespan ≤ 78. If the upper
bound is < 2.0, Option 3 (data-layout coalescing under fixed routing) is
dead and we should not pursue kernel-level per-hop restructuring. If ≥ 3.0,
Option B (cross-round same-edge coalescing) is worth prototyping.

## Approach

### Phase 1: direct measurement on existing schedules (cheap)

For each shipped schedule (cpsat_literal_warm-78, spread_greedy_k2-92,
literal_greedy-87, orbit_greedy_full-85), expand flows into per-(edge, round)
contributions using `path`, then count both uncoalesced and coalesced
descriptors. Output per-schedule coalescing factors.

### Phase 2: CP-SAT re-scheduling for max coalescing (expensive, conditional)

Build a CP-SAT model with the same `y[f, s]` variables as `cpsat_literal` plus
edge-active booleans `a[(u,v), tau]` and edge-break booleans `b[(u,v), tau]`.
Constraints: each flow gets exactly one start round; edge capacity = 1 per
round; `a == sum of y contributing to that (edge, round)`; `b[edge, tau] = a[edge, tau] AND NOT a[edge, tau-1]`.
Objective: minimize `sum of b`. With `t_upper = 78`, this finds the
makespan-feasible schedule whose post-hoc coalescing factor is maximal.
Time budget: 1 hour with 8 workers. Reports best incumbent.

## Decision rules

(See plan: 2026-05-17-coalescing-upper-bound.md for decision-rule thresholds.)

## Compute budget

Phase 1: minutes. Phase 2: ~1 hour CP-SAT run, plus ~1 day of modeling work.
