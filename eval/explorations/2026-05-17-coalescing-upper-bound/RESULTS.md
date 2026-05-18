# Results: Coalescing upper-bound diagnostic (Option D)

## Background

The makespan-78 cpsat_literal_warm Pallas kernel measured 132764 gbps on TPU
v5e, ~1.3% below the P2P reference (134541 gbps) despite a simulator projection
of +7.5%. Per-DMA scalar setup cost is the dominant bottleneck (inline-kernel
datapoint = -50% throughput). The only way to reduce DMA count under fixed
routing + no multi-dest DMA is per-edge coalescing of per-hop DMAs. This probe
measures the coalescing factor available.

## Phase 1: Direct measurement on shipped schedules

| Schedule | Makespan | n_flows | avg_hops | Uncoalesced DMAs | Coalesced DMAs | Coalescing Factor | Break-Even | Headroom |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpsat_literal_warm | 78 | 16256 | 3.465 | 56320 | 3822 | 14.736 | 3.465 | +11.271 |
| spread_greedy_k2 | 92 | 16256 | 3.465 | 56320 | 7369 | 7.643 | 3.465 | +4.178 |
| literal_greedy_lpt | 87 | 16256 | 3.465 | 56320 | 5024 | 11.210 | 3.465 | +7.746 |
| orbit_greedy_full | 85 | 16256 | 3.465 | 56320 | 5536 | 10.173 | 3.465 | +6.709 |

**Interpretation:** Per-DMA cost is the binding constraint on TPU. The "current"
kernel issues 16,256 single-DMA-per-flow descriptors. Switching to a per-hop
kernel with per-edge coalescing would issue `coalesced_descriptors` instead.
The kernel-switch is worthwhile iff `coalesced_descriptors < 16,256`, i.e.,
`coalescing_factor > avg_hops` (= break_even).

**Production schedule (`cpsat_literal_warm`, makespan 78):**
- Uncoalesced (per-hop) descriptors: 56,320
- Coalesced descriptors: 3,822
- Coalescing factor: 14.736
- Break-even threshold (= avg_hops): 3.465
- Headroom above break-even: +11.271
- **DMA-count reduction vs current single-DMA-per-flow kernel: 16,256 / 3,822 = 4.25× fewer descriptors**

The plan's original decision rule used coalescing-factor thresholds (e.g.,
cf < 1.5 → stop) under an assumed avg_hops ≈ 2.0. Actual avg_hops = 3.465 for
this routing, so the correct decision metric is **headroom** (= cf − avg_hops),
not raw cf. By that metric:
- headroom < 0 → STOP, Option 3 dead
- 0 ≤ headroom < 1 → marginal, Phase 2 needed to confirm
- headroom ≥ 1 → strong signal; run Phase 2 to find true upper bound

`cpsat_literal_warm` headroom = +11.271 — strong signal, well above any
reasonable threshold. The OTHER three schedules all also show headroom > +4,
suggesting this is structural (the routing produces highly clusterable edge
usage), not a quirk of one scheduler.

## Phase 1 decision: RUN PHASE 2

Phase 1 gives only a LOWER bound on the true upper bound (Phase 1 measures
an existing schedule; Phase 2 CP-SAT chooses the schedule to maximize
coalescing). Since Phase 1's lower bound is already 14.736 (factor of 4.25×
reduction below the current kernel), Phase 2 will either confirm this or
push it higher. Proceeding to Task 5 (CP-SAT coalescing model) and Task 6
(real-problem run).
