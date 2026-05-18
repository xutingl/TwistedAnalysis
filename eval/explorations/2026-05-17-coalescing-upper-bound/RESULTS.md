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

## Phase 2: CP-SAT upper-bound probe (1 h budget)

| Metric | Value |
|---|---:|
| Solver status | `UNKNOWN` (no incumbent found) |
| Wall-clock budget | 3600 s |
| Wall-clock used | 3632.6 s |
| Model size | 1,346,944 booleans (1.27M y-vars + ~60k a-vars + ~60k b-vars) |
| LP iterations | 24,163 |
| Conflicts during search | 0 |
| Solver `best_bound` (loose LP lower-bound on objective) | 188 |

The cold-start CP-SAT model did not find any feasible incumbent within the
1-hour budget. The 0-conflicts / 24163-LP-iterations / 1.35M-boolean profile
indicates the LP relaxation alone consumed the entire wall-clock; the search
never reached a state where it could backtrack and learn from feasibility
constraints.

**Bound implication.** Because the existing Phase 1 schedule
(`cpsat_literal_warm`, makespan 78, coalescing factor 14.736) is itself a
feasible solution to the Phase 2 model, the true upper bound on the
coalescing factor is **≥ 14.736**. Phase 2 was unable to tighten this bound
within the budget — but it cannot weaken it either. Any future warm-started
or model-reduced variant of Phase 2 would only push the bound higher.

**Why this still answers the original question.** Even with the Phase 1
value (≥ 14.736) as the conservative lower bound on the upper bound, the
headroom above break-even (= 14.736 − 3.465 = +11.271) is far above the
"strong signal" threshold of +1.0. The decision does not depend on the
precise upper bound.

## Recommendation: OPTION 3 HAS LARGE HEADROOM

The coalescing-factor lower bound (= 14.736 from the Phase 1 production
schedule) exceeds the break-even threshold (= avg_hops = 3.465) by **4.25×**.
Translated to absolute DMA counts: switching from the current single-DMA-
per-flow kernel (16,256 descriptors) to a per-hop kernel with cross-round
same-edge coalescing would issue ≤ 3,822 descriptors — a **4.25× reduction**
in total DMA descriptor count for the production makespan-78 schedule. The
other three shipped schedulers all show headroom ≥ +4.1, confirming this is
a structural property of the loaded routing rather than a quirk of
`cpsat_literal`.

**Recommend prototyping Option B** (per-hop Pallas kernel with cross-round
same-edge coalescing) on the existing `cpsat_literal_warm` makespan-78
schedule:

1. **Step 1 (cheap pre-flight, ~½ day):** for each physical edge, dump the
   list of (round, source-flow) tuples from the makespan-78 schedule and
   compute the cross-round-coalesced descriptor list (using the same logic
   as `descriptor_counter.py`). Confirm that the per-edge maximal-run
   distribution has a fat tail (a small fraction of edges carrying most of
   the descriptors) — that's where coalescing pays off most. Histogram the
   payload sizes that each coalesced descriptor would carry; verify none
   exceed the 2^17 VMEM/pipeline-collapse threshold observed in the TPU
   measurements (current per-DMA payload = 32 KB; coalescing 2–4 same-edge
   DMAs across rounds = 64 KB–128 KB, safely below 2^17 = 128 KB at the
   boundary).

2. **Step 2 (kernel prototype, ~1 week):** modify one Pallas kernel
   (suggest starting from `_ragged_a2a_kernel_cpsat_literal_warm_8_4_4.py`)
   to issue per-hop coalesced descriptors. The intermediate-device scatter +
   forward logic is the new engineering risk; budget time for VMEM staging
   and semaphore plumbing.

3. **Step 3 (TPU measurement):** if the prototype outperforms the current
   132,764 gbps baseline by >5%, scale out to the other schedules. If it
   underperforms or matches, the per-DMA cost is NOT the dominant
   constraint and Option 3 should be deprioritized in favor of profiling-
   led optimizations (XLA/TPU profiler, packet-size sweep at 2^13/2^14/2^16,
   scalar-overhead reduction).

**Caveats**:

- The 4.25× DMA-count reduction is a theoretical maximum under the
  simulator's "1 round = 1 wall-clock unit" model. Real wall-clock impact
  depends on the fraction of per-DMA cost that is true scalar-dispatch
  overhead vs payload-bandwidth-bound. The inline-kernel datapoint
  (50% throughput) suggests scalar overhead is substantial, so the
  translation is likely material — but not 1-to-1.
- Coalescing increases per-descriptor payload size by up to 4× (when
  fusing 4 adjacent same-edge rounds). The 2^17-packet-size regression
  observed previously implies the upper safe payload is ~2× current
  32 KB. So a per-hop kernel can safely fuse 2–4 adjacent rounds per
  edge but NOT arbitrary numbers — Phase 1's 14.7× coalescing factor
  may not all be achievable in practice once VMEM/pipeline constraints
  are added back in. A realistic achievable gain is probably 2–3×
  rather than the theoretical 4.25×.
- The cpsat_literal_warm schedule already produces the highest coalescing
  factor (14.7×); the other schedulers' factors are lower because they
  spread flows across more edges. If a per-hop coalescing kernel is
  shipped, sticking with the cpsat_literal_warm makespan-78 schedule
  is the right pairing.
