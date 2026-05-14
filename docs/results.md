# Results

## Headline Numbers (Eval Run, 2026-05-13)

Source: `results/2026-05-13/`

Column definitions:

- `router` — routing strategy: `ilp` (ILPRouter, load-balanced minimal) or `dor` (dimension-order).
- `sched` — schedule name.
- `sched-LB` — max link load over flows the schedule actually injects.
- `full-LB` — max link load over all `N·(N-1)` flows (headline lower bound).
- `flows` — coverage (flows injected / total AllToAll flows).
- `makespan` — realized steps to completion.
- `ratio` — `makespan / sched-LB`.

## DOR vs ILP Routing (Impact on LB)

For full-coverage schedules, `full-LB` equals `sched-LB`. ILPRouter lowers the max
link load by distributing traffic across all minimal paths.

| Topology | DOR LB | ILP LB | Reduction |
|----------|--------|--------|-----------|
| 2×4      | 4      | 3      | 25%       |
| 2×2×4    | 7      | 5      | 29%       |
| 2×4×4    | 16     | 11     | 31%       |
| 4×8      | 26     | 21     | 19%       |
| 4×4×8    | 86     | 74     | 14%       |

The 3D `{2, 2, 4}` and `{2, 4, 4}` topologies see the biggest LB reductions
(29–31%) — the DOR tie-breaking pattern interacts particularly badly with the
multi-twist structure of these shapes.

## Schedule Comparison (with ILP Routing)

| name                   | router | sched                 | sched-LB | full-LB | flows         | makespan | ratio |
|------------------------|--------|-----------------------|----------|---------|---------------|----------|-------|
| 2x4_dim_phased         | ilp    | dim_phased            | 3        | 3       | 32/56         | 3        | 1.00  |
| 2x4_ilp                | ilp    | ilp_optimal           | 3        | 3       | 56/56         | 3        | 1.00  |
| 2x4_ilp_symmetric      | ilp    | ilp_optimal_symmetric | 3        | 3       | 56/56         | 3        | 1.00  |
| 2x4_orbit_greedy       | ilp    | orbit_greedy          | 3        | 3       | 56/56         | 3        | 1.00  |
| 2x4_pipelined_orbit    | ilp    | pipelined_orbit       | 3        | 3       | 56/56         | 3        | 1.00  |
| 2x4_rr                 | ilp    | round_robin           | 3        | 3       | 56/56         | 13       | 4.33  |
| 2x4_xla                | ilp    | xla                   | 3        | 3       | 56/56         | 13       | 4.33  |
| 2x2x4_dim_phased       | ilp    | dim_phased            | 3        | 5       | 80/240        | 4        | 1.33  |
| 2x2x4_ilp              | ilp    | ilp_optimal           | 5        | 5       | 240/240       | 5        | 1.00  |
| 2x2x4_ilp_symmetric    | ilp    | ilp_optimal_symmetric | 5        | 5       | 240/240       | 5        | 1.00  |
| 2x2x4_orbit_greedy     | ilp    | orbit_greedy          | 5        | 5       | 240/240       | 5        | 1.00  |
| 2x2x4_pipelined_orbit  | ilp    | pipelined_orbit       | 5        | 5       | 240/240       | 5        | 1.00  |
| 2x2x4_rr               | ilp    | round_robin           | 5        | 5       | 240/240       | 37       | 7.40  |
| 2x2x4_xla              | ilp    | xla                   | 5        | 5       | 240/240       | 37       | 7.40  |
| 2x4x4_dim_phased       | ilp    | dim_phased            | 3        | 11      | 224/992       | 9        | 3.00  |
| 2x4x4_ilp_symmetric    | ilp    | ilp_optimal_symmetric | 11       | 11      | 992/992       | 11       | 1.00  |
| 2x4x4_orbit_greedy     | ilp    | orbit_greedy          | 11       | 11      | 992/992       | 11       | 1.00  |
| 2x4x4_pipelined_orbit  | ilp    | pipelined_orbit       | 11       | 11      | 992/992       | 12       | 1.09  |
| 2x4x4_rr               | ilp    | round_robin           | 11       | 11      | 992/992       | 88       | 8.00  |
| 2x4x4_xla              | ilp    | xla                   | 11       | 11      | 992/992       | 88       | 8.00  |
| 4x8_dim_phased         | ilp    | dim_phased            | 8        | 21      | 320/992       | 13       | 1.62  |
| 4x8_ilp_symmetric      | ilp    | ilp_optimal_symmetric | 21       | 21      | 992/992       | 21       | 1.00  |
| 4x8_orbit_greedy       | ilp    | orbit_greedy          | 21       | 21      | 992/992       | 21       | 1.00  |
| 4x8_pipelined_orbit    | ilp    | pipelined_orbit       | 21       | 21      | 992/992       | 22       | 1.05  |
| 4x8_rr                 | ilp    | round_robin           | 21       | 21      | 992/992       | 122      | 5.81  |
| 4x8_xla                | ilp    | xla                   | 21       | 21      | 992/992       | 122      | 5.81  |
| 4x4x8_dim_phased       | ilp    | dim_phased            | 8        | 74      | 1664/16256    | 18       | 2.25  |
| 4x4x8_ilp_symmetric    | ilp    | ilp_optimal_symmetric | 74       | 74      | 16256/16256   | 74       | 1.00  |
| 4x4x8_orbit_greedy     | ilp    | orbit_greedy          | 74       | 74      | 16256/16256   | 74       | 1.00  |
| 4x4x8_pipelined_orbit  | ilp    | pipelined_orbit       | 74       | 74      | 16256/16256   | 79       | 1.07  |
| 4x4x8_rr               | ilp    | round_robin           | 74       | 74      | 16256/16256   | 637      | 8.61  |
| 4x4x8_xla              | ilp    | xla                   | 74       | 74      | 16256/16256   | 637      | 8.61  |

## DOR Routing Reference

| name                       | router | sched                 | sched-LB | full-LB | flows         | makespan | ratio |
|----------------------------|--------|-----------------------|----------|---------|---------------|----------|-------|
| 2x4_dim_phased_dor         | dor    | dim_phased            | 3        | 4       | 32/56         | 5        | 1.67  |
| 2x4_ilp_dor                | dor    | ilp_optimal           | 4        | 4       | 56/56         | 4        | 1.00  |
| 2x4_orbit_greedy_dor       | dor    | orbit_greedy          | 4        | 4       | 56/56         | 4        | 1.00  |
| 2x4_pipelined_orbit_dor    | dor    | pipelined_orbit       | 4        | 4       | 56/56         | 4        | 1.00  |
| 2x4_rr_dor                 | dor    | round_robin           | 4        | 4       | 56/56         | 13       | 3.25  |
| 2x4_xla_dor                | dor    | xla                   | 4        | 4       | 56/56         | 13       | 3.25  |
| 2x2x4_dim_phased_dor       | dor    | dim_phased            | 3        | 7       | 80/240        | 6        | 2.00  |
| 2x2x4_ilp_dor              | dor    | ilp_optimal           | 7        | 7       | 240/240       | 7        | 1.00  |
| 2x2x4_orbit_greedy_dor     | dor    | orbit_greedy          | 7        | 7       | 240/240       | 7        | 1.00  |
| 2x2x4_pipelined_orbit_dor  | dor    | pipelined_orbit       | 7        | 7       | 240/240       | 7        | 1.00  |
| 2x2x4_rr_dor               | dor    | round_robin           | 7        | 7       | 240/240       | 37       | 5.29  |
| 2x2x4_xla_dor              | dor    | xla                   | 7        | 7       | 240/240       | 37       | 5.29  |
| 2x4x4_dim_phased_dor       | dor    | dim_phased            | 3        | 16      | 224/992       | 9        | 3.00  |
| 2x4x4_orbit_greedy_dor     | dor    | orbit_greedy          | 16       | 16      | 992/992       | 16       | 1.00  |
| 2x4x4_pipelined_orbit_dor  | dor    | pipelined_orbit       | 16       | 16      | 992/992       | 16       | 1.00  |
| 2x4x4_rr_dor               | dor    | round_robin           | 16       | 16      | 992/992       | 90       | 5.63  |
| 2x4x4_xla_dor              | dor    | xla                   | 16       | 16      | 992/992       | 90       | 5.63  |
| 4x8_dim_phased_dor         | dor    | dim_phased            | 10       | 26      | 320/992       | 18       | 1.80  |
| 4x8_orbit_greedy_dor       | dor    | orbit_greedy          | 26       | 26      | 992/992       | 26       | 1.00  |
| 4x8_pipelined_orbit_dor    | dor    | pipelined_orbit       | 26       | 26      | 992/992       | 26       | 1.00  |
| 4x8_rr_dor                 | dor    | round_robin           | 26       | 26      | 992/992       | 116      | 4.46  |
| 4x8_xla_dor                | dor    | xla                   | 26       | 26      | 992/992       | 116      | 4.46  |
| 4x4x8_dim_phased_dor       | dor    | dim_phased            | 10       | 86      | 1664/16256    | 23       | 2.30  |
| 4x4x8_orbit_greedy_dor     | dor    | orbit_greedy          | 86       | 86      | 16256/16256   | 86       | 1.00  |
| 4x4x8_pipelined_orbit_dor  | dor    | pipelined_orbit       | 86       | 86      | 16256/16256   | 90       | 1.05  |
| 4x4x8_rr_dor               | dor    | round_robin           | 86       | 86      | 16256/16256   | 650      | 7.56  |
| 4x4x8_xla_dor              | dor    | xla                   | 86       | 86      | 16256/16256   | 650      | 7.56  |

## Key Findings

**0. OrbitGreedy (default `lpt_tail_asc` ordering) achieves `makespan = LB` on
all 10 (topology, router) cells — no ILP needed.**

| Topology | Router | LB | OrbitGreedy default | symmetric ILP |
|----------|--------|---:|---:|---:|
| 2×4   | ilp | 3  | **3 (1.00, μs)**  | 3 (1.00, 14s)  |
| 2×4   | dor | 4  | **4 (1.00, μs)**  | n/a            |
| 2×2×4 | ilp | 5  | **5 (1.00, μs)**  | **5 (1.00, ~1s)** ✓ |
| 2×2×4 | dor | 7  | **7 (1.00, μs)**  | n/a            |
| 2×4×4 | ilp | 11 | **11 (1.00, μs)** | **11 (1.00, ~1s)** ✓ |
| 2×4×4 | dor | 16 | **16 (1.00, μs)** ✓ | n/a          |
| 4×8   | ilp | 21 | **21 (1.00, μs)** | 21 (1.00, 14s) |
| 4×8   | dor | 26 | **26 (1.00, μs)** | n/a            |
| 4×4×8 | ilp | 74 | **74 (1.00, μs)** | **74 (1.00, ~6 min)** ✓ |
| 4×4×8 | dor | 86 | **86 (1.00, μs)** | n/a            |

The 2×4×4 DOR cell — which previously missed by 1 step under plain `lpt` ordering
— now hits LB exactly after we replaced the tie-breaker. The new default
`lpt_tail_asc` orders orbits by longest path first, then by ascending load of
the *tail edge orbit* (the `(dim, dir)` class of the orbit's final hop).
Orbits whose final hop lands on a low-load edge have few "exit slots"
available, so we schedule them first to claim a bottleneck slot early and
leave the corresponding exit slot still open. Orbits with flexible tails fill
in around them.

The 4×4×8 symmetric ILP cell — previously marked "intractable" — was actually
tractable in ~6 minutes once we seeded `T_upper = LB` (using OrbitGreedy's
result as an existence witness). The 2×2×4 and 2×4×4 symmetric ILPs solve
in ~1 s each. On every ILP-routed cell, the symmetric ILP oracle agrees with
the greedy's `makespan = LB`.

**Note: the previous 2×4×4 DOR gap (greedy=17, LB=16) was a scheduler bug, not
a topology limit.** Plain LPT ordering miss-scheduled two length-2 orbits
ending on the low-load (0,−1) edge: both got hop-0 on the bottleneck at t=14,
forcing one to roll to t=16 on the tail. The `lpt_tail_asc` ordering schedules
the (0,−)-tail orbits first (low tail load → claim early bottleneck slot),
producing makespan=16=LB on this cell. Tests
`test_orbit_greedy_default_achieves_lb` and
`test_orbit_greedy_lpt_misses_lb_on_2x4x4_dor` lock both behaviors.

OrbitGreedy is the recommended scheduler: agrees with the ILP oracle on every
solvable cell, microseconds vs minutes, no CBC dependency. Ordering matters:
shortest-path-first gives a 5–16% gap; **PipelinedOrbit** (force gap=1 per
orbit) gives 0–9% gap. See [schedules.md](schedules.md) for the algorithm and
a Leighton-Maggs-Rao bound discussion.

**1. ILP routing reduces LB by 14–31% across topologies.**
Distributing traffic across all minimal paths (ILPRouter) lowers the max-link load
versus DOR's lexicographic tie-breaking. The 3D `{2, 2, 4}` and `{2, 4, 4}`
shapes (added in this iteration) see the biggest reductions — DOR's tie-breaking
interacts poorly with multiple twisting dimensions.

| Topology | DOR LB | ILP LB | Reduction |
|----------|--------|--------|-----------|
| 2×4      | 4      | 3      | 25%       |
| 2×2×4    | 7      | 5      | 29%       |
| 2×4×4    | 16     | 11     | 31%       |
| 4×8      | 26     | 21     | 19%       |
| 4×4×8    | 86     | 74     | 14%       |

**2. Symmetric ILP scheduling achieves LB exactly on every ILP-routed cell.**
With ILP routing, the symmetric ILP scheduler produces `makespan = LB` on
2×4 (3), 2×2×4 (5), 2×4×4 (11), 4×8 (21), and 4×4×8 (74). Solve times range
from ~1 s (2×2×4, 2×4×4) to ~6 min (4×4×8 with `T_upper = LB` warm-start).
(As of finding #0, OrbitGreedy achieves the same zero gap in microseconds,
so the symmetric ILP is primarily a verification oracle now.)

**3. The previous 4×8 gap was a DOR artifact, not an intrinsic topology limit.**
Under DOR routing (LB = 26), the full ILP gives `makespan = 4 = LB` only on 2×4.
On 4×8, DOR's poor tie-breaking raises LB by 24% relative to ILP routing, making
the lower bound harder to achieve. Switching to ILP routing eliminates the gap.

**4. Round-robin is still far from optimal across all topologies.**

| Topology | Router | RR makespan | LB  | Ratio |
|----------|--------|-------------|-----|-------|
| 2×4      | ilp    | 13          | 3   | 4.33  |
| 2×2×4    | ilp    | 37          | 5   | 7.40  |
| 2×4×4    | ilp    | 88          | 11  | 8.00  |
| 4×8      | ilp    | 122         | 21  | 5.81  |
| 4×4×8    | ilp    | 637         | 74  | 8.61  |

The Latin-square round-robin schedule accumulates phase-makespan overhead across
`N-1` back-to-back phases. The gap is a **scheduling** inefficiency (phase-boundary
idle time), not a routing problem. (The 2×4×4 and 4×4×8 ratios near 8× and 9×
respectively show the gap grows with topology size.)

**5. DimPhased ratios improve with ILP routing.**
ILP routing lowers both the schedule LB and the realized makespan for DimPhased:

| Topology | DOR ratio | ILP ratio |
|----------|-----------|-----------|
| 2×4      | 1.67      | 1.00      |
| 2×2×4    | 2.00      | 1.33      |
| 2×4×4    | 3.00      | 3.00      |
| 4×8      | 1.80      | 1.62      |
| 4×4×8    | 2.30      | 2.25      |

DimPhased on 2×4 is now optimal (ratio = 1.00) under ILP routing. The 2×4×4
cell is unusual: DOR and ILP ratios are identical (3.00) — DimPhased's
per-dim phase makespan is dominated by the longest in-dim AllToAll which is
the same regardless of router choice. Recall that DimPhased is still
partial-coverage (see Caveats).

**6. XLA's destination-core randomization gives the same makespan as round-robin in our model.**

XLA's permutation formula `permute_idx = ((p · 33617 + 1299721) mod (N-1)) + 1`
is a bijection on `{1..N-1}` for our N values (N=8, 32, 128), because
`gcd(33617, N-1) = 1` holds in each case. As a result, the set of phase
workloads produced by XLA is exactly a permutation of round-robin's: both visit
the same set of `(src, dst)` displacements, just in a different order.

In our step-synchronous cost model, phases execute back-to-back and the total
makespan is the sum of per-phase makespans. Because per-phase makespan depends
only on the displacement `permute_idx` in that phase — not on the order phases
are visited — sum is commutative and both schedules accumulate identical
phase-boundary idle time. Empirically:

| Topology | RR makespan | XLA makespan |
|----------|-------------|--------------|
| 2×4 (ilp) | 13 | 13 |
| 2×2×4 (ilp) | 37 | 37 |
| 2×4×4 (ilp) | 88 | 88 |
| 4×8 (ilp) | 122 | 122 |
| 4×4×8 (ilp) | 637 | 637 |
| 2×4 (dor) | 13 | 13 |
| 2×2×4 (dor) | 37 | 37 |
| 2×4×4 (dor) | 90 | 90 |
| 4×8 (dor) | 116 | 116 |
| 4×4×8 (dor) | 650 | 650 |

XLA's randomization advantage over round-robin only manifests in cost models
with phase overlap (pipelined injection), real-hardware effects (cache/queue/hotspot
persistence), or background traffic — none of which we model. See
[schedules.md](schedules.md) for the XLA phase construction.

## Caveats

**LP relaxation is weak.** The LP relaxation (`lp_relax_lower_bound`) minimizes the
weighted-average completion time (sum of `(t+1) * x[u, last, t]`), not the
makespan. Its value can be below `LB`. It is not a tight makespan bound. See
[lp_formulation.md](lp_formulation.md) for the precise statement.

**DimPhased is not a full AllToAll — interpret with care.**
DimPhased covers only `(src, dst)` pairs that differ in exactly one dimension
(57% of flows on 2×4, 32% on 4×8, 10% on 4×4×8). The `Ratio` column uses
`sched-LB` (the subset's own lower bound), so it is internally consistent, but not
directly comparable to RoundRobin or ILP ratios which cover the full workload.
The `full-LB` column is shown for context.

**Symmetric ILP on 4×4×8 is tractable when warm-started.** ~32k binary variables
at `T = LB = 74`; CBC solves in ~6 minutes when `T_upper` is pinned to `LB` (so
no binary search above `LB` is needed). It is the binary-search starting at
`T_upper = 4·LB` that was intractable (~113k variables, infeasible to build in
PuLP within reasonable time). With OrbitGreedy as a `T_upper = LB` oracle, the
ILP is now an *independent verifier*, not a primary scheduler.

**OrbitGreedy assumes uniform-AllToAll workload symmetry.** OrbitGreedy and
PipelinedOrbit rely on `compute_orbits`, which assumes a translation-symmetric
workload. For skewed traffic, fall back to `ilp_optimal` (Schedule D).

**OrbitGreedy LB-match is mechanically proven per cell.** On every one of the
10 cells in the experiment matrix, `makespan = LB` follows from a König +
Smith's-deadline-feasibility argument (see
[orbit_greedy_optimality.md](orbit_greedy_optimality.md) §4.3) machine-verified
by `scripts/verify_smith_proof.py`. A closed-form proof for the entire
`{S, 2S}^n` family reduces to standard canonical-path enumeration; the work
is bookkeeping rather than research.

## Future Work

- Run symmetric ILP on 4×4×8 (or estimate solve time vs. LB horizon).
- `m` sweep (`msg_size ∈ {1, 4, 16}`): test whether ratio is invariant to message
  size as predicted by the bandwidth-only model.
- Improve RoundRobin by overlapping phases (pipelined injection) to close the 4–8×
  gap.
- Bisection-bandwidth lower bound: compute min-cut bound and report
  `max(LB_link, LB_bisection)`.
- DOR dim-order ablation (smallest-first vs. largest-first): effect on LB and makespan.

## How to Regenerate

```bash
bash eval/run_all.sh
cat results/$(date +%Y-%m-%d)/headlines.csv
```

See [evaluation.md](evaluation.md) for the full reproduction walkthrough.
