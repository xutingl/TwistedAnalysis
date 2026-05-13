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
| 4×8      | 26     | 21     | 19%       |
| 4×4×8    | 86     | 74     | 14%       |

## Schedule Comparison (with ILP Routing)

| name                   | router | sched                 | sched-LB | full-LB | flows         | makespan | ratio |
|------------------------|--------|-----------------------|----------|---------|---------------|----------|-------|
| 2x4_dim_phased         | ilp    | dim_phased            | 3        | 3       | 32/56         | 3        | 1.00  |
| 2x4_ilp                | ilp    | ilp_optimal           | 3        | 3       | 56/56         | 3        | 1.00  |
| 2x4_ilp_symmetric      | ilp    | ilp_optimal_symmetric | 3        | 3       | 56/56         | 3        | 1.00  |
| 2x4_rr                 | ilp    | round_robin           | 3        | 3       | 56/56         | 13       | 4.33  |
| 2x4_xla                | ilp    | xla                   | 3        | 3       | 56/56         | 13       | 4.33  |
| 4x8_dim_phased         | ilp    | dim_phased            | 8        | 21      | 320/992       | 13       | 1.62  |
| 4x8_ilp_symmetric      | ilp    | ilp_optimal_symmetric | 21       | 21      | 992/992       | 21       | 1.00  |
| 4x8_rr                 | ilp    | round_robin           | 21       | 21      | 992/992       | 122      | 5.81  |
| 4x8_xla                | ilp    | xla                   | 21       | 21      | 992/992       | 122      | 5.81  |
| 4x4x8_dim_phased       | ilp    | dim_phased            | 8        | 74      | 1664/16256    | 18       | 2.25  |
| 4x4x8_rr               | ilp    | round_robin           | 74       | 74      | 16256/16256   | 637      | 8.61  |
| 4x4x8_xla              | ilp    | xla                   | 74       | 74      | 16256/16256   | 637      | 8.61  |

## DOR Routing Reference

| name                   | router | sched                 | sched-LB | full-LB | flows         | makespan | ratio |
|------------------------|--------|-----------------------|----------|---------|---------------|----------|-------|
| 2x4_dim_phased_dor     | dor    | dim_phased            | 3        | 4       | 32/56         | 5        | 1.67  |
| 2x4_ilp_dor            | dor    | ilp_optimal           | 4        | 4       | 56/56         | 4        | 1.00  |
| 2x4_rr_dor             | dor    | round_robin           | 4        | 4       | 56/56         | 13       | 3.25  |
| 2x4_xla_dor            | dor    | xla                   | 4        | 4       | 56/56         | 13       | 3.25  |
| 4x8_dim_phased_dor     | dor    | dim_phased            | 10       | 26      | 320/992       | 18       | 1.80  |
| 4x8_rr_dor             | dor    | round_robin           | 26       | 26      | 992/992       | 116      | 4.46  |
| 4x8_xla_dor            | dor    | xla                   | 26       | 26      | 992/992       | 116      | 4.46  |
| 4x4x8_dim_phased_dor   | dor    | dim_phased            | 10       | 86      | 1664/16256    | 23       | 2.30  |
| 4x4x8_rr_dor           | dor    | round_robin           | 86       | 86      | 16256/16256   | 650      | 7.56  |
| 4x4x8_xla_dor          | dor    | xla                   | 86       | 86      | 16256/16256   | 650      | 7.56  |

## Key Findings

**1. ILP routing reduces LB by 14–25% across topologies.**
Distributing traffic across all minimal paths (ILPRouter) lowers the max-link load
versus DOR's lexicographic tie-breaking:

| Topology | DOR LB | ILP LB | Reduction |
|----------|--------|--------|-----------|
| 2×4      | 4      | 3      | 25%       |
| 4×8      | 26     | 21     | 19%       |
| 4×4×8    | 86     | 74     | 14%       |

**2. Symmetric ILP scheduling on 4×8 achieves LB exactly.**
With ILP routing (LB = 21), the symmetric ILP scheduler (factor-32 variable
reduction; ~14 s solve) produces `makespan = 21 = LB`, ratio = 1.00. This is the
first zero-gap result on 4×8.

**3. The previous 4×8 gap was a DOR artifact, not an intrinsic topology limit.**
Under DOR routing (LB = 26), the full ILP gives `makespan = 4 = LB` only on 2×4.
On 4×8, DOR's poor tie-breaking raises LB by 24% relative to ILP routing, making
the lower bound harder to achieve. Switching to ILP routing eliminates the gap.

**4. Round-robin is still far from optimal across all topologies.**

| Topology | Router | RR makespan | LB  | Ratio |
|----------|--------|-------------|-----|-------|
| 2×4      | ilp    | 13          | 3   | 4.33  |
| 4×8      | ilp    | 122         | 21  | 5.81  |
| 4×4×8    | ilp    | 637         | 74  | 8.61  |

The Latin-square round-robin schedule accumulates phase-makespan overhead across
`N-1` back-to-back phases. The gap is a **scheduling** inefficiency (phase-boundary
idle time), not a routing problem.

**5. DimPhased ratios improve with ILP routing.**
ILP routing lowers both the schedule LB and the realized makespan for DimPhased:

| Topology | DOR ratio | ILP ratio |
|----------|-----------|-----------|
| 2×4      | 1.67      | 1.00      |
| 4×8      | 1.80      | 1.62      |
| 4×4×8    | 2.30      | 2.25      |

DimPhased on 2×4 is now optimal (ratio = 1.00) under ILP routing. Recall that
DimPhased is still partial-coverage (see Caveats).

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
| 4×8 (ilp) | 122 | 122 |
| 4×4×8 (ilp) | 637 | 637 |
| 2×4 (dor) | 13 | 13 |
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

**Symmetric ILP not attempted on 4×4×8.** The 4×4×8 symmetric ILP would have
~47,000 binary variables at T = LB; tractability is unknown. LP relaxation only
for this topology.

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
