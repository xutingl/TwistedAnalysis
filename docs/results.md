# Results

## Headline Numbers (First Eval Run, 2026-05-12)

Source: `results/2026-05-12/headlines.csv`

| Experiment | Topology | Schedule | Coverage | Schedule LB | Full-AllToAll LB | Makespan | Ratio |
|---|---|---|---:|---:|---:|---:|---:|
| `2x4_ilp` | 2×4 | ILP-optimal | 56/56 | 4 | 4 | **4** | **1.00** |
| `2x4_dim_phased` | 2×4 | DimPhased | 32/56 | 3 | 4 | 5 | 1.67 |
| `2x4_rr` | 2×4 | RoundRobin | 56/56 | 4 | 4 | 13 | 3.25 |
| `4x8_dim_phased` | 4×8 | DimPhased | 320/992 | 10 | 26 | 18 | 1.80 |
| `4x8_rr` | 4×8 | RoundRobin | 992/992 | 26 | 26 | 116 | 4.46 |
| `4x4x8_dim_phased` | 4×4×8 | DimPhased | 1664/16256 | 10 | 86 | 23 | 2.30 |
| `4x4x8_rr` | 4×4×8 | RoundRobin | 16256/16256 | 86 | 86 | 650 | 7.56 |

`Schedule LB` = max link load over the flows the schedule actually injects.
`Full-AllToAll LB` = max link load over all `N·(N-1)` flows (the project's headline LB).
`Ratio = makespan / Schedule LB`. For full-coverage schedules (ILP, RR), the two LBs are equal. Reproduced with `bash eval/run_all.sh`.

## Key Findings

**1. ILP achieves the lower bound on 2×4.**
For the 2×4 topology under dimension-order routing with `m=1`, the ILP produces
`makespan = 4 = LB`. The routing table has no intrinsic gap; the bandwidth lower
bound is achievable, so `gap_routing = M_opt / LB = 1.00`. There is no inherent
loss from the twisted-torus DOR on this topology.

**2. Round-robin is far from optimal; the gap grows with topology size.**

| Topology | RR ratio |
|---|---|
| 2×4 | 3.25× |
| 4×8 | 4.46× |
| 4×4×8 | 7.56× |

The Latin-square round-robin schedule accumulates phase-makespan overhead across
`N-1` back-to-back phases. Since the ILP achieves `LB` on 2×4, the round-robin gap
is a **scheduling** inefficiency — specifically, phase-boundary idle time — not a
routing problem.

**3. DimPhased is not a full AllToAll — interpret with care.**
DimPhased covers only `(src, dst)` pairs that differ in exactly one dimension
(57% of flows on 2×4, 32% on 4×8, 10% on 4×4×8). The reported `Ratio` is computed
against DimPhased's own subset LB so it is honest (≥ 1.0) and comparable to
RoundRobin on the same denominator-shape. The `Full-AllToAll LB` column is shown
for context: DimPhased's makespan against the full-workload LB *would* look
artificially small, since the denominator's workload is much larger than what
DimPhased actually runs. Bottom line: DimPhased's `Ratio` quantifies its
inefficiency *for the subset it covers*; it is not directly comparable to
RoundRobin's `Ratio` (different workloads).

## Caveats

**LP relaxation is weak.** The LP relaxation (`lp_relax_lower_bound`) minimizes the
weighted-average completion time (sum of `(t+1) * x[u, last, t]`), not the
makespan. Its value can be below `LB`. It is not a tight makespan bound. See
[lp_formulation.md](lp_formulation.md) for the precise statement.

**ILP not run on 4×8 or 4×4×8.** Whether those topologies' lower bounds are
achievable is still open. The variable count for 4×8 is ~90,000 and for 4×4×8 is
~27 million; both require symmetry reduction or LP relaxation only. Future work:
run ILP on 4×8 with orbit-based symmetry breaking.

## Future Work

- Run ILP on 4×8 to determine `gap_routing` (whether `LB` is achievable).
- DOR dim-order ablation (smallest-first vs. largest-first): effect on `LB` and
  makespan.
- `m` sweep (`msg_size ∈ {1, 4, 16}`): test whether the ratio is invariant to
  message size as predicted by the bandwidth-only model.
- Improve RoundRobin by reordering phases by twist-induced shape, or by overlapping
  phases (pipelined injection).
- Bisection-bandwidth lower bound: compute min-cut bound and report
  `max(LB_link, LB_bisection)`.

## How to Regenerate

```bash
bash eval/run_all.sh
cat results/$(date +%Y-%m-%d)/headlines.csv
```

See [evaluation.md](evaluation.md) for the full reproduction walkthrough.
