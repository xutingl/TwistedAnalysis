# CNS Schedules

Schedules for the loaded (8, 4, 4) twisted-torus routing
(`fixtures/routing_table_8x4x4_twist.json`), renamed for the CNS pipeline
((8, 4, 4) and (4, 4, 8) refer to the same physical topology under
dimension-label permutation).

| CNS filename | Source fixture | Scheduler | Makespan | Physical-edge violations |
|---|---|---:|---:|---:|
| `schedule_spreadgreedyk2_4x4x8_twisted.json` | `schedule_8x4x4_loaded_spread_greedy_k2.json` | `spread_greedy(k=2)` — per-device DMA-cap variant of `literal_greedy` | 92 | 0 |
| `schedule_cpsatliteralwarm_4x4x8_twisted.json` | `schedule_8x4x4_loaded_cpsat_literal_warm.json` | `cpsat_literal` warm-started from makespan-80 (OR-Tools, t_upper=79, 4 h budget) | **78** | 0 |
| `schedule_cpsatliteral_4x4x8_twisted.json` | `schedule_8x4x4_loaded_cpsat_literal.json` | `cpsat_literal` (OR-Tools, t_upper=80, 30 min budget) | 80 | 0 |
| `schedule_orbitfull_4x4x8_twisted.json` | `schedule_8x4x4_loaded_orbit_greedy_full_lpt_tail_asc.json` | `orbit_greedy_full` | 85 | 0 |
| `schedule_literalgreedy_4x4x8_twisted.json` | `schedule_8x4x4_loaded_literal_greedy_lpt.json` | `literal_greedy` | 87 | 0 |
| `schedule_orbit_4x4x8_twisted.json` | `schedule_8x4x4_loaded_lpt_tail_asc.json` | original `orbit_greedy` (pre-fix) | 73 | **8160 — DO NOT BENCHMARK AS-IS** |

LB for this routing = 75 (max physical-edge load).

**Recommended for production measurement runs: `spreadgreedyk2` (and run a side-by-side TPU benchmark vs `cpsatliteralwarm`).** The makespan-78 `cpsatliteralwarm` schedule measured 132764 gbps on TPU v5e — essentially unchanged from `orbitfull` (132758 gbps) and ~1.3% below the P2P reference (134541 gbps), despite a simulator projection of +7.5%. The leading hypothesis is that per-device DMA-engine oversubscription dominates per-round wall-clock; `spread_greedy(k=2)` caps each device at 2 simultaneous outgoing AND incoming DMAs per round, trading higher simulator makespan (92 vs 78) for lower per-round contention. The other K values (`spread_greedy_k1` — P2P-style, makespan 145; `spread_greedy_k3` — makespan 88; `spread_greedy_k4` — makespan 86) are shipped in `fixtures/` for TPU-side comparison but not promoted to `cns_schedules/`. Provenance: `eval/explorations/2026-05-17-spread-scheduling/`.

The previously-recommended `cpsatliteralwarm` (makespan 78, projected +7.5% vs P2P; measured ~0%) is retained as the makespan-optimal baseline.

## ⚠ Note on `schedule_orbit_4x4x8_twisted.json`

This is the schedule produced by the pre-Task-10 `orbit_greedy` algorithm,
which keys its busy table on `(dim, dir)` classes. That keying conflates
twist-wrap edges with standard edges, so the resulting schedule double-books
physical edges 8160 times (verified by
`twisted_analysis.schedules.verify.verify_capacity`).

The schedule's apparent makespan of 73 (below LB = 75) is *not real*: it
relies on edge collisions that the hardware will serialize at runtime,
inflating actual wall-clock. If you benchmark this file as if it were a
valid schedule, the comparison against `orbitfull` / `literalgreedy` will be
unfair — `orbit` will look faster than it actually is.

Recommended uses:
- **Production / measurement runs:** use `orbitfull` (makespan 85) or
  `literalgreedy` (makespan 87) — both are capacity-feasible.
- **Negative-control / historical reference:** keep `orbit` *only* if you
  want to demonstrate the bug (e.g., show that the corrected schedule
  closes a measurable wall-clock gap on real silicon).

If CNS doesn't need the broken file, delete it.

## ILP-optimal schedule (CP-SAT got close; CBC still intractable)

An exact ILP schedule on the literal `N(N-1) = 16,256` flow set would close
the remaining gap to LB = 75. CBC (via the `ilp_literal` scheduler) cannot
do this: it failed to find any feasible incumbent within 80 minutes on the
root LP relaxation alone (1.37 M binary vars × ~50 k constraints, 500 MB
MPS). The codebase's symmetric ILP scheduler in
`twisted_analysis/schedules/lp_symmetric.py` is ~130× smaller and finishes
in ~6 min on the same topology — but it assumes routing translation-
equivariance under `(dim, dir)`, which this loaded routing violates (same
reason `orbit_greedy` failed). So neither CBC-based variant is available
for this routing at this size.

**However**, the `cpsat_literal` scheduler (Google OR-Tools CP-SAT, native
at-most-one + parallel search + 8 workers) can find feasible incumbents at
this scale. With a 30-min wall-clock budget per `t_upper` probe and binary
search over `t_upper ∈ {84, 83, 82, 81, 80, 78, 76}`, CP-SAT reached
makespan **80** (saved as `schedule_cpsatliteral_4x4x8_twisted.json`).
The remaining 80 → 75 gap is open: CP-SAT timed out at `t_upper ∈ {76, 78}`
without an incumbent — that's evidence the search is hard at those bounds,
not proof of infeasibility. A longer compute budget (4-8 h per probe) or a
warm-start from the current makespan-80 incumbent could plausibly close
more of the gap. See `eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/`
for the full search log.
