# CNS Schedules

Schedules for the loaded (8, 4, 4) twisted-torus routing
(`fixtures/routing_table_8x4x4_twist.json`), renamed for the CNS pipeline
((8, 4, 4) and (4, 4, 8) refer to the same physical topology under
dimension-label permutation).

| CNS filename | Source fixture | Scheduler | Makespan | Physical-edge violations |
|---|---|---:|---:|---:|
| `schedule_spreadgreedyk1_4x4x8_twisted.json` | `schedule_8x4x4_loaded_spread_greedy_k1.json` | `spread_greedy(k=1)` — P2P-style: 1 DMA per device per round, LB-aware ordering | 145 | 0 |
| `schedule_spreadgreedyk2_4x4x8_twisted.json` | `schedule_8x4x4_loaded_spread_greedy_k2.json` | `spread_greedy(k=2)` — per-device DMA-cap variant of `literal_greedy` | 92 | 0 |
| `schedule_cpsatliteralwarm_4x4x8_twisted.json` | `schedule_8x4x4_loaded_cpsat_literal_warm.json` | `cpsat_literal` warm-started from makespan-80 (OR-Tools, t_upper=79, 4 h budget) | **78** | 0 |
| `schedule_cpsatliteral_4x4x8_twisted.json` | `schedule_8x4x4_loaded_cpsat_literal.json` | `cpsat_literal` (OR-Tools, t_upper=80, 30 min budget) | 80 | 0 |
| `schedule_orbitfull_4x4x8_twisted.json` | `schedule_8x4x4_loaded_orbit_greedy_full_lpt_tail_asc.json` | `orbit_greedy_full` | 85 | 0 |
| `schedule_literalgreedy_4x4x8_twisted.json` | `schedule_8x4x4_loaded_literal_greedy_lpt.json` | `literal_greedy` | 87 | 0 |
| `schedule_orbit_4x4x8_twisted.json` | `schedule_8x4x4_loaded_lpt_tail_asc.json` | original `orbit_greedy` (pre-fix) | 73 | **8160 — DO NOT BENCHMARK AS-IS** |

LB for this routing = 75 (max physical-edge load).

**Recommended for production measurement runs: side-by-side TPU benchmark of `spreadgreedyk1`, `spreadgreedyk2`, and `cpsatliteralwarm` against the reference P2P kernel.** The makespan-78 `cpsatliteralwarm` schedule measured 132764 gbps on TPU v5e — essentially unchanged from `orbitfull` (132758 gbps) and ~1.3% below the P2P reference (134541 gbps), despite a simulator projection of +7.5%. The leading hypothesis is that per-device DMA-engine oversubscription dominates per-round wall-clock; `spread_greedy(k=K)` caps each device at K simultaneous outgoing AND incoming DMAs per round, trading higher simulator makespan for lower per-round contention. Two K values are promoted: `spreadgreedyk1` (makespan 145; same per-round structure as reference P2P but with LB-aware destination ordering instead of pure rotation) and `spreadgreedyk2` (makespan 92; smallest 2-way pipelining). The other K values (`spread_greedy_k3` — makespan 88; `spread_greedy_k4` — makespan 86) are shipped in `fixtures/` for follow-up comparison but not promoted to `cns_schedules/`. Provenance: `eval/explorations/2026-05-17-spread-scheduling/`.

The previously-recommended `cpsatliteralwarm` (makespan 78, projected +7.5% vs P2P; measured ~0%) is retained as the makespan-optimal baseline.

## Additional twisted-torus cells — `orbit_greedy_full` (2026-06-23)

`orbit_greedy_full` (order `lpt_tail_asc`) schedules for three further loaded
twisted-torus routings supplied as `fixtures/routcache_torus_<coords>_twisted.json`.
The flatten `slice` is the torus coords with largest dim first (verified by
single-hop topology consistency, not assumed). All are capacity-feasible (0
physical-edge violations) and cover the full `N*(N-1)` AllToAll flow set.
Regenerate via `scripts/generate_routcache_orbitfull_schedules.py`.

| CNS filename | Source routcache | slice | N | Makespan | LB | Physical-edge violations |
|---|---|---|---:|---:|---:|---:|
| `schedule_orbitfull_4x8_twisted.json` | `routcache_torus_4x8_twisted.json` | (8, 4) | 32 | 23 | 21 | 0 |
| `schedule_orbitfull_8x16_twisted.json` | `routcache_torus_8x16_twisted.json` | (16, 8) | 128 | 188 | 170 | 0 |
| `schedule_orbitfull_4x8x8_twisted.json` | `routcache_torus_4x8x8_twisted.json` | (8, 8, 4) | 256 | 223 | 184 | 0 |

Source fixtures (same content as the CNS copies):
`schedule_<slice>_loaded_orbit_greedy_full_lpt_tail_asc.json` for slices
`8x4`, `16x8`, `8x8x4` respectively.

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

## Ragged A2A schedules (2026-07-14)

Schedules for the **ragged** (per-pair-sized) AllToAll workload
`fixtures/ragged_a2a_workload_node_128_min_32_max_1024_discrete.json` on the
same loaded (8, 4, 4) twisted-torus routing. Sizes are multiples of 32 in
[32, 1024]; quantum = 32; **LB = 12,608 bytes = 394 quanta** (max
size-weighted physical-edge load). All entries are verified 0-violation
(rate-capacity sweepline) and coverage-exact (per-pair chunk sizes sum to
the workload demand). Regenerate via `eval/run_ragged_a2a.sh`.

**Format extension:** entries carry two additional fields beyond
`{round, src, dst, path}`:

- `rate` (float in (0, 1]) — the flow's share of every link on its path.
- `size` (int, bytes) — the bytes moved by this entry; a flow may be split
  across multiple chunk entries (its sizes sum to the workload demand, so
  the demand matrix is recoverable from the schedule alone).

Time model (pipelined-stream): an entry occupies path edge `i` during
`[round + i, round + i + (size/32)/rate)` in quantum units; per-edge
capacity is `sum(rate) <= 1` at all times. Greedy schedules use `rate = 1.0`
throughout; only `raggedfluid` uses fractional rates (as small as 32/12608
≈ 0.0025 — check DMA-sharing granularity before benchmarking it as-is).

| CNS filename | Source fixture | Scheduler | Makespan (quanta) | Gap vs LB=394 | Entries | Max chunks/flow | Violations |
|---|---|---|---:|---:|---:|---:|---:|
| `schedule_raggedgreedylptpre_4x4x8_twisted.json` | `schedule_8x4x4_loaded_ragged_greedy_lpt_pre.json` | `ragged_greedy` lpt, preemptive | **394** | **0.00%** | 19,959 | 6 | 0 |
| `schedule_raggedfluid_4x4x8_twisted.json` | `schedule_8x4x4_loaded_ragged_fluid.json` | `ragged_fluid` (closed-form water-filling) | 399 | +1.27% | 16,256 | 1 | 0 |
| `schedule_raggedgreedylpt_4x4x8_twisted.json` | `schedule_8x4x4_loaded_ragged_greedy_lpt.json` | `ragged_greedy` lpt, non-preemptive | 410 | +4.06% | 16,256 | 1 | 0 |
| `schedule_raggedgreedynatural_4x4x8_twisted.json` | `schedule_8x4x4_loaded_ragged_greedy_natural.json` | `ragged_greedy` natural, non-preemptive | 540 | +37.06% | 16,256 | 1 | 0 |
| `schedule_raggedgreedyspt_4x4x8_twisted.json` | `schedule_8x4x4_loaded_ragged_greedy_spt.json` | `ragged_greedy` spt, non-preemptive | 588 | +49.24% | 16,256 | 1 | 0 |

**Recommended for production measurement runs: `raggedgreedylptpre`** — it
achieves the lower bound exactly (394 quanta, a certified-optimal integral
schedule; every entry at `rate = 1.0`, so no fractional-rate hardware
assumptions), at the cost of ~23% more DMA descriptors than one-entry-per-
flow schedules (19,959 entries, at most 6 chunks per flow).
`raggedgreedylpt` (410, one entry per flow, rate 1.0) is the
minimal-descriptor fallback if per-DMA setup cost dominates at this entry
count. `raggedfluid` is the theoretical baseline: makespan-optimal in the
continuous-rate model but with 127 concurrent DMAs per device at tiny
fractional rates — benchmark it only if the DMA engine's bandwidth-sharing
behavior is what you want to measure.
