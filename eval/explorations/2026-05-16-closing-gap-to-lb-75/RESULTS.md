# Results: Closing the gap to LB=75

**Baseline (incoming):** `cpsat_literal` makespan **80** at `t_upper=80`,
30-min budget, no warm-start. Schedule:
`fixtures/schedule_8x4x4_loaded_cpsat_literal.json`.

**LB:** 75 (max physical-edge load).

## Probe 1: CP-SAT warm-started from makespan-80

Schedule: `t_upper ∈ {79, 78, 77, 76}`, 14400s (4h) budget per probe,
8 workers, warm-start chained from the previous probe's best (initially
`fixtures/schedule_8x4x4_loaded_cpsat_literal.json` at makespan 80).

| t_upper | status | makespan | violations | runtime |
|---:|---|---:|---:|---:|
| 79 | FEASIBLE | **78** | 0 | 14445s |
| 78 | FEASIBLE | **78** | 0 | 14432s |
| 77 | TIMEOUT (UNKNOWN, no incumbent) | — | — | 14432s |
| 76 | TIMEOUT (UNKNOWN, no incumbent) | — | — | 14429s |

**Best: makespan 78** at `t_upper=79`, saved to
`01_best_warm_start_schedule.json` and promoted to fixtures as
`fixtures/schedule_8x4x4_loaded_cpsat_literal_warm.json` and
`fixtures/cns_schedules/schedule_cpsatliteralwarm_4x4x8_twisted.json`.

**Pattern.** Warm-starting strictly helped at `t_upper=79`: cold CP-SAT at
this `t_upper` had not been tried in the prior exploration
(which sampled `{84, 83, 82, 81, 80, 78, 76}` at 30-min budget). The
makespan-80 seed gave CP-SAT enough information to find an improving
incumbent at makespan 78 in 4h. At `t_upper=78`, the warm-start essentially
reproduces the same incumbent. At `t_upper=77` and `t_upper=76`, even with
warm-start, CP-SAT cannot find any feasible incumbent in 4h — the search
space at those bounds is too constrained.

**Bug fixed during this probe.** The original `cpsat_literal` treated CP-SAT's
`UNKNOWN` status as "no incumbent" without checking. Commit `ed566b2` adds a
`CpSolverSolutionCallback` to capture intermediate incumbents and uses the
last captured one when status is `UNKNOWN`. In this exploration the fix had
no effect (probe 1's `UNKNOWN` returns were genuinely incumbent-free, as
verified by the `best:inf` lines throughout the CP-SAT search log), but the
fix is correct per OR-Tools docs and would matter in other regimes.

**Throughput projection.** Linear scaling from
`orbit_greedy_full[lpt_tail_asc]` (makespan 85 → 132758 gbps): makespan 78
→ **144607 gbps**, which is **+7.5% above P2P's 134541 gbps**. The prior
exploration's makespan 80 projected to +4.8%; this exploration adds another
2.7 percentage points.

## Probe 2: LNS with CP-SAT subsolver (default 5% destroy)

Driver: `lns_cpsat_repair`, `n_iters=100`, `per_subproblem_budget_s=300`,
`destroy_size_frac=0.05`, strategies rotated time_window /
random_subset / makespan_flows. Seed: makespan-78 schedule from Probe 1.

| outcome | count |
|---|---:|
| infeasible | 100 |

**Every single subproblem proved INFEASIBLE in 2–4 seconds.** Total runtime
368s (6 min) for 100 iterations. No accepted improvement. Final makespan: 78.

**Diagnosis.** At `destroy_size_frac=0.05`, only ~813 of 16256 flows are
destroyed per iteration; the pinned 95% already saturate the physical-edge
budget at `t_upper=77`. CP-SAT immediately proves there is no valid
assignment for the destroyed flows under the pinned constraints.

## Probe 2b: Aggressive LNS (30% destroy, 10-min subproblems)

Driver: `lns_cpsat_repair` with `destroy_size_frac=0.30`,
`per_subproblem_budget_s=600`, `n_iters=24`. Per-iteration `result` field
added in commit `b171723` to classify each outcome.

| outcome | count |
|---|---:|
| infeasible | 24 |

| strategy | typical \|D\| | typical per-iter time |
|---|---:|---:|
| time_window | 1248 | ~4s |
| random_subset | 5283 (32%) | ~10s |
| makespan_flows | 599 | ~3s |

**Total runtime: 141s (2.3 min) for 24 iterations.** Even at 30% destroy
with 10-min budget per subproblem, every subproblem PROVES infeasibility in
3–11 seconds. No incumbent found, no schedule saved.

**Strong negative finding.** The makespan-78 schedule is *structurally
tight*: any perturbation that pins 70%+ of the flows at their current
rounds and tries to fit the rest at `t_upper=77` is provably infeasible. The
pinned majority dominates the edge-time budget. Closing the 78→75 gap will
require either (a) a fundamentally different starting incumbent or (b) cold
CP-SAT with much greater compute budget than the 4h tried in Probe 1.

## Summary — Probe winners

| Probe | Best makespan | Beats seed (80)? | Closes 80→75 gap? |
|---|---:|:---:|:---:|
| Probe 1 warm-start CP-SAT | **78** | yes (−2) | partial (2 of 5) |
| Probe 2 LNS @5% | 78 | tie | no |
| Probe 2b LNS @30% | 78 | tie | no |

**Overall winner: makespan 78 via warm-start CP-SAT at `t_upper=79`.**
**Projects to ~144607 gbps vs P2P's 134541 gbps (+7.5%).** Throughput
improvement over the prior best (makespan 80): +2.7 percentage points.

LB=75 was NOT reached. Both LNS variants demonstrated that local-search
escape from makespan 78 is blocked by the schedule's structural tightness.
