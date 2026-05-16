# Results: Closing the gap to LB=75

**Baseline (incoming):** `cpsat_literal` makespan **80** at `t_upper=80`,
30-min budget, no warm-start. Schedule:
`fixtures/schedule_8x4x4_loaded_cpsat_literal.json`.

**LB:** 75 (max physical-edge load).

## Probe 1: CP-SAT warm-started from makespan-80

Schedule: `t_upper ∈ {79, 78, 77, 76}`, 14400s (4h) budget per probe,
8 workers, warm-start from `fixtures/schedule_8x4x4_loaded_cpsat_literal.json`.

| t_upper | status | makespan | violations | runtime |
|---:|---|---:|---:|---:|
| (pending) | | | | |

## Probe 2: LNS with CP-SAT subsolver

Driver: `lns_cpsat_repair`, `n_iters=100`, `per_subproblem_budget_s=300`,
`destroy_size_frac=0.05`, strategies rotated time_window / random_subset /
makespan_flows.

| iter | strategy | destroy_size | new_makespan | accepted | runtime |
|---:|---|---:|---:|:---:|---:|
| (pending) | | | | | |

## Summary

(Filled in after probes complete.)
