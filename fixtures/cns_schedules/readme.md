# CNS Schedules

Schedules for the loaded (8, 4, 4) twisted-torus routing
(`fixtures/routing_table_8x4x4_twist.json`), renamed for the CNS pipeline
((8, 4, 4) and (4, 4, 8) refer to the same physical topology under
dimension-label permutation).

| CNS filename | Source fixture | Scheduler | Makespan | Physical-edge violations |
|---|---|---:|---:|---:|
| `schedule_orbit_4x4x8_twisted.json` | `schedule_8x4x4_loaded_lpt_tail_asc.json` | original `orbit_greedy` (pre-fix) | 73 | **8160 — DO NOT BENCHMARK AS-IS** |
| `schedule_orbitfull_4x4x8_twisted.json` | `schedule_8x4x4_loaded_orbit_greedy_full_lpt_tail_asc.json` | `orbit_greedy_full` | 85 | 0 |
| `schedule_literalgreedy_4x4x8_twisted.json` | `schedule_8x4x4_loaded_literal_greedy_lpt.json` | `literal_greedy` | 87 | 0 |

LB for this routing = 75 (max physical-edge load).

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

## ILP-optimal schedule (not provided)

An exact ILP schedule on the literal `N(N-1) = 16,256` flow set would close
the remaining gap to LB = 75, but is intractable at this size: CBC failed
to find any feasible incumbent within 80 minutes of wall-clock on the root
LP relaxation alone (1.37 M binary vars × ~50 k constraints, 500 MB MPS).
The codebase's symmetric ILP scheduler in
`twisted_analysis/schedules/lp_symmetric.py` is ~130× smaller and finishes
in ~6 min on the same topology — but it assumes routing translation-
equivariance under `(dim, dir)`, which this loaded routing violates (same
reason `orbit_greedy` failed). So neither ILP variant is available for
this routing at this size.
