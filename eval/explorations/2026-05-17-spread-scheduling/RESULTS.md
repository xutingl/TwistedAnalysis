# Results: Spread-scheduling K-sweep on loaded 8×4×4

**Baseline (incoming):**
- `cpsat_literal` warm-started, makespan 78 (the production fixture at
  `fixtures/schedule_8x4x4_loaded_cpsat_literal_warm.json`).
  Measured on TPU v5e: 132764 gbps. Simulator-projected: ~144607 gbps.
  Simulator-to-reality gap: ~9% over-estimate.

**LB:** 75 (max physical-edge load).

## K-sweep (probe 1)

`spread_greedy(k, order="lpt")` on `fixtures/routing_table_8x4x4_twist.json`.

| K | makespan | viol | max DMAs/device-round | avg DMAs/device-round | n_rounds_with_dma | runtime |
|---:|---:|---:|:---:|---:|---:|---:|
| (pending) | | | | | | |

## Summary

(Filled in after probe runs.)
