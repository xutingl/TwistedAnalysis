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
| 1 | 145 | 0 | 1 / 1 | 1.000 | 145 | ~1.0 s |
| 2 | 92  | 0 | 2 / 2 | 1.614 | 91  | ~0.5 s |
| 3 | 88  | 0 | 3 / 3 | 1.703 | 87  | ~0.5 s |
| 4 | 86  | 0 | 4 / 4 | 1.725 | 85  | ~0.5 s |

(Exact runtime values are in `01_spread_sweep_results.json`.)

## Summary

| K | makespan | vs cpsat_warm (78) | shipped as |
|---:|---:|:---:|---|
| 1 | 145 | +67 (much higher; pure P2P-style) | fixture only |
| 2 |  92 | +14 (headline)                    | fixture + cns + Pallas kernel (regular + inline) |
| 3 |  88 | +10                               | fixture only |
| 4 |  86 |  +8 (close to literal_greedy's 87) | fixture only |

**Key takeaway**: K=2 makespan is 92, an 18 % increase over `cpsat_literal_warm`'s
makespan 78. If TPU measurement on the K=2 kernel comes in at or above the
cpsat_warm kernel's 132764 gbps, the DMA-oversubscription hypothesis is
supported and `spread_greedy` becomes the production-recommended scheduler.
If K=2 measures notably below 132764 gbps, the hypothesis is rejected for this
routing — the makespan-78 schedule is genuinely near-optimal at the simulator
level AND the gap to P2P is due to something other than DMA cap (HBM, ICI
per-link bandwidth, or per-DMA setup).

Note that K=1 (makespan 145, exceeds N-1=127) shows the loaded routing's
edge-conflict constraints push beyond the trivial device LB even when each
device emits only one DMA per round — the routing itself is not single-DMA-
balanced.
