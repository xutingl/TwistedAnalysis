# Spread-scheduling: per-device DMA-cap variant of literal_greedy

## Problem

The 2026-05-16 exploration produced a makespan-78 schedule (warm-started
CP-SAT) on the loaded 8×4×4 routing — simulator-projected to ~144607 gbps
(+7.5% above P2P's measured 134541 gbps). When the corresponding Pallas
kernel was run on TPU v5e, the actual measured throughput was **132764
gbps** — essentially identical to the orbit_greedy-85 kernel's 132758 gbps,
and ~1.3% **below** the P2P reference. The +9% simulator gain translated
to ~0% wall-clock change.

The leading hypothesis: the simulator's "1 round = 1 unit of wall-clock"
model is wrong on TPU. Real wall-clock per round is dominated by per-device
DMA-engine setup, ICI link bandwidth, HBM bandwidth, and semaphore wait
latency — none of which the simulator models. A schedule that issues many
DMAs from the same device in the same round oversubscribes the DMA engine;
the apparent "shorter makespan" is offset by per-round serialization that
the simulator doesn't see.

## Goal

Implement and ship a scheduling algorithm — `spread_greedy(k)` — that
explicitly limits per-device outgoing AND incoming DMAs per round to a
tunable cap K. Generate schedules at K ∈ {1, 2, 3, 4} on the loaded 8×4×4
routing. Pick K=2 as the headline (lowest non-trivial spread → maximum
"P2P-like" load distribution while keeping makespan competitive for
on-TPU testing). Ship the headline schedule as a fixture + Pallas kernel.

## Approach (one probe)

1. [01_spread_sweep.py](01_spread_sweep.py) — Run `spread_greedy(k, order="lpt")`
   for K ∈ {1, 2, 3, 4} on the loaded 8×4×4 routing. For each K, compute:
   - Makespan
   - Physical-edge capacity violations (must be 0 for all)
   - Max outgoing DMAs per device per round (must be ≤ K)
   - Max incoming DMAs per device per round (must be ≤ K)
   - Average DMAs per device per round (a measure of pipeline density)
   - Number of rounds with at least one DMA (a measure of "spread")

   Save all four schedule JSONs and a `01_spread_sweep_results.json`
   comparison table.

## Headline choice

K=2 is the headline. Rationale: it is the smallest K > 1, so it preserves
the most of the per-device uniformity that makes P2P competitive on TPU,
while permitting two-way pipelining per device per round (vs P2P's strict
one-way). If TPU measurement on this kernel disagrees with the simulator's
makespan ranking — i.e. K=2 outperforms cpsat_literal_warm (makespan 78)
despite a higher makespan — that is direct evidence that per-round
wall-clock, not round count, is the binding constraint. If K=2 is also
beaten on TPU by K=1 or K=4, the other K values are already saved as
fixtures and can be tested without re-running the probe.

The four K values are all promoted to `fixtures/` so a TPU operator can
deploy any of them; only K=2 gets the cns_schedules entry, the recommended
`fixtures/cns_schedules/readme.md` row, and the pre-generated Pallas kernel.

## Compute budget

Minutes. Greedy is fast (each K runs in seconds on N=128). No long
background runs.

## Outcome

(Filled in after the probe completes.)
