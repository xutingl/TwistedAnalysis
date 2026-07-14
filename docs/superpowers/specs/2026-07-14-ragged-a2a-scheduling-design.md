# Ragged A2A Scheduling — Design

**Date:** 2026-07-14
**Status:** Approved (fluid + greedy scope; CP-SAT family deferred)

## Problem

Schedule a *ragged* AllToAll — per-pair demands of unequal size — on the
loaded 8×4×4 twisted-torus routing, minimizing makespan (primary) and
schedule-entry count (secondary).

- **Workload:** `fixtures/ragged_a2a_workload_node_128_min_32_max_1024_discrete.json`
  — flat list of `{src, dst, size}`. All 128×127 = 16,256 ordered pairs
  present, no self-pairs. Sizes are multiples of 32 in [32, 1024]
  (61% are 32; 303 pairs at 1024). Per-source totals range 7,360–18,688:
  genuinely imbalanced, translation symmetry is broken, so orbit-based
  schedulers do not apply — literal-flow algorithms only.
- **Routing:** `fixtures/routing_table_8x4x4_twist.json` (the physical
  4×4×8 v5e slice under the flatten convention). Paths are fixed per pair;
  max path length 6 hops.
- **New schedule capability:** entries may carry a `rate` field
  (float ≤ 1) giving the flow's share of every link on its path, and a
  `size` field so one flow can be split across multiple chunk entries.

### Key quantities (computed 2026-07-14)

| Quantity | Value |
|---|---:|
| Quantum (gcd of sizes) | 32 |
| Total demand | 1,583,712 |
| Max size-weighted directed-edge load (**LB**) | 12,608 = **394 quanta** |
| Uniform-workload sanity check on same routing | LB = 75 (matches prior work) |
| Max hops | 6 |
| Fluid-schedule makespan (see below) | 399 quanta = LB + max_hops − 1 |

## Why the fluid problem is closed-form

Allowing a continuous `rate` is exactly the LP relaxation of the old
one-flow-per-edge-per-step ILP — and with fixed paths the relaxation has a
closed-form optimum. Static rates `rate_f = d_f / LB` (where `d_f` is flow
f's demand) load every edge at exactly `load(e)/LB ≤ 1` and finish every
flow at exactly `LB`. No schedule can beat `LB` (the bottleneck edge must
carry `load(e)` data at capacity 1), so water-filling is makespan-optimal
*and* entry-count-optimal (one entry per flow). The hardness of the
uniform-case problem (CP-SAT stuck at 78 vs LB 75) *was* the integrality
gap; the rate field deletes it by fiat. Any reintroduced discreteness
(rate menus, per-device DMA caps, chunk budgets) brings NP-hardness back —
that is the deferred CP-SAT family's territory.

## Time model (pipelined-stream)

Generalizes the existing pipelined hop model (hop `i` of a flow with round
`r` fires at `t = r + i`) and is exactly backward compatible with it:

> A chunk `{round=r, rate=ρ, size=m·quantum, path}` occupies directed edge
> `i` of its path during `[r + i, r + i + m/ρ)` (times in quantum units),
> consuming `ρ` of that edge's unit capacity.
> Feasibility: Σρ ≤ 1 per (edge, time). Chunk finish time:
> `r + (L−1) + m/ρ` for an L-hop path.

- With `m = 1, ρ = 1` this reproduces today's semantics exactly, including
  the `finish = r + L` convention in `schedule_makespan`.
- The hop offset stays **+i quanta** (constant pipeline fill), *not*
  `i/ρ`: cut-through hardware forwards the head packet at wire latency
  regardless of bandwidth share; scaling fill by `1/ρ` would spuriously
  punish low-rate flows.

## Schedule format extension

`io/schedule.py` entries gain two optional fields:

- `rate`: float, `0 < rate ≤ 1`, default 1.0.
- `size`: positive int in **workload units** (bytes here), default 1.
  Duration in quanta is `(size/quantum)/rate`; the quantum comes from the
  workload, so interpreting a ragged schedule requires its workload file
  (legacy uniform schedules: size 1, quantum 1 — unchanged semantics).

A flow may appear as multiple chunk entries. Validation (when a workload
is supplied): per-pair chunk sizes sum exactly to the workload demand;
every workload pair covered; no pairs outside the workload. Existing
uniform schedules remain valid with no changes.

## Components

### 1. `io/workload.py` (new)

`load_workload(path) -> RaggedWorkload`. Validates: ints, `size > 0`,
`src ≠ dst`, no duplicate pairs.

### 2. `model/ragged.py` (new)

`RaggedWorkload` dataclass over flat IDs (unlike coordinate-based
`AllToAll`, since ragged workloads arrive as flat-ID JSON):
`demand: dict[(int, int), int]`, `quantum` (gcd of sizes, cached),
`link_load(table)` (size-weighted, same convention as
`AllToAll.link_load`), `lower_bound(table)` = max link load,
`bottleneck_edges(table)`.

### 3. `schedules/ragged_fluid.py` (new)

Closed-form water-filling: one entry per flow,
`{round: 0, rate: d_f/LB, size: d_f}` (rates as float64; `d_f/LB` computed
in consistent units). O(F·E). Makespan `LB + max_hops − 1` = 399 quanta;
serves as the LB certificate and degenerate-concurrency baseline.

### 4. `schedules/ragged_greedy.py` (new)

Integral (`rate = 1`) earliest-feasible greedy over flows in a
deterministic order; per-edge busy-quantum sets (all durations integral in
quanta, so set-based accounting suffices).

- **Orders:** `lpt` (default): sort by `(−size, −hops, src, dst)`;
  `spt`: `(size, hops, src, dst)`; `natural`: workload file order.
- **Non-preemptive** (`preemptive=False`, default): smallest `start` such
  that every edge `i` is free throughout `[start+i, start+i+d_f)`. One
  entry per flow.
- **Preemptive** (`preemptive=True`): assign the flow's `d_f` quanta to
  the earliest times `t` where all edges `i` are free at `t+i`; each
  maximal contiguous run of assigned quanta becomes one chunk entry.
  Lower makespan, more entries — the makespan-vs-descriptor tradeoff is
  the point; both variants report entry counts.

### 5. `schedules/verify.py` (extend, existing functions untouched)

- `verify_capacity_ragged(schedule, *, quantum=1)`: per-directed-edge
  sweepline over chunk occupancy intervals; violation when accumulated
  rate exceeds `1 + 1e-6`. Reduces to the existing set-semantics check on
  integral schedules; `quantum=1` with defaulted `rate`/`size` makes
  legacy uniform schedules verify unchanged.
- `schedule_makespan_ragged(schedule, *, quantum=1)`:
  `max_f (round + (L−1) + (size/quantum)/rate)`.
- `verify_workload_coverage(schedule, workload)`: chunk-sum /
  coverage / no-extra-pairs check described above.

### 6. Wiring

- `io/schedule.py`: validate optional `rate`/`size` fields; add
  `schedule_from_ragged_fluid` / `schedule_from_ragged_greedy` adapters
  and dispatch entries (both take a `workload` kwarg).
- `scripts/generate_ragged_schedule.py`: CLI
  `(workload, routing-table, algorithm, order, preemptive) → fixtures/schedule_<slice>_<router>_<algorithm>[_variant].json`.
- `eval/run_ragged_a2a.sh`: reproducible run of
  {fluid, greedy non-preemptive, greedy preemptive (lpt / spt / natural)}
  → table of {makespan (quanta), gap to LB=394, total entries, max
  entries/flow}, written under `results/<date>/`.

## Testing

1. Workload I/O round-trip + validation errors (duplicate pair, size ≤ 0).
2. `RaggedWorkload.lower_bound` = 12,608 on the fixture + loaded routing;
   uniform demand through the same code path reproduces LB = 75.
3. Fluid: verifier-clean; makespan = 399 quanta; one entry per flow;
   coverage check passes.
4. Greedy (both variants, three orders): verifier-clean; coverage check
   passes; non-preemptive has exactly one entry per flow; preemptive
   makespan ≤ non-preemptive makespan on the fixture.
5. Backward compat: existing uniform schedule fixture
   (`schedule_8x4x4_loaded_cpsat_literal_warm.json`) passes
   `verify_capacity_ragged` with defaults and gives the same makespan as
   `schedule_makespan`.
6. Time-model unit test: a single 2-hop chunk `(m=4, ρ=0.5)` occupies edge
   0 over [r, r+8) and edge 1 over [r+1, r+9); finish = r + 1 + 8.

## Out of scope (deferred)

- Discrete-rate CP-SAT (`rate ∈ {1, ½, ¼}`, interval vars, per-edge
  cumulative capacity, warm-started from greedy) — pursue if the greedy
  gap to 394 is large.
- Per-device concurrent-DMA caps (spread_greedy analog for ragged).
- Fluid-decomposition few-phase schedules ((1+ε)·LB with O(1/ε)
  entries/flow).
- Pallas kernel generation for ragged schedules (kernel currently assumes
  uniform message size; rate realization on hardware is implicit
  fair-sharing and needs its own design).
