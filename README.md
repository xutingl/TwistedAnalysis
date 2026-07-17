# TwistedAnalysis

Quantifies the AllToAll performance gap on twisted-torus topologies under
multiple routing strategies (DOR, ILPRouter, and externally-supplied "loaded"
routings such as TPU OCS routes) and multiple scheduling algorithms.

## What

For twisted-torus topologies in the `{S, 2S}` shape family (2×4, 2×2×4, 2×4×4,
4×8, 4×4×8 — i.e. 2D and 3D shapes with one or more twisted dimensions), we
compute and compare:

### Routing strategies

The routing chooses, for every `(src, dst)` pair, a sequence of physical-edge
hops through the torus. Different routers produce different bandwidth lower
bounds `LB` (max physical-edge load) and different *path-symmetry* properties
that downstream schedulers depend on:

| Router | LB on 4×4×8 | Translation-equivariant under `(dim, dir)`? |
|---|---:|---|
| **DOR** (dimension-order) | 86 | Yes, by construction |
| **ILPRouter** (load-balanced minimal, LP-based) — default | 74 | On small cells only; *fails* on (2,4,4), (4,8), (4,4,8) |
| **Loaded** (e.g. `fixtures/routing/routing_table_8x4x4_twist.json`) | 75 | Fails (uses 10 edge-orbit classes incl. twist-wraps + escape-VC routing) |

ILP routing reduces LB by 14–31% vs. DOR, but its translation-equivariance
properties matter to the scheduler. Translation-equivariance means
`path(σ·u, σ·v) = σ · path(u, v)` — i.e., per-source paths are translates of
the canonical path. When this holds, orbit-class capacity equals physical-edge
capacity. When it fails, the two diverge and any orbit-based scheduler is
working with a weaker (smaller) feasible set than it claims.

### Scheduling algorithms

| Scheduler | What it optimizes | Optimality |
|---|---|---|
| `orbit_greedy` (default) | Orbit-greedy with full-physical-edge accounting (delegates to `orbit_greedy_full` since 2026-05-15) | LB-tight on (2,4) and (2,2,4) ILP; +1 over LB on (2,4,4) and (4,8) ILP (confirmed via literal ILP); +9 over LB on the loaded 8×4×4 routing with `lpt_tail_asc` (but only +9 with `lpt`) |
| `orbit_greedy_full` | Same algorithm; explicit name | Same as above. On loaded 8×4×4, `order='lpt'` gives makespan 84 vs `lpt_tail_asc`'s 85 — recommended default on non-translation-equivariant routings |
| `literal_greedy` | LMR-style per-flow earliest-feasible | Bounded by `O(c + d)` per LMR; +5–14% over `orbit_greedy_full` empirically |
| `spread_greedy` | `literal_greedy` plus per-device-per-round DMA cap K | At K=1, each device sends/receives 1 DMA per round (makespan ≥ N-1). At K=∞, identical to `literal_greedy`. Intermediate K trades simulator makespan for per-device uniformity. Motivated by the 2026-05-17 finding that the makespan-78 `cpsat_literal_warm` kernel measured 132764 gbps on TPU v5e — within noise of `orbit_greedy`'s 132758 and below P2P's 134541, despite +9% simulator projection. K=2 on loaded 8×4×4 gives makespan 92. See [2026-05-17 exploration](eval/explorations/2026-05-17-spread-scheduling/) |
| `orbit_pack` | **Step-model** (barrier-delimited) scheduler: FFD-packs whole orbits into ≤ K-orbit steps under a whole-path edge-load cap C. Targets `--per-step-barrier` (TPU v4 / pfc) execution, where the barrier serializes steps — so staggered cross-round capacity is irrelevant and within-step whole-path congestion (which `verify_capacity` never checks) is what binds. Orbits are permutations, so every device sends and receives exactly K DMAs per step | On loaded 8×4×4 at C=3 (= P2P rotation's own worst per-round whole-path load): K=2 → **64 steps**, K=3 → **43**, K=6 → **27** — vs `orbit_greedy_full`'s 80 barrier steps and P2P's 127. Staggered-infeasible **by design**: verify with `verify_capacity_step`, generate kernels with `--capacity-model step --step-edge-cap C`. Motivated by the 2026-07 TPU v4 finding that balanced-step schedules (orbit_greedy_full, 80 steps) beat P2P while the device-jagged `cpsat_literal_warm` (Σ_t max-per-device sends = 268 vs 127; incast up to 8) loses to it |
| `cpsat_literal` | CP-SAT (OR-Tools) on literal flow set; supports `warm_start_schedule` and `fixed_assignments` kwargs | Stronger than CBC-based `ilp_literal` on large cells. Cold @30 min/probe found **makespan 80** on loaded 8×4×4 (vs orbit_greedy's 84-85). Warm-started @4 h/probe found **makespan 78** at `t_upper=79` — see [2026-05-16 exploration](eval/explorations/2026-05-16-closing-gap-to-lb-75/). **2026-07 TPU v4 caveat:** underperforms the P2P baseline in wall-clock (device-jagged rounds; asymmetric, so it cannot use `--per-step-barrier`) — see `orbit_pack` |
| `lp_rounding` | LP relaxation of `ilp_literal` + Raghavan-Thompson randomized rounding | Polynomial-time but CBC's LP solve is intractable at N=128 (>6.5h, no completion). Could be revisited with HiGHS/Gurobi |
| `local_search` | Hill-climbing shift-earlier moves on a seed schedule | Refinement only; zero improvement on all tested seeds. Use as a post-pass to certify local tightness, not as primary search |
| `lns_cpsat` | Large-Neighborhood Search: destroy a subset of a seed schedule, re-solve with CP-SAT, accept strictly-better incumbents. Three destroy strategies (`time_window`, `random_subset`, `makespan_flows`) | Requires `seed_schedule` kwarg. On loaded 8×4×4 with the makespan-78 seed at `destroy_size_frac ∈ {0.05, 0.30}`, every subproblem proved INFEASIBLE in 3–11s — the schedule is structurally tight. Useful when greedy/single-shot CP-SAT lands at a local optimum that's NOT structurally tight |
| `ilp_literal` | Exact ILP on the literal `N(N-1)` flow set (CBC) | Provably optimal under physical-edge capacity. Tractable up to N=32 (~minutes); intractable at N=128 (CBC hangs on 1.37M binary vars) |
| `pipelined_orbit` | Orbit greedy with `t_{i+1} = t_i + 1` constraint | Diagnostic only; not optimal in general |
| `round_robin` / `xla` | Latin-square rotation (baseline) | 4–9× over LB on full AllToAll |
| `ragged_fluid` | Ragged workloads: closed-form water-filling (`rate = size/LB`), one entry per flow | Provably makespan- and entry-count-optimal in the continuous-rate (fluid) model; on the 128-node ragged fixture: LB = 394 quanta, makespan 399 (LB + pipeline fill) |
| `ragged_greedy` | Ragged workloads: integral (`rate = 1`) earliest-feasible greedy; non-preemptive (1 entry/flow) or preemptive (chunked) variants; orders `lpt`/`spt`/`natural` | No LB guarantee; on the 128-node ragged fixture (LB = 394, `eval/run_ragged_a2a.sh`): non-preemptive `lpt` 410 (+4.06%), `natural` 540 (+37.06%), `spt` 588 (+49.24%); preemptive `lpt` reaches the LB exactly (394, 0.00% gap) at the cost of chunking (19959 entries, up to 6 chunks/flow, vs 16256 entries / 1 chunk non-preemptive) |

A gap of 1 means the routing+schedule saturate every bottleneck link; >1
quantifies inefficiency. **Performance varies materially by routing**: see
`docs/results.md` for the matrix.

For empirical searches over scheduling algorithms specifically targeting
the loaded 8×4×4 routing (which the deployed Pallas kernel actually uses):

- [eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/](eval/explorations/2026-05-15-beating-p2p-loaded-8x4x4/) — cold CP-SAT @30 min/probe reaches makespan 80, projected +4.8% above P2P.
- [eval/explorations/2026-05-16-closing-gap-to-lb-75/](eval/explorations/2026-05-16-closing-gap-to-lb-75/) — warm-started CP-SAT @4 h/probe reaches **makespan 78** at `t_upper=79`, projected **+7.5% above P2P**. LNS at 5–30% destroy cannot escape the makespan-78 local optimum (every subproblem provably INFEASIBLE in seconds). LB=75 remains open.
- [eval/explorations/2026-05-17-spread-scheduling/](eval/explorations/2026-05-17-spread-scheduling/) — `spread_greedy(k)` shipped to test the hypothesis that per-device DMA-engine oversubscription, not round count, is the binding TPU wall-clock constraint. K=2 is the headline (makespan 92, vs cpsat_literal_warm's 78), saved as `fixtures/nonragged/schedule_8x4x4_loaded_spread_greedy_k2.json` with pre-generated Pallas kernel.

### TPU v5e wall-clock measurements (loaded 8×4×4)

Empirical AllToAll throughput of each generated Pallas kernel against the
reference P2P rotation kernel on TPU v5e, `slice=(8, 4, 4)`, routing =
`fixtures/routing/routing_table_8x4x4_twist.json`. All measurements use the same
payload and reference kernel scaffolding — only the iteration-order /
schedule-driven destination table differs.

| Schedule | Sim makespan | Rounds (≈ per-device DMAs in series) | Measured throughput | vs P2P |
|---|---:|---:|---:|---:|
| reference P2P (rotation) | ~340 (simulator 4–9× sub-LB) | 127 (= N − 1) | **134 541 gbps** | baseline |
| `orbit_greedy_full` (`lpt_tail_asc`) | 85 | ≤ 85 | 132 758 gbps | −1.3 % |
| `cpsat_literal_warm` (warm-started, `t_upper=79`, 4 h) | 78 | ≤ 78 | 132 764 gbps | −1.3 % |
| `spread_greedy(k=1)` (P2P-style, LB-aware order) | 145 | 145 | 132 682 gbps | −1.4 % |
| `spread_greedy(k=2)` (2-way pipelining) | 92 | 91 | 132764 gbps | —1.3 % |
| `cpsat_literal_warm_packed` (Option A: packed SMEM preamble) | 78 | ≤ 78 | TBD (to measure on TPU) | TBD |
| `cpsat_literal_warm_pipelined` (Option B: batch-127 wait drain) | 78 | ≤ 78 | TBD (to measure on TPU) | TBD |

**LB:** physical-edge lower bound on this routing = 75.

**What this table says:**

- **Simulator makespan is a poor predictor of TPU wall-clock at this scale.** The makespan-78 schedule (`cpsat_literal_warm`) and the makespan-145 schedule (`spread_greedy(k=1)`) measure within 0.1 % of each other (132 682 vs 132 764 gbps) — and both are within 1.4 % of P2P's makespan-340 rotation. The schedule barely moves the needle.
- **K=1 decomposition** (the one apples-to-apples per-round comparison vs P2P, since both emit 1 DMA per device per round): K=1 has 1.142× more rounds but is 1.014× slower wall-clock → implied **0.888×** per-round time (≈11 % faster per round than P2P). LB-aware destination ordering does pay off, but the 18 extra rounds (forced by physical-edge conflicts in the routing) eat almost all the gain.
- **If K=1 hit N−1=127 rounds at the same per-round time**, projected throughput would be ~151 500 gbps (+12.6 % vs P2P). That's the gap a better routing or near-LB scheduling could close.
- **The cluster at ~132 700 gbps** across three very different schedules (orbit_greedy K~5, cpsat_warm K=∞, spread_greedy K=1) means *something else* dominates wall-clock — per-DMA setup, HBM bandwidth, or per-step kernel overhead — not the round count, not the LB-awareness, not the per-device DMA cap.

### Packet-size sweep (`cpsat_literal_warm` kernel, 2026-05-18)

Hardware sensitivity to the kernel's `packet_size` knob (see [`pallas_kernel/reference_kernel.py:466-475`](pallas_kernel/reference_kernel.py#L466-L475)). All other settings held identical; only `packet_size` varies. Measured on `cpsat_literal_warm` (makespan 78) on the same TPU v5e setup as the table above.

| `packet_size` (bytes) | Measured throughput | vs 2¹⁵ peak |
|---|---:|---:|
| 2¹³ = 8 KB    | 126 987 gbps | −4.4 % |
| 2¹⁴ = 16 KB   | 130 743 gbps | −1.5 % |
| 2¹⁵ = 32 KB   | **132 764 gbps** | baseline (current default) |
| 2¹⁶ = 64 KB   | 123 356 gbps | −7.1 % |
| 2¹⁷ = 128 KB  | 111 780 gbps | −15.8 % |

**What this curve says:**

- **32 KB is the genuine sweet spot.** Throughput is convex with a peak at the current default — no easy win from changing this knob alone.
- **The dropoff is asymmetric.** Halving packet size loses only −1.5 % (16 KB) and quartering loses −4.4 % (8 KB), but doubling loses −7.1 % (64 KB) and quadrupling loses −15.8 % (128 KB). The right edge is a cliff (VMEM/pipeline collapse); the left edge is a gentle slope.
- **Per-DMA setup cost is bounded by ~5–10 % of total throughput.** If setup cost were ~50 % of per-packet wall-clock (the naïve reading of the inline-vs-regular kernel comparison), 4× more DMAs (8 KB vs 32 KB) would cost ~30 %, not 4.4 %. So the headroom available to *any* DMA-count-reduction strategy is bounded by ~5–10 %.
- **The asymmetry kills naïve coalescing.** Per-edge coalescing inherently increases per-descriptor payload (fusing N adjacent rounds → N× packet size). 2× coalescing → 64 KB → starts at −7.1 % baseline penalty; the DMA-count savings would have to *exceed* that just to break even. 4× coalescing → 128 KB → starts at −15.8 %; almost certainly net negative.

This refines the [coalescing upper-bound diagnostic](eval/explorations/2026-05-17-coalescing-upper-bound/) verdict: the *theoretical* coalescing factor was 14.7×, but the *achievable* factor under the 64–128 KB packet cap is ~1.5–2×. Combined with the ~5–10 % bound on total DMA-cost-savings, the realistic Option-3 net gain is probably ≤ a few percent — not the original 4.25× DMA reduction.

**Next data points needed:** TPU profiler trace of the 64 KB run (which microarchitectural resource is saturating?), and a sub-flow chunking probe (whether sub-32-KB chunking inside a per-flow DMA recovers more pipeline parallelism than 8 KB simple packets did).

### Two capacity models — the source of historical confusion

Earlier versions of this project claimed `orbit_greedy` was "LB-tight on
10/10 cells". That claim was correct in the **orbit-class capacity model**
(one orbit firing per `(dim, dir)` class per step), but the actual Pallas
kernel executes against the **physical-edge model** (one flow per directed
edge per step). The two are equivalent iff the routing is translation-
equivariant under the `(dim, dir)` action; for several non-trivial cells
they diverge, and the orbit-class LB is unattainable in the physical model.
See [docs/orbit_greedy_optimality.md §6](docs/orbit_greedy_optimality.md)
for the full reconciliation.

## Quickstart

```bash
uv venv
uv pip install -e ".[dev]"
uv run pytest                                # all tests should pass
bash eval/run_all.sh                          # run every experiment
cat results/$(date +%Y-%m-%d)/headlines.csv   # aggregated summary
```

## Layout

- `twisted_analysis/topology/` — twisted-torus lattice, DOR router, ILPRouter.
- `twisted_analysis/io/` — routing-table and schedule JSON I/O + flat-id utilities.
- `twisted_analysis/model/` — AllToAll workload, link load, lower bound.
- `twisted_analysis/schedules/` — RoundRobin, XLA, DimPhased, OrbitGreedy (headline), PipelinedOrbit (constrained variant), LP-optimal.
- `twisted_analysis/simulator/` — step-synchronous engine + instrumentation.
- `twisted_analysis/lp/` — time-indexed ILP + LP relaxation (PuLP/CBC).
- `twisted_analysis/viz/` — matplotlib plot helpers.
- `fixtures/` — persisted inputs/outputs, organized into three subfolders:
  - `routing/` — routing tables (`routing_table_<slice>_<router>.json`), loaded route caches (`routcache_torus_<coords>_twisted.json`), and legacy CSV from `scripts/dump_routing_tables.py`.
  - `nonragged/` — dense-AllToAll schedules (`schedule_<slice>_<router>_<order>.json`), plus a `cns_schedules/` subfolder of CNS-pipeline copies (and their `readme.md`).
  - `ragged/` — ragged workloads (`ragged_a2a_workload_<...>.json`) and their schedules (`schedule_<slice>_loaded_ragged_<scheduler>.json`), plus a `cns_schedules/` subfolder of ragged CNS copies. Ragged entries carry `rate`/`size` fields (see the 2026-07-14 spec).
- `pallas_kernel/` — Pallas TPU kernel generator (consumes a routing table + schedule, emits `outputs/_ragged_a2a_kernel_<scheduler>_<slice>.py`). Current outputs include `orbit_greedy_8_4_4`, `orbit_greedy_full_8_4_4`, `literal_greedy_8_4_4`, `cpsat_literal_warm_8_4_4` (the makespan-78 production recommendation for the loaded 8×4×4 routing), and `cpsat_literal_warm_inline_8_4_4` (same schedule with `--inline-destinations`: per-step destinations baked as `jax.lax.switch` branches instead of an SMEM `dest_table_ref` input), and `spread_greedy_k2_8_4_4` / `spread_greedy_k2_inline_8_4_4` (per-device DMA-cap K=2; testbed for the DMA-oversubscription hypothesis from 2026-05-17), and `cpsat_literal_warm_torus_2_2_4` / `cpsat_literal_warm_torus_2_4_4` (non-twisted-torus AllToAll on slices (2,2,4) and (2,4,4); see [eval/explorations/2026-05-23-cpsat-warm-non-twisted/](eval/explorations/2026-05-23-cpsat-warm-non-twisted/)), and `orbit_pack_k{2,3,6}c3_8_4_4_pfc` (step-model orbit packing for TPU v4 per-step-barrier execution: 64/43/27 barrier steps at whole-path edge cap 3; regenerate via `eval/regenerate_8x4x4_orbit_pack_kernels.sh`).
- `scripts/` — reproducible CLIs:
  - `generate_routing_table.py` — `(slice, router) → fixtures/routing/routing_table_<slice>_<router>.json`
  - `generate_schedule.py` — `(routing-table, scheduler, order) → fixtures/nonragged/schedule_<slice>_<router>_<order>.json`
- `experiments/` — one YAML per experiment.
- `eval/run_all.sh` — reproduces everything.
- `docs/` — algorithm, topology, schedules, LP, evaluation, results.

## Pipeline

The end-to-end TPU-kernel pipeline runs in three stages, each producing an inspectable on-disk artifact:

1. **Router** → `fixtures/routing/routing_table_<slice>_<router>.json`. Matrix of paths (`[src][dst] → {"path": [{"node_id": int}, ...]}`). Run via `scripts/generate_routing_table.py` or call `twisted_analysis.io.save_routing_table` directly. Convention: paths are sequences of single-hop topology neighbors; flatten convention is dim-0 most significant (e.g. slice `(8,4,4)` → `flat = i*16 + j*4 + k`).
2. **Scheduler** → `fixtures/nonragged/schedule_<slice>_<router>_<order>.json`. Flat list of `{round, src, dst, path}` entries; `path` is a list of flat-IDs from src to dst. Run via `scripts/generate_schedule.py`.
3. **Kernel generator** → `pallas_kernel/outputs/_ragged_a2a_kernel_<scheduler>_<slice>.py`. Run via `pallas_kernel/gen_orbit_greedy_kernel.py`, which orchestrates stages 1 and 2 (or accepts an existing routing table via `--routing-table FILE`, or a precomputed schedule via `--schedule-in FILE`).

Pre-generated example: `fixtures/routing/routing_table_8x4x4_twist.json` is a 4×4×8 TPU v5e twisted torus represented as slice=(8,4,4) under our flatten convention (largest dim first per `{S,2S}^n`). Use it via:
```
python pallas_kernel/gen_orbit_greedy_kernel.py \
    --slice 8,4,4 \
    --routing-table fixtures/routing/routing_table_8x4x4_twist.json \
    --order lpt_tail_asc
```
This produces `fixtures/nonragged/schedule_8x4x4_loaded_lpt_tail_asc.json` and `pallas_kernel/outputs/_ragged_a2a_kernel_orbit_greedy_8_4_4.py` (makespan 85, the orbit-greedy baseline).

**For the production-recommended makespan-78 kernel** (warm-started CP-SAT, projected +7.5% above the P2P reference), load the precomputed schedule directly:
```
python pallas_kernel/gen_orbit_greedy_kernel.py \
    --slice 8,4,4 \
    --routing-table fixtures/routing/routing_table_8x4x4_twist.json \
    --schedule-in fixtures/nonragged/schedule_8x4x4_loaded_cpsat_literal_warm.json
```
The pre-generated kernel is at [`pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_8_4_4.py`](pallas_kernel/outputs/_ragged_a2a_kernel_cpsat_literal_warm_8_4_4.py).

## Docs

Individual documentation files:

| File | Description |
|---|---|
| [docs/algorithm.md](docs/algorithm.md) | Cost model and lower-bound proof. |
| [docs/topology.md](docs/topology.md) | Twisted-torus neighbor function, DOR routing, ILPRouter. |
| [docs/schedules.md](docs/schedules.md) | All schedule implementations (RoundRobin, XLA, DimPhased, OrbitGreedy, ILP-optimal). |
| [docs/orbit_greedy_optimality.md](docs/orbit_greedy_optimality.md) | OrbitGreedy with `lpt_tail_asc`: setting, algorithm, König + Smith's-deadline-feasibility proof of `makespan = LB` (machine-verified per cell), and the tail-load tiebreak analysis. |
| [docs/lp_formulation.md](docs/lp_formulation.md) | ILP and symmetric scheduling ILP formulation. |
| [docs/evaluation.md](docs/evaluation.md) | Reproduction guide and experiment matrix. |
| [docs/results.md](docs/results.md) | Measured headline numbers and key findings. |
| [docs/superpowers/specs/](docs/superpowers/specs/) | Design specs. |
| [docs/superpowers/plans/](docs/superpowers/plans/) | Implementation plans. |

### Reference: twisted-torus neighbor function

```python
def twisted_torus_neighbor(node, slice, ndim, ndir):
    assert all(s in (min(slice), 2 * min(slice)) for s in slice)
    neighbor = list(node)
    neighbor[ndim] += ndir
    wrapped = neighbor[ndim] < 0 or neighbor[ndim] >= slice[ndim]
    if wrapped:
        for i in range(len(neighbor)):
            neighbor[i] = (neighbor[i] + slice[ndim]) % slice[i]
    return neighbor
```
