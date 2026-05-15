# TwistedAnalysis

Quantifies the AllToAll performance gap on twisted-torus topologies under
load-balanced minimal routing (ILPRouter) and dimension-order routing (DOR).

## What

For twisted-torus topologies in the `{S, 2S}` shape family (2×4, 2×2×4, 2×4×4,
4×8, 4×4×8 — i.e. 2D and 3D shapes with one or more twisted dimensions), we
compute:

1. The bandwidth lower bound `LB` from max directed-link load. The default router is
   **ILPRouter** (load-balanced minimal routing, LP-based); `router: dor` selects
   dimension-order routing. ILP routing reduces LB by 14–31% vs. DOR.
2. The makespan `M_S` for several schedules and report `gap(S) = M_S / LB`:
   - **OrbitGreedy** (constructive, no ILP, default order=`lpt_tail_asc`) —
     **headline scheduler**. `makespan = LB` on every (topology, router) cell
     tested (10/10). Plain `lpt` (no tail-load tiebreak) misses 2×4×4 DOR by
     1 step; the tail-load-ascending tiebreak prevents low-load tail edges
     from rolling the makespan past LB.
   - **PipelinedOrbit** (constructive, **not optimal**) — same orbit ordering
     as OrbitGreedy plus the extra constraint *gap=1 between consecutive hops
     of the same orbit* (i.e., `t_{i+1} = t_i + 1`). Strict subset of
     OrbitGreedy's solution space. Achieves LB on 7/10 cells; gaps of 1–5
     steps on 4×8 ilp, 4×4×8 dor, 4×4×8 ilp, 2×4×4 ilp. Use as a
     pipelined-injection diagnostic, not as the production scheduler.
   - Latin-square round-robin (full AllToAll, 4–9× gap due to phase-boundary idle).
   - XLA (destination-core randomization; bijection on phases, equal to RoundRobin in
     our model).
   - Dimension-ordered phased (partial coverage only).
3. Optimal-makespan references:
   - **Full ILP** (`ilp_optimal`): small instances only.
   - **Symmetric ILP** (`ilp_optimal_symmetric`): made 4×8 tractable (~14 s) before
     OrbitGreedy reproduced it in microseconds. LP relaxation `M_LP` is reported but
     is a weak bound (minimizes weighted-completion-time, not makespan).

A gap of 1 means the routing+schedule saturate every bottleneck link; >1 quantifies
the inefficiency.

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
- `fixtures/` — persisted routing tables and schedules (`routing_table_<slice>_<router>.json`, `schedule_<slice>_<router>_<order>.json`); also legacy CSV from `scripts/dump_routing_tables.py`.
- `pallas_kernel/` — Pallas TPU kernel generator (consumes a routing table + schedule, emits `outputs/_ragged_a2a_kernel_orbit_greedy_<slice>.py`).
- `scripts/` — reproducible CLIs:
  - `generate_routing_table.py` — `(slice, router) → fixtures/routing_table_<slice>_<router>.json`
  - `generate_schedule.py` — `(routing-table, scheduler, order) → fixtures/schedule_<slice>_<router>_<order>.json`
- `experiments/` — one YAML per experiment.
- `eval/run_all.sh` — reproduces everything.
- `docs/` — algorithm, topology, schedules, LP, evaluation, results.

## Pipeline

The end-to-end TPU-kernel pipeline runs in three stages, each producing an inspectable on-disk artifact:

1. **Router** → `fixtures/routing_table_<slice>_<router>.json`. Matrix of paths (`[src][dst] → {"path": [{"node_id": int}, ...]}`). Run via `scripts/generate_routing_table.py` or call `twisted_analysis.io.save_routing_table` directly. Convention: paths are sequences of single-hop topology neighbors; flatten convention is dim-0 most significant (e.g. slice `(8,4,4)` → `flat = i*16 + j*4 + k`).
2. **Scheduler** → `fixtures/schedule_<slice>_<router>_<order>.json`. Flat list of `{round, src, dst, path}` entries; `path` is a list of flat-IDs from src to dst. Run via `scripts/generate_schedule.py`.
3. **Kernel generator** → `pallas_kernel/outputs/_ragged_a2a_kernel_orbit_greedy_<slice>.py`. Run via `pallas_kernel/gen_orbit_greedy_kernel.py`, which orchestrates stages 1 and 2 (or accepts an existing routing table via `--routing-table FILE`).

Pre-generated example: `fixtures/routing_table_8x4x4_twist.json` is a 4×4×8 TPU v5e twisted torus represented as slice=(8,4,4) under our flatten convention (largest dim first per `{S,2S}^n`). Use it via:
```
python pallas_kernel/gen_orbit_greedy_kernel.py \
    --slice 8,4,4 \
    --routing-table fixtures/routing_table_8x4x4_twist.json \
    --order lpt_tail_asc
```
This produces `fixtures/schedule_8x4x4_loaded_lpt_tail_asc.json` and `pallas_kernel/outputs/_ragged_a2a_kernel_orbit_greedy_8_4_4.py`.

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
