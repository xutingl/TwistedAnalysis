# TwistedAnalysis

Quantifies the AllToAll performance gap on twisted-torus topologies under
load-balanced minimal routing (ILPRouter) and dimension-order routing (DOR).

## What

For twisted-torus topologies in the `{S, 2S}` shape family (e.g., 2x4, 4x8, 4x4x8),
we compute:

1. The bandwidth lower bound `LB` from max directed-link load. The default router is
   **ILPRouter** (load-balanced minimal routing, LP-based); `router: dor` selects
   dimension-order routing. ILP routing reduces LB by 14–25% vs. DOR.
2. The ILP-optimal makespan `M_opt` (small instances). A symmetric ILP variant
   (translational orbits) made 4×8 tractable (~14 s, ratio = 1.00). LP relaxation
   `M_LP` is used for 4×4×8.
3. The makespan `M_S` for heuristic schedules:
   - Latin-square round-robin (full AllToAll coverage).
   - XLA (replicates XLA's destination-core randomization; bijection on phases, same makespan as round-robin in our model).
   - Dimension-ordered phased (partial coverage only).

The headline metric is `gap(S) = M_S / LB`. A gap of 1 means the routing+schedule
saturate every bottleneck link; >1 quantifies the inefficiency.

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
- `twisted_analysis/model/` — AllToAll workload, link load, lower bound.
- `twisted_analysis/schedules/` — RoundRobin, XLA, DimPhased, LP-optimal.
- `twisted_analysis/simulator/` — step-synchronous engine + instrumentation.
- `twisted_analysis/lp/` — time-indexed ILP + LP relaxation (PuLP/CBC).
- `twisted_analysis/viz/` — matplotlib plot helpers.
- `experiments/` — one YAML per experiment.
- `eval/run_all.sh` — reproduces everything.
- `docs/` — algorithm, topology, schedules, LP, evaluation, results.

## Docs

Individual documentation files:

| File | Description |
|---|---|
| [docs/algorithm.md](docs/algorithm.md) | Cost model and lower-bound proof. |
| [docs/topology.md](docs/topology.md) | Twisted-torus neighbor function, DOR routing, ILPRouter. |
| [docs/schedules.md](docs/schedules.md) | All schedule implementations (RoundRobin, XLA, DimPhased, ILP-optimal). |
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
