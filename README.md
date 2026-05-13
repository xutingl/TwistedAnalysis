# TwistedAnalysis

Quantifies the AllToAll performance gap on twisted-torus topologies under fixed
dimension-order routing.

## What

For twisted-torus topologies in the `{S, 2S}` shape family (e.g., 2x4, 4x8, 4x4x8),
we compute:

1. The bandwidth lower bound `LB` from max directed-link load under fixed DOR.
2. The ILP-optimal makespan `M_opt` (small instances) and the LP relaxation `M_LP`.
3. The makespan `M_S` for two heuristic schedules: Latin-square round-robin, and
   dimension-ordered phased.

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

- `twisted_analysis/topology/` — twisted-torus lattice + DOR router.
- `twisted_analysis/model/` — AllToAll workload, link load, lower bound.
- `twisted_analysis/schedules/` — RoundRobin, DimPhased, LP-optimal.
- `twisted_analysis/simulator/` — step-synchronous engine + instrumentation.
- `twisted_analysis/lp/` — time-indexed ILP + LP relaxation (PuLP/CBC).
- `twisted_analysis/viz/` — matplotlib plot helpers.
- `experiments/` — one YAML per experiment.
- `eval/run_all.sh` — reproduces everything.
- `docs/` — algorithm, topology, schedules, LP, evaluation, results.

See [docs/](docs/) for details and [the design spec](docs/superpowers/specs/2026-05-12-twisted-torus-alltoall-design.md).

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
