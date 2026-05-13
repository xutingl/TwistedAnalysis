# Evaluation

## Experiment Matrix

22 experiments per `eval/run_all.sh`: 3 topologies × 2 routers × {round_robin,
xla, dim_phased, ilp_optimal, ilp_optimal_symmetric} where applicable.

| Topology | Router | RoundRobin | XLA | DimPhased | ILP-optimal | Symmetric ILP |
|---|---|:---:|:---:|:---:|:---:|:---:|
| 2×4 | ilp | ✓ | ✓ | ✓ (partial) | ✓ | ✓ |
| 2×4 | dor | ✓ | ✓ | ✓ (partial) | ✓ | — |
| 4×8 | ilp | ✓ | ✓ | ✓ (partial) | — (intractable) | ✓ (~14s) |
| 4×8 | dor | ✓ | ✓ | ✓ (partial) | — | — |
| 4×4×8 | ilp | ✓ | ✓ | ✓ (partial) | — | — (~47k vars) |
| 4×4×8 | dor | ✓ | ✓ | ✓ (partial) | — | — |

Per experiment: makespan, ratio to `LB`, idle-step count on bottleneck edges,
runtime.

**DimPhased caveat.** DimPhased covers only pairs differing in a single dimension.
The reported `ratio = makespan / sched-LB` uses the schedule's own subset lower
bound (so it is ≥ 1.0), but is not directly comparable to RoundRobin or ILP
ratios which cover the full workload. The `full-LB` column is reported separately.
See [schedules.md](schedules.md).

## Reproducing Results

Install the environment once:

```bash
uv venv
uv pip install -e ".[dev]"
```

Run all experiments:

```bash
bash eval/run_all.sh
```

Results land under `results/<YYYY-MM-DD>/`, one subdirectory per experiment YAML.
A summary CSV is aggregated at `results/<YYYY-MM-DD>/headlines.csv`.

### What `run_all.sh` does

1. Iterates over every `experiments/*.yaml`.
2. For each YAML, patches the `output_dir` to `results/<date>/<name>`, then calls:
   ```
   python -m twisted_analysis.cli run <patched_yaml>
   ```
3. Aggregates all `summary.json` files into `headlines.csv`.

Each experiment writes:
- `summary.json` — name, slice, msg_size, schedule, lower_bound, makespan, ratio,
  bottleneck_edges, idle_steps_on_bottleneck.
- `gantt.csv` — per-unit injection / per-hop / delivery steps.
- `heatmap.png` — links × time utilization heatmap (if enabled in YAML).

## Running a Single Experiment

```bash
.venv/bin/python -m twisted_analysis.cli run experiments/2x4_ilp.yaml
```

Or directly from Python:

```python
from twisted_analysis.topology import Topology, ILPRouter
from twisted_analysis.model import AllToAll
from twisted_analysis.schedules import RoundRobinSchedule
from twisted_analysis.simulator.engine import Simulator

t = Topology(slice=(2, 4))
r = ILPRouter(t)              # or DORRouter(t) for dimension-order routing
w = AllToAll(t, r, msg_size=1)
schedule = RoundRobinSchedule()
injections = schedule.emit(w)
sim = Simulator(t, r, list(w.flows))
for inj in injections:
    sim.inject(inj)
makespan = sim.run()
print(makespan, '/', w.lower_bound, '=', makespan / w.lower_bound)
```

`Router` is a Protocol type; the concrete implementations are `DORRouter` and
`ILPRouter`.

## Adding a New Experiment

Create a YAML file under `experiments/`. The schema is:

```yaml
name: <experiment_name>          # identifier; used for output directory name
slice: [2, 4]                    # topology shape; must satisfy {S, 2S} constraint
msg_size: 1                      # message size in flow units (integer >= 1)
schedule: round_robin            # see "Allowed schedule values" below
router: ilp                      # ilp (default) or dor
output_dir: results/<name>       # overridden by run_all.sh; set for manual runs
```

Allowed `schedule` values:

| Value | Class |
|---|---|
| `round_robin` | `RoundRobinSchedule` (Latin-square, full coverage) |
| `xla` | `XLASchedule` (XLA destination-core randomization, full coverage) |
| `dim_phased` | `DimPhasedSchedule` (partial coverage: one-dim-diff pairs only) |
| `ilp_optimal` | runs full ILP, extracts LP-optimal schedule |
| `ilp_optimal_symmetric` | runs symmetric ILP, extracts orbit-level schedule |

Allowed `router` values:

| Value | Class |
|---|---|
| `ilp` (default) | `ILPRouter` — load-balanced minimal routing |
| `dor` | `DORRouter` — dimension-order routing |

`run_all.sh` automatically picks up any new YAML in `experiments/` — no other
change is needed.

## Tests

```bash
uv run pytest          # or: .venv/bin/python -m pytest
```

71 unit tests + 2 slow tests (gated by `-m slow`). Test categories:

| File | What it checks |
|---|---|
| `tests/test_topology_neighbor.py` | neighbor() correctness against reference |
| `tests/test_topology_links_bfs.py` | BFS diameter equals expected value |
| `tests/test_router.py` | DOR hop count equals BFS distance; deterministic |
| `tests/test_router_protocol.py` | Router Protocol; DORRouter and ILPRouter conform |
| `tests/test_ilp_router.py` | ILPRouter correctness + LB ≤ DOR's LB |
| `tests/test_routing_fixtures.py` | committed DOR + ILP routing fixtures regression |
| `tests/test_model.py` | load aggregation; LB = max load |
| `tests/test_schedules_base.py` | Injection / Schedule base types |
| `tests/test_round_robin.py` | RoundRobin makespan ≥ LB |
| `tests/test_dim_phased.py` | DimPhased makespan and phase coverage |
| `tests/test_simulator.py` | step-by-step simulation correctness |
| `tests/test_instrumentation.py` | record_history / event logging |
| `tests/test_ilp.py` | tiny ILP instance solved to known optimum |
| `tests/test_lp_relaxation.py` | LP relaxation bound ≤ ILP optimum |
| `tests/test_lp_replay.py` | LP-extracted schedule replays correctly |
| `tests/test_orbit.py` | translational orbit detection |
| `tests/test_symmetric_ilp.py` | symmetric ILP matches asymmetric on 2×4 |
| `tests/test_symmetric_scale.py` (slow) | 4×8 symmetric ILP tractable; 4×8 zero-gap regression |
| `tests/test_bounds.py` | bisection-bandwidth lower bound |
| `tests/test_cli.py` | CLI run command end-to-end (router + schedule dispatch) |
| `tests/test_viz.py` | plot functions write non-empty PNG files |
| `tests/test_smoke.py` | full pipeline smoke test |

Run fast tests only: `uv run pytest -v -m "not slow"`.
Run including slow: `uv run pytest -v`.

## Future Work / Ablations

- **DOR dim-order ablation.** Smallest-dim-first vs. largest-dim-first routing:
  effect on `LB` and makespan.
- **m-sweep.** Run all schedules with `msg_size ∈ {1, 4, 16}` to test gap invariance.
- **Symmetric ILP on 4×4×8.** ~47k variables at T = LB = 74; tractability unknown.
- **Pipelined RoundRobin / XLA.** Overlap phases instead of running back-to-back.
  In the current step-sync model, XLA's randomization gives the same makespan as
  RoundRobin (see [results.md](results.md) finding 6); pipelined injection would
  let randomization actually matter.

## See Also

- [results.md](results.md) — actual headline numbers from the first eval run.
- [schedules.md](schedules.md) — schedule definitions and their caveats.
- [lp_formulation.md](lp_formulation.md) — ILP details for `ilp_optimal` experiments.
