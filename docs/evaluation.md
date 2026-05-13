# Evaluation

## Experiment Matrix

| Topology | `LB` | ILP optimum | RoundRobin | DimPhased |
|---|---|---|---|---|
| 2×4 (t=2) | computed | solved (tractable) | measured | measured (partial) |
| 4×8 (t=4) | computed | not run (too large) | measured | measured (partial) |
| 4×4×8 | computed | not run (too large) | measured | measured (partial) |

Per experiment: makespan, ratio to `LB`, idle-step count on bottleneck edges,
runtime.

**DimPhased caveat.** DimPhased covers only pairs differing in a single dimension.
Its ratio is computed against the full-AllToAll `LB` and can be below 1.0, which
simply means fewer flows were scheduled. It is a diagnostic for per-dim efficiency,
not a comparison to the full workload. See [schedules.md](schedules.md).

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
from twisted_analysis.topology import Topology, Router
from twisted_analysis.model import AllToAll
from twisted_analysis.schedules import RoundRobinSchedule
from twisted_analysis.simulator.engine import Simulator

t = Topology(slice=(2, 4))
r = Router(t)
w = AllToAll(t, r, msg_size=1)
schedule = RoundRobinSchedule()
injections = schedule.emit(w)
sim = Simulator(t, r, list(w.flows))
for inj in injections:
    sim.inject(inj)
makespan = sim.run()
print(makespan, '/', w.lower_bound, '=', makespan / w.lower_bound)
```

## Adding a New Experiment

Create a YAML file under `experiments/`. The schema is:

```yaml
name: <experiment_name>          # identifier; used for output directory name
slice: [2, 4]                    # topology shape; must satisfy {S, 2S} constraint
msg_size: 1                      # message size in flow units (integer >= 1)
schedule: round_robin            # one of: round_robin, dim_phased, ilp_optimal
output_dir: results/<name>       # overridden by run_all.sh; set for manual runs
```

Allowed `schedule` values:

| Value | Class |
|---|---|
| `round_robin` | `RoundRobinSchedule` |
| `dim_phased` | `DimPhasedSchedule` |
| `ilp_optimal` | runs ILP, extracts LP-optimal schedule |

`run_all.sh` automatically picks up any new YAML in `experiments/` — no other
change is needed.

## Tests

```bash
uv run pytest          # or: .venv/bin/python -m pytest
```

53 unit tests. Test categories:

| File | What it checks |
|---|---|
| `tests/test_topology_neighbor.py` | neighbor() correctness against reference |
| `tests/test_topology_links_bfs.py` | BFS diameter equals expected value |
| `tests/test_router.py` | DOR hop count equals BFS distance; deterministic |
| `tests/test_routing_fixtures.py` | shared routing fixtures used across tests |
| `tests/test_model.py` | load aggregation; LB = max load |
| `tests/test_schedules_base.py` | Injection / Schedule base types |
| `tests/test_round_robin.py` | RoundRobin makespan >= LB |
| `tests/test_dim_phased.py` | DimPhased makespan and phase coverage |
| `tests/test_simulator.py` | step-by-step simulation correctness |
| `tests/test_instrumentation.py` | record_history / event logging |
| `tests/test_ilp.py` | tiny ILP instance solved to known optimum |
| `tests/test_lp_relaxation.py` | LP relaxation bound <= ILP optimum |
| `tests/test_lp_replay.py` | LP-extracted schedule replays correctly |
| `tests/test_bounds.py` | lower-bound computations |
| `tests/test_cli.py` | CLI run command end-to-end |
| `tests/test_viz.py` | plot functions write non-empty PNG files |
| `tests/test_smoke.py` | full pipeline smoke test |

## Future Work / Ablations

The following experiments are listed in the design spec but not yet run:

- **DOR dim-order ablation.** Smallest-dim-first vs. largest-dim-first routing:
  effect on `LB` and makespan.
- **m-sweep.** Run all schedules with `msg_size ∈ {1, 4, 16}` to test gap invariance.
- **ILP on 4×8.** Tractability with symmetry reduction (fixing one node's start
  step, orbit-based symmetry breaking); currently not attempted.
- **Latin-square phase-order ablation.** Ordering the round-robin phases `r` by
  twist-induced shape to see if phase reordering narrows the gap.

## See Also

- [results.md](results.md) — actual headline numbers from the first eval run.
- [schedules.md](schedules.md) — schedule definitions and their caveats.
- [lp_formulation.md](lp_formulation.md) — ILP details for `ilp_optimal` experiments.
