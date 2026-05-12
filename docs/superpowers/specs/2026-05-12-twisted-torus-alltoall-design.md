# TwistedAnalysis — Design Spec

- **Date:** 2026-05-12
- **Status:** Draft (pending user review)
- **Scope:** Quantify the AllToAll performance gap on twisted-torus topologies under fixed dimension-order routing.

## 1. Problem

For a twisted-torus topology with shape `R x C` (later `R x C x D`) where the *smaller* dimension is twisted by a half-shift `t = C/2`, and a **fixed** dimension-order routing (DOR) table mapping every `(src, dst)` to a single deterministic path, we want to quantify the gap between:

- a bandwidth-based **lower bound** `LB` on AllToAll makespan (from max link load under the fixed routing), and
- the makespan `M_S` achievable by various flow-scheduling strategies `S`.

The headline metric is `gap(S) = M_S / LB`. A gap of 1 means the routing+schedule are bandwidth-efficient; >1 means the schedule cannot keep every bottleneck link saturated. We also separately report `gap_routing = M_opt / LB`, where `M_opt` is the ILP-optimal makespan — this isolates the *intrinsic* hardness from the routing table alone (independent of any heuristic schedule).

### Workload

- **Personalized AllToAll, uniform size.** Each of `N = prod(slice)` nodes sends a distinct message of size `m` units to each of the other `N - 1` nodes. Default `m = 1`; sweeps over `m ∈ {1, 4, 16}` are an ablation.
- No self-loops.

### Cost model — step-synchronous, fluid, bandwidth-only

- Time is discrete integer steps.
- Each *directed* link has capacity 1 flow-unit per step; full-duplex (the two directions are independent links).
- Store-and-forward: unit `k` of flow `f` may use link `e_{i+1}` at step `s+1` only if it used `e_i` at step `s`.
- At each link, at most one flow-unit per step.
- Per-hop latency `alpha` is omitted; message latency = path length × 1 step, dominated by bandwidth for `m` large. Reported separately so an alpha-beta translation is trivial.

### Lower bound

For any directed edge `e`, let `load(e) = m × |{f : e ∈ path(f)}|`. Then `LB = max_e load(e)`. Proof sketch: any feasible schedule moves at most one flow-unit across `e` per step, so `makespan ≥ load(e)` for every `e`. Tightness — existence of a schedule achieving `LB` — is *not* guaranteed; the gap is exactly what we are measuring.

## 2. Topology and routing

### Twisted-torus convention

We adopt the neighbor function in the project README as canonical. For a `slice = (s_0, ..., s_{d-1})` satisfying `∀i. s_i ∈ {S, 2S}` where `S = min(slice)`:

```
def neighbor(node, dim, dir):
    n = list(node); n[dim] += dir
    if 0 <= n[dim] < slice[dim]:
        return tuple(n)                          # in-plane step
    return tuple((n[i] + slice[dim]) % slice[i]  # wrap: shift all coords by slice[dim]
                 for i in range(len(n)))
```

Consequences:

| Wrap | Effective behavior on `(R x 2R)` slices |
|---|---|
| Smaller dim (size `R`) | shifts longer dim by `R = C/2` → **half-shift twist** |
| Longer dim (size `2R`) | shifts smaller dim by `2R mod R = 0` → plain torus wrap |

For `4 x 4 x 8` (3D), both smaller dims twist into the size-8 dim by 4; the size-8 dim wraps cleanly.

### Routing table — DOR on twisted torus

The smaller-dim wraparound displaces the longer dim, so DOR cannot independently pick a wrap direction per dim. The router:

1. Enumerates the small set of candidate displacement vectors `(δ_0, δ_1, ...)` that satisfy `src + δ ≡ dst` modulo twist (typically 2 per dim — wrap and no-wrap — with the twist's induced cross-dim shift factored in).
2. Picks the candidate with minimum total hop count. Deterministic tie-break: prefer no-wrap; then prefer `+dir`.
3. Walks the chosen displacements in fixed dim order (default: largest-dim first; smallest-dim first is an ablation).

Every `(src, dst)` maps to exactly one path — a true fixed table.

**Validation.** All-pairs BFS on the topology yields the true shortest-hop distance. We assert `len(DOR.path(s, d)) == BFS.dist(s, d)` for every `(s, d)` on every topology. Routing tables are dumped to `fixtures/routing_<topology>.csv` for inspection.

## 3. Flow model and link load

A `Flow` is `(src, dst, size)`. The AllToAll workload materializes `flows = [Flow(s, d, m) for s, d in S×S if s != d]`.

Given the router, every flow has `path(f) = (e_1, ..., e_h)`. The link-load table `load: Edge → int` is computed once; `LB = max(load.values())`. We additionally expose:

- `bottleneck_edges()` — every edge attaining `LB`.
- A load histogram across all directed edges (skew indicator).
- A "twist-attributed" decomposition: how much of `load(e)` is from flows whose path crosses a twisted wraparound link.

We attempt closed-form `LB(R, C)` and `LB(R, C, D)` expressions using the vertex-transitive symmetry of the twisted torus, falling back to numerical computation when the closed form does not emerge cleanly.

## 4. Schedules

All schedules implement a common interface:

```python
class Schedule(Protocol):
    name: str
    def emit(self, workload: AllToAll) -> list[Injection]: ...
```

An `Injection` is `(flow, start_step, link_priorities)`. The simulator consumes injections and produces the realized makespan — schedules are pure coordination policies and never bypass the cost model.

### Schedule A — Round-robin / Latin-square

- `N - 1` phases; in phase `r ∈ {1..N-1}`, node `i` sends its size-`m` message to node `(i + r) mod N` (flat-id ordering via `ravel(coords, slice)`).
- All `N` flows in a phase are injected at the same step; the next phase starts when the current one drains.
- Ablation: topology-aware ordering of `r` by the twist-induced "shape" of `r`.

### Schedule B — Dimension-ordered phases

- 2D `(R, C)`: Phase 1 does AllToAll along the longer dim within each row (`R` independent rings of size `C`); Phase 2 does AllToAll along the shorter dim within each column.
- 3D: three sequential phases, one per dim.
- Each phase only uses one dim's links → phases never contend with each other. Natural fit for DOR; clean closed-form link-load formulas per phase.

### Schedule C — LP-optimal (small instances)

The makespan-minimizing schedule extracted from the ILP (§5). Defines the *ceiling*: `gap_routing = M_opt / LB`. On instances where the ILP is intractable we fall back to the LP relaxation lower bound `M_LP` and report it instead.

### Result object

A `ScheduleResult` carries:

- `makespan`,
- `ratio = makespan / LB`,
- per-step link-utilization trace,
- idle-step count per bottleneck edge,
- a Gantt-style injection log (CSV).

## 5. LP / ILP formulation

Built on PuLP with CBC by default; Gurobi optional via solver argument.

### Variables

- `x[f, i, t] ∈ {0, 1}` for flow `f`, hop index `i ∈ [0, len(path(f)))`, step `t ∈ [0, T)`.
- Interpretation: a unit of flow `f` traverses `path(f)[i]` at step `t`.

### Constraints

1. **Per-hop fire-once.** `Σ_t x[f, i, t] == size(f)` for every `(f, i)`. With `m = 1` this is 1.
2. **Causal order (store-and-forward).** For every `(f, i, s)`: `Σ_{t ≤ s} x[f, i+1, t] ≤ Σ_{t ≤ s-1} x[f, i, t]`.
3. **Link capacity.** For every `(e, t)`: `Σ x[f, i, t] over (f, i) with path(f)[i] = e ≤ 1`.

### Objective

- **Binary-search-on-`T`** (preferred): fix `T`, solve feasibility, binary-search the minimum `T`.
- **Direct minimization** of an explicit makespan variable is available as an alternate solve.

### LP relaxation

Drop integrality on `x`. Produces a fractional lower bound `M_LP` with `LB ≤ M_LP ≤ M_opt`. Always reported alongside `LB` to give a tighter analytical floor.

### Sizing

- `2x4` (N=8, 56 flows): ILP tractable in seconds.
- `4x8` (N=32, ~1000 flows): ILP "best effort" with symmetry-reduction; LP relaxation always reported.
- `4x4x8` (N=128, ~16k flows): LP relaxation only.

### Outputs

`M_opt` (or `M_LP`), solver runtime, and the per-step schedule extracted from the variables (consumed by Schedule C and by the simulator-replay validation in §6).

## 6. Simulator

Step-synchronous, deterministic, single-process Python.

### State

- `link_queue: dict[Edge, deque[Unit]]`
- `flow_state: dict[Flow, int]` — current hop index per unit (`m > 1` is unrolled into `m` independent units sharing a path)
- `clock: int`

### Step semantics

1. Inject every `Injection` with `start_step == clock` onto its flow's first hop.
2. Each link with a non-empty queue selects one unit per the schedule's per-link priority (default FIFO). At most one transmission per directed link per step.
3. Transmitted units advance one hop (`dequeue e_i`, `enqueue e_{i+1}`) or are marked delivered if `i` was last.
4. `clock += 1`.

Terminates when all units are delivered; returns `makespan = clock`.

### Instrumentation

- Per-step link occupancy mask → heatmap.
- Per-flow injection / per-hop / delivery steps → Gantt CSV.
- Per-bottleneck-edge idle-step trace, with annotations of what was queued elsewhere → "where does the gap come from" analysis.

### Validation

For every instance where the ILP solves, extract its `x[f, i, t]` schedule, feed it to the simulator, and assert that the realized makespan equals `M_opt` and per-step link usage matches the LP. Catches simulator bugs and LP-encoding bugs jointly.

### Performance budget

- `2x4`: milliseconds per run.
- `4x8`: seconds.
- `4x4x8`: ~30–60 s per schedule (acceptable; no premature optimization).

## 7. Evaluation plan

### Matrix

| Topology | `LB` | LP relax | ILP optimum | Schedule A (RR) | Schedule B (DOR-phased) |
|---|---|---|---|---|---|
| 2x4 (t=2) | ✓ | ✓ | ✓ | ✓ | ✓ |
| 4x8 (t=4) | ✓ | ✓ | best-effort | ✓ | ✓ |
| 4x4x8 | ✓ | ✓ | likely intractable | ✓ | ✓ |

Per cell: makespan, ratio to `LB`, per-bottleneck idle-step count, runtime.

### Plots (per topology)

1. Bar chart: `LB`, `M_LP`, `M_opt` (if solved), `M_A`, `M_B`, normalized by `LB`.
2. Per-link load histogram under DOR.
3. Bottleneck idle-time per schedule.
4. Per-step link-utilization heatmap (links × time) for `2x4`.

### Ablations

- DOR dim order (largest-first vs smallest-first): effect on `LB` and `M_S`.
- Latin-square round ordering by twist-shape: helps?
- `m` sweep: gap-invariance test.

### Headline question

For each of `2x4`, `4x8`, `4x4x8` under DOR routing:

- Is `LB` achievable? (Does `M_opt == LB`?)
- If not, what is the unavoidable gap `M_opt / LB`?
- How close do A and B come to `M_opt`?
- Which class of flows / which link is the gap-creator?

### Reproducibility

Every experiment is a YAML under `experiments/`. `eval/run_all.sh` runs every experiment, stores results under `results/<date>/`, and regenerates plots. Single command to reproduce: `bash eval/run_all.sh`.

## 8. Project structure

```
TwistedAnalysis/
├── README.md
├── pyproject.toml
├── uv.lock
├── .venv/                            # gitignored
├── twisted_analysis/
│   ├── __init__.py
│   ├── topology/
│   │   ├── lattice.py                # Topology, neighbor(), links(), BFS
│   │   └── router.py                 # DOR router, table dump/load
│   ├── model/
│   │   ├── flow.py                   # Flow, AllToAll workload, link load, LB
│   │   └── bounds.py                 # closed-form LB; bottleneck reporting
│   ├── schedules/
│   │   ├── base.py                   # Schedule protocol, Injection, ScheduleResult
│   │   ├── round_robin.py
│   │   ├── dim_phased.py
│   │   └── lp_optimal.py
│   ├── simulator/
│   │   ├── engine.py
│   │   └── instrumentation.py
│   ├── lp/
│   │   ├── ilp.py
│   │   └── relaxation.py
│   ├── viz/
│   │   ├── load_histogram.py
│   │   ├── gantt.py
│   │   └── heatmap.py
│   └── cli.py
├── experiments/                      # YAML per experiment
├── eval/run_all.sh
├── results/                          # gitignored except headers
├── fixtures/                         # committed routing tables, expected loads
├── tests/
│   ├── test_topology.py              # neighbor() matches reference; BFS == diameter
│   ├── test_router.py                # DOR is shortest; deterministic
│   ├── test_model.py                 # load aggregation, LB = max load
│   ├── test_schedules.py             # makespan >= LB for every schedule
│   ├── test_simulator.py             # equivalence with LP-extracted schedule
│   └── test_lp.py                    # tiny instance solved to known optimum
└── docs/
    ├── algorithm.md                  # cost model, LB proof, twist semantics
    ├── topology.md                   # convention, routing table walked through
    ├── schedules.md                  # each schedule's recipe and pseudocode
    ├── lp_formulation.md             # ILP details + complexity
    ├── evaluation.md                 # experiment matrix and reproduction
    ├── results.md                    # latest headline numbers (regenerated by eval/run_all.sh)
    └── superpowers/specs/2026-05-12-twisted-torus-alltoall-design.md
```

## 9. Defaults, open questions, scope

### Decided defaults

- Python ≥ 3.11, project-local `uv` venv (`uv venv && uv pip install -e .`).
- Solver: PuLP + CBC; Gurobi optional.
- Default `m = 1`; ablation sweeps `{1, 4, 16}`.
- DOR dim order: largest-first by default; smallest-first as an ablation.
- 3D: both small dims twist into the longest dim (natural reference-code behavior).
- RNG-free; deterministic by construction.

### Parked until we have data

1. **Twist orientation symmetry.** The reference rule shifts by `+slice[dim]` for *both* forward and backward wraps. We will write a unit test asserting that wrapping forward and wrapping backward from the same node land in the same other-dim column (`+R mod 2R == -R mod 2R` because `2R mod 2R = 0`), and document the consequence in `docs/topology.md`.
2. **Bisection-bandwidth check.** Independently compute the bisection-bandwidth lower bound via min-cut over the directed graph; report `max(LB, BW_bound)` as the analytical floor.
3. **Whether `LB` is achievable** on each topology — this is the project's outcome, not a design input.

### Out of scope for v1

- Adaptive routing (the point is fixed routing).
- Non-uniform message sizes.
- Wormhole / cut-through (BW-bound only).
- Real hardware validation.
- Topologies outside the `{S, 2S}` family.
- Latency-bound regime (small `m`, alpha-dominated).

### Risks and mitigations

- **ILP intractability on `4x8`.** Mitigation: LP relaxation + symmetry reduction (fix one node's start step; orbit-based symmetry breaking) + binary search on `T`.
- **Simulator bugs masquerading as gap.** Mitigation: LP-extracted-schedule replay test (§6).
- **DOR-with-twist subtle bugs.** Mitigation: assert DOR hop count equals BFS distance for every `(src, dst)` on every topology.

### Testing discipline

- TDD for `topology/`, `model/`, `simulator/` (small, well-defined contracts).
- LP layer validated by oracle checks: `M_LP ≥ LB`; LP-schedule replay yields `makespan == M_opt`.

## 10. Glossary

- **DOR** — dimension-order routing. Resolves displacement in a fixed dim order.
- **Flow** — a single `(src, dst, size)` traffic demand.
- **`LB`** — bandwidth lower bound = max directed-link load under fixed routing.
- **`M_S`** — makespan achieved by schedule `S`.
- **`M_opt` / `M_LP`** — ILP-optimal / LP-relaxation makespan.
- **`gap(S)` / `gap_routing`** — `M_S / LB` and `M_opt / LB` respectively.
- **Twist** — the half-shift `t = C/2` applied to the smaller dim's wraparound.
