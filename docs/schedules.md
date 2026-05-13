# Schedules

All schedules implement the `Schedule` protocol from `twisted_analysis/schedules/base.py`:

```python
class Schedule(Protocol):
    name: str
    def emit(self, workload: AllToAll) -> list[Injection]: ...
```

An `Injection` is `(flow, start_step, priority, hop_schedule)`. The simulator
consumes injections and produces the realized makespan; schedules are pure
coordination policies and never bypass the cost model. See [algorithm.md](algorithm.md)
for the cost model.

## Schedule A: Round-Robin (Latin Square)

**Class:** `RoundRobinSchedule` in `twisted_analysis/schedules/round_robin.py`

**Phase structure.** There are `N - 1` phases, where `N` is the number of nodes.
In phase `r ∈ {1, ..., N-1}`, each node `i` sends its message to node
`(i + r) mod N` (using the flat index from the topology's iteration order).
All `N` flows in a phase are injected simultaneously at the phase's start step.

**Phase makespan accumulation.** Each phase is dry-run independently to compute
its makespan; the next phase starts when the previous one drains:

```python
phase_start = 0
for r in range(1, N):
    phase_flows = [Flow(nodes[i], nodes[(i + r) % N], m) for i in range(N)]
    for f in phase_flows:
        injections.append(Injection(flow=f, start_step=phase_start))
    sim = Simulator(topology, router, phase_flows)
    for f in phase_flows:
        sim.inject(Injection(flow=f, start_step=0))
    phase_start += sim.run()
```

**Properties.** Latin-square round-robin is a complete, balanced schedule: every
`(src, dst)` pair appears in exactly one phase. Each phase keeps every node busy
(one send per node). However, phase boundaries are conservative: the next phase
does not start until the slowest flow of the current phase drains. Gaps between
phases accumulate, leading to a large total makespan.

**Observed gap.** On 2×4, `makespan = 13` vs `LB = 4`, a 3.25× gap. On 4×4×8,
`makespan = 650` vs `LB = 86`, a 7.56× gap. The gap grows with topology size.
Analysis suggests this is a scheduling inefficiency (phase-makespan accumulation)
rather than a routing problem; see [results.md](results.md).

## Schedule B: Dimension-Ordered Phases (DimPhased)

**Class:** `DimPhasedSchedule` in `twisted_analysis/schedules/dim_phased.py`

**IMPORTANT CAVEAT: Partial coverage.** DimPhased covers only the `(N-1) × ndim /
ndim_total`-fraction of flows where source and destination differ in exactly one
dimension. This is not a full AllToAll. It is provided as a diagnostic to measure
per-dim link efficiency; its makespan and ratio are not comparable to RoundRobin or
ILP on the full workload. The ratio can be below 1.0 because `LB` is computed for
the full AllToAll but only a fraction of those flows are scheduled.

**Phase definition.** For each dimension `d` (resolved in decreasing-size order):
phase `d` schedules all `(src, dst)` pairs where `src` and `dst` differ only in
coordinate `d`. Each phase uses only dim-`d` links; different phases do not contend.

**Pseudocode:**

```python
dim_order = sorted(range(ndim), key=lambda d: -slice[d])  # largest dim first
phase_start = 0
for d in dim_order:
    phase_flows = [
        Flow(src, dst, m)
        for src in nodes
        for dst in nodes
        if src != dst
        and all(src[i] == dst[i] for i in range(ndim) if i != d)
    ]
    for f in phase_flows:
        injections.append(Injection(flow=f, start_step=phase_start))
    sim = Simulator(topology, router, phase_flows)
    for f in phase_flows: sim.inject(Injection(flow=f, start_step=0))
    phase_start += sim.run()
```

**Properties.** Because phases use disjoint link sets, they cannot contend with
each other — the total makespan equals the sum of per-phase makespans. Within a
phase, the flows form independent rings (one ring per slice along that dim), so
the bottleneck is the per-ring AllToAll makespan. The twist only affects phases
in the smaller dims (where the wraparound cross-shifts the larger dim coordinate).

## Schedule C: ILP-Optimal (Symmetric)

**Module:** `twisted_analysis/schedules/lp_optimal_symmetric.py`

**Construction.** The symmetric ILP (see [lp_formulation.md](lp_formulation.md))
produces an assignment `y[orbit, hop_idx, step]` giving the step at which each
orbit traverses each hop. `symmetric_lp_assignment_to_injections` expands orbit
assignments back to per-unit `Injection` records by applying the `N` translations:

```python
def symmetric_lp_assignment_to_injections(flows, router, orbit_assignment) -> list[Injection]:
    # For each orbit o and each translated copy v:
    #   src = (0 + v) mod topology
    #   dst = (canonical_dst + v) mod topology
    #   start_step  = orbit_assignment fire step at hop 0
    #   hop_schedule = tuple of fire steps for every hop
    ...
```

**Scope.** Only applicable when the topology has full translational symmetry and
ILPRouter is used for routing. Falls back to the full `ILPOptimal` schedule otherwise.

**Validation.** Same as Schedule D (below): the realized simulator makespan is
asserted to equal `M_opt` for every instance.

**Performance.** On 4×8 with ILP routing, `makespan = 21 = LB` — the lower bound
is exactly achieved. Solve time ~14 s (CBC).

## Schedule D: LP-Optimal

**Module:** `twisted_analysis/schedules/lp_optimal.py`

**Construction.** The ILP (see [lp_formulation.md](lp_formulation.md)) produces
an assignment `x[unit, hop_idx, step]` giving the step at which each unit traverses
each hop. `lp_assignment_to_injections` translates this into `Injection` records:

```python
def lp_assignment_to_injections(flows, router, assignment) -> list[Injection]:
    # For each unit:
    #   start_step  = LP's fire step at hop 0
    #   hop_schedule = tuple of LP fire steps for every hop (hop 0, 1, ...)
    ...
```

Storing the full per-hop schedule in `Injection.hop_schedule` allows the simulator
to use the LP-intended ordering at every intermediate link (not just at injection
time). This matters when the LP spaces out hops non-pipelinedly: a unit that waits
at an intermediate link because the LP scheduled it later must not displace a unit
the LP intended to fire first.

**Link priority.** The simulator's `_step` method selects one unit per link per
step using `(effective_priority(), seq)` as the ordering key.
`effective_priority()` returns `hop_schedule[next_hop_idx]` when available,
otherwise falls back to the static `priority` field.

**Validation.** For every ILP-solved instance, the LP-extracted schedule is fed
back to the simulator and the realized makespan is asserted to equal `M_opt`.

## ScheduleResult

All schedules produce a `ScheduleResult`:

```python
@dataclass(frozen=True)
class ScheduleResult:
    name: str
    makespan: int
    lower_bound: int
    per_step_busy: tuple[int, ...]           # link utilization per step
    idle_steps_on_bottleneck: dict[tuple, int]  # idle count per bottleneck edge

    @property
    def ratio(self) -> float:
        return self.makespan / self.lower_bound
```

## See Also

- [algorithm.md](algorithm.md) — cost model and lower bound definition.
- [lp_formulation.md](lp_formulation.md) — ILP that drives Schedule C.
- [results.md](results.md) — measured makespans and ratios.
