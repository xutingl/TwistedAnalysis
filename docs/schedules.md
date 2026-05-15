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

**Observed gap.** On 2×4, `makespan = 13` vs `LB = 3`, a 4.33× gap. On 4×4×8,
`makespan = 637` vs `LB = 74`, a 8.61× gap. The gap grows with topology size.
Analysis suggests this is a scheduling inefficiency (phase-makespan accumulation)
rather than a routing problem; see [results.md](results.md).

## Schedule A2: XLA (Destination-Core Randomization)

**Class:** `XLASchedule` in `twisted_analysis/schedules/xla.py`

**Phase structure.** Replicates XLA's native destination-core randomization for
AllToAll lowering. Two static prime constants `A = 33617`, `B = 1299721`. For
phase `p ∈ {0, ..., N-2}`:

```
permute_idx = ((p * A) + B) mod (N - 1) + 1
dst         = (src + permute_idx) mod N         for every src
```

Each phase, like round-robin, has every node act as a source exactly once and a
destination exactly once — guaranteeing zero endpoint contention within a phase.
The pseudo-random permutation order is intended (in real hardware) to mitigate
persistent hot-spot patterns across phases.

**Bijection condition.** For `permute_idx` to visit every value in `{1, ..., N-1}`
exactly once across the `N-1` phases, we need `gcd(A, N-1) = 1`. For our N values:

| N | N-1 | gcd(33617, N-1) | Bijection? |
|---|----:|----:|---|
| 8 | 7 | 1 | ✓ |
| 32 | 31 | 1 | ✓ |
| 128 | 127 | 1 | ✓ |

So the set of `(src, dst)` flows XLA emits is exactly a permutation of round-robin's.

**Equivalence with RoundRobin in our cost model.**
Because back-to-back phases each take a makespan determined only by their
displacement (the set of flows in that phase), and total makespan is the sum,
phase order is invariant: `Σ_δ makespan_phase(δ) = makespan_phase(π(δ))` for any
permutation `π`. Empirically, XLA's makespan equals RoundRobin's in every cell
(2x4/4x8/4x4x8 × {ilp, dor}); see [results.md](results.md).

XLA's randomization only matters when the cost model breaks one of these
assumptions: phase overlap (pipelined injection), real-hardware effects
(cache/queue persistence), or background traffic. None of these are modeled here.

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

## Schedule B': OrbitGreedy (constructive, no ILP — headline scheduler)

**Module:** `twisted_analysis/schedules/orbit_greedy.py` (`OrbitGreedySchedule`)

**Motivation.** The symmetric ILP (Schedule C below) solves a time-indexed
integer program over edge-orbit × time slots — 14s for 4×8, intractable for
4×4×8 without a warm start. Inspecting its output
([scripts/inspect_symmetric_schedule.py](../scripts/inspect_symmetric_schedule.py))
showed the schedule has no closed-form shift pattern (only 5/31 orbits on 4×8
are pipelined; hop-gaps range 1–18). The ILP is doing real combinatorial
packing. OrbitGreedy is a **polynomial-time constructive replacement** that
achieves the same optimum in microseconds.

**Algorithm.** Process translation orbits in some order (default:
**`lpt_tail_asc` = longest-path-first, tiebreak by ascending tail-edge load**).
For each orbit, schedule its hops in path order at the earliest time strictly
after the previous hop's time, skipping `(dim, dir, t)` slots already claimed
by another orbit. **Path-internal gaps are allowed** when an edge orbit is
contended — this is the key flexibility OrbitGreedy uses.

```python
edge_load = Counter((dim, dir) -> hits across all orbits' canonical paths)
def tail_load(o): return edge_load[(path[o][-1][2], path[o][-1][3])]
edge_busy: dict[(dim, dir), set[int]] = defaultdict(set)

# lpt_tail_asc: longest path first; ties broken by tail-edge load ascending.
for orbit_id in sorted(orbits, key=lambda o: (-len(path[o]), tail_load(o), o)):
    prev_t = -1
    for i, (_, _, dim, dir) in enumerate(path[orbit_id]):
        t = prev_t + 1
        while t in edge_busy[(dim, dir)]:
            t += 1   # <-- this loop may skip slots: path-internal gaps allowed
        assignment[(orbit_id, i, t)] = 1.0
        edge_busy[(dim, dir)].add(t)
        prev_t = t
```

The output `(orbit_id, hop_i, t)` is fed through `symmetric_assignment_to_injections`
— the same adapter the symmetric ILP uses — so all `N` translations of each
orbit become per-unit `Injection`s with `hop_schedule` populated.

**Why the tail-load tiebreak matters.** Plain LPT scheduled two length-2
orbits on 2×4×4 DOR — both ending on the low-load (0,−1) edge — back-to-back
on the bottleneck at t=14. Their tail hops then both wanted slot t=15 on
(0,−), forcing one to t=16 and pushing makespan to 17 vs LB=16. With the
tail-load tiebreak, the orbits with low-load tails get scheduled FIRST
(while bottleneck slots are still abundant), claiming early bottleneck slots
that leave their corresponding tail-edge slots open. Orbits with flexible
high-load tails fill in around them.

**Performance — optimal on every cell.** Empirically (see
[results.md](results.md)):

| Topology + Router | LB | default (lpt_tail_asc) | lpt | spt |
|---|---:|---:|---:|---:|
| 2×4 ilp   | 3  | **3 (1.00)** | 3 (1.00) | 4 (1.33) |
| 2×4×4 dor | 16 | **16 (1.00)** | 17 (1.06) | — |
| 4×8 ilp   | 21 | **21 (1.00)** | 21 (1.00) | 25 (1.19) |
| 4×4×8 ilp | 74 | **74 (1.00)** | 74 (1.00) | 86 (1.16) |

**OrbitGreedy with the default `lpt_tail_asc` ordering achieves
`makespan = LB` on every (topology, router) cell tested — 10/10.** Plain
`lpt` (no tiebreak) hits LB on 9/10, missing only the 2×4×4 DOR cell.
Runtime is dominated by orbit/router setup (~21s on 4×4×8 for `ILPRouter`'s
LP); the scheduling step itself takes microseconds.

> **⚠ Update (2026-05-15):** The "10/10 LB-tight" claim above is in the
> **orbit-class capacity model**. In the **physical-edge capacity model**
> (what the Pallas kernel executes against), `orbit_greedy` is LB-tight
> on (2,4), (2,2,4), all DOR cells, and (2,4) ILP; but it is LB+1 on
> (2,4,4)-ilp and (4,8)-ilp (verified against `ilp_literal` at LB).
> The two models diverge on routings where `(dim, dir)` translation-
> equivariance fails — including ILPRouter on multiple cells and all
> "loaded" routings (e.g. TPU OCS routes). See
> [orbit_greedy_optimality.md §6](orbit_greedy_optimality.md#6-open-questions)
> "Update (2026-05-15, evening)" and the
> [pallas_kernel scheduler-choice matrix](../pallas_kernel/README.md#routing--scheduler-performance-matrix-physical-edge-model)
> for the corrected numbers.

**Theoretical context.** This is packet routing with given paths
(Leighton-Maggs-Rao 1988/1994): for any congestion-`c`, dilation-`d` instance,
`O(c+d)` makespan is constructive. In our case `d ≤ diameter ≈ ⌈sum(slice)/2⌉`
is small relative to `c = LB`:

| Topology | c=LB | d=diameter | LMR bound c+d | empirical |
|---|---:|---:|---:|---:|
| 2×4 ilp   | 3  | 2 | 5  | 3  |
| 4×8 ilp   | 21 | 4 | 25 | 21 |
| 4×4×8 ilp | 74 | 6 | 80 | 74 |

We hit `c` not `c+d` because the bipartite-style edge-orbit structure admits
a König + Smith's-deadline-feasibility proof on every tested cell — see
[orbit_greedy_optimality.md](orbit_greedy_optimality.md) §4.3. The proof is
machine-verified per cell by `scripts/verify_smith_proof.py`. A closed-form
extension to all `{S, 2S}^n` shapes reduces to standard canonical-path
enumeration.

**Caveat — workload symmetry.** OrbitGreedy relies on `compute_orbits`,
which assumes translation symmetry of the workload (uniform AllToAll). It
does not handle skewed traffic patterns. The ILP-optimal schedules (C, D) make
no such assumption.

See [orbit_greedy_optimality.md](orbit_greedy_optimality.md) for the
full proof: König-style bipartite-multi-graph setup, Smith's deadline
reduction, worked 2×4 ILP proof (§4.3.14), and the 2×4×4 DOR failure-mode
analysis.

## Schedule B'': PipelinedOrbit (constrained variant — **not optimal**)

**Module:** `twisted_analysis/schedules/orbit_greedy.py` (`PipelinedOrbitSchedule`)

**Relationship to OrbitGreedy.** PipelinedOrbit is OrbitGreedy **with one
extra constraint**: every orbit must fire its hops at *consecutive* time
steps — `t_i = start + i` for `i = 0, …, L − 1`. No path-internal gaps are
allowed. The orbit ordering, the use of `symmetric_assignment_to_injections`,
and the `compute_orbits` dependency are all identical to OrbitGreedy; only
the per-orbit hop-placement rule differs.

```python
# PipelinedOrbit: find the smallest `start` such that the *entire* contiguous
# window (e_0, start), (e_1, start+1), ..., (e_{L-1}, start+L-1) is free.
for orbit_id in sorted(orbits, key=...):  # same ordering as OrbitGreedy
    path = canon[orbit_id]
    start = 0
    while True:
        if all((start + i) not in edge_busy[(d, dr)]
               for i, (_, _, d, dr) in enumerate(path)):
            break
        start += 1
    # Commit the entire pipelined window
    for i, (_, _, d, dr) in enumerate(path):
        assignment[(orbit_id, i, start + i)] = 1.0
        edge_busy[(d, dr)].add(start + i)
```

**Why PipelinedOrbit is strictly weaker than OrbitGreedy.** Any schedule
PipelinedOrbit produces is also a valid OrbitGreedy output (it happens to
have all gaps = 1). So PipelinedOrbit's solution space is a *strict subset*
of OrbitGreedy's. When the optimal `makespan = LB` schedule requires some
orbit to *wait* between its hops (e.g., because an intermediate edge is
contended), PipelinedOrbit cannot produce it and must shift the orbit's
entire window later — sometimes past `LB`.

**Performance — sub-optimal on 3 of 10 cells.** PipelinedOrbit
(`lpt_tail_asc` ordering) matches LB on 7 cells but misses on 3:

| Topology + Router | LB | OrbitGreedy | PipelinedOrbit | Gap |
|---|---:|---:|---:|---:|
| 2×4 ilp/dor       | 3, 4   | LB | LB | 0 |
| 2×2×4 ilp/dor     | 5, 7   | LB | LB | 0 |
| 2×4×4 dor         | 16     | LB | LB | 0 |
| 4×8 dor           | 26     | LB | LB | 0 |
| **2×4×4 ilp**     | 11     | 11 | **12** | **+1 (1.09×)** |
| **4×8 ilp**       | 21     | 21 | **22** | **+1 (1.05×)** |
| **4×4×8 dor**     | 86     | 86 | **90** | **+4 (1.05×)** |
| **4×4×8 ilp**     | 74     | 74 | **79** | **+5 (1.07×)** |

PipelinedOrbit is therefore **not the headline scheduler**. It is included
as:

1. **A diagnostic** that answers the structural question "does this topology
   admit a fully-pipelined LB-optimal schedule?" — sometimes yes (2×4,
   2×2×4, 2×4×4 dor, 4×8 dor), sometimes no (the four cells above).
2. **A simpler baseline** (no per-hop slot search; just find a contiguous
   window). The simpler structure also makes it easier to analyze in
   restricted settings.

**Use OrbitGreedy as the production scheduler.** Use PipelinedOrbit only
when the pipelined structure is independently desirable (e.g., for
implementations that must inject orbit-hops back-to-back due to hardware
constraints not modeled here).

## Schedule C: ILP-Optimal (Symmetric)

**Module:** `twisted_analysis/schedules/lp_symmetric.py`

**Construction.** The symmetric ILP (see [lp_formulation.md](lp_formulation.md))
produces an assignment `y[orbit, hop_idx, step]` giving the step at which each
orbit traverses each hop. `symmetric_assignment_to_injections` expands orbit
assignments back to per-unit `Injection` records by applying the `N` translations:

```python
def symmetric_assignment_to_injections(flows, router, orbit_assignment) -> list[Injection]:
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
is exactly achieved. Solve time ~14 s (CBC). On 4×4×8 with ILP routing,
`makespan = 74 = LB` is achievable with the symmetric ILP, but only when
`T_upper` is pinned to `LB` (set `ilp_T_upper_multiplier: 1` in the YAML).
Default `T_upper = 4·LB` causes a model with ~113k binary variables that PuLP
struggles to build; the tight `T_upper = LB` formulation has ~32k variables
and CBC solves it in ~6 minutes — a single feasibility check, no binary
search. In practice, use OrbitGreedy LPT (Schedule B') first to get the
witness, then optionally verify with the symmetric ILP.

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
