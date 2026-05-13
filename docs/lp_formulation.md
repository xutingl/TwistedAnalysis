# LP / ILP Formulation

The ILP and LP relaxation live in `twisted_analysis/lp/`. The solver is PuLP with
CBC by default; Gurobi can be passed as `solver_name="GUROBI_CMD"`.

## Variable Definition

A flow of size `m` is unrolled into `m` independent **units**, each sharing the
same path. This matches the simulator semantics exactly.

For each unit `u`, hop index `i ∈ [0, len(path(u)))`, and time step `t ∈ [0, T)`:

```
x[u, i, t] ∈ {0, 1}
```

Interpretation: unit `u` traverses `path(u)[i]` (i.e., uses directed link `e_i` of
its path) at time step `t`.

The horizon `T` is fixed when solving feasibility; binary search over `T` finds the
minimum feasible horizon.

## Constraints

**1. Per-hop fire-once.** Each unit traverses each hop link exactly once:

```
∀ u, i:   Σ_{t=0}^{T-1} x[u, i, t] == 1
```

**2. Causal order (store-and-forward).** Unit `u` cannot use hop `i+1` at step `s`
unless it used hop `i` at some step `t < s`:

```
∀ u, i, s:   Σ_{t=0}^{s} x[u, i+1, t]  ≤  Σ_{t=0}^{s-1} x[u, i, t]
```

Note the strict inequality in the subscript range: if hop `i` fires at step `s-1`,
hop `i+1` may fire at step `s` (next step only).

**3. Link capacity.** At most one unit crosses each directed link per step:

```
∀ e, t:   Σ_{(u,i) : path(u)[i] = e} x[u, i, t]  ≤  1
```

## Objective: Binary Search on T

The preferred approach fixes `T` and solves feasibility (objective = 0):

```python
def solve_makespan(topology, router, flows, T_upper, solver_name="PULP_CBC_CMD"):
    # Compute LB = max link load
    lo, hi = lb, T_upper
    while lo < hi:
        mid = (lo + hi) // 2
        assignment = _solve_feasibility(units, mid, solver)
        if assignment is not None:
            hi = mid
        else:
            lo = mid + 1
    return lo, best_assignment
```

The upper bound `T_upper` is doubled automatically if the initial bound is
infeasible. Binary search terminates in `O(log T)` ILP solves.

Invoke as:

```python
from twisted_analysis.lp.ilp import solve_makespan
from twisted_analysis.topology import Topology, ILPRouter
from twisted_analysis.model import AllToAll

t = Topology(slice=(2, 4))
r = ILPRouter(t)
w = AllToAll(t, r, 1)
T_opt, assignment = solve_makespan(t, r, list(w.flows), T_upper=w.lower_bound * 4)
print(T_opt)  # 4 for (2,4) with m=1
```

## LP Relaxation

**File:** `twisted_analysis/lp/relaxation.py`

The LP relaxation drops the integrality constraint, replacing `x ∈ {0,1}` with
`x ∈ [0, 1]`. It minimizes the expected completion time:

```
minimize  M
subject to:
    (same constraints as ILP)
    ∀ u:  Σ_t (t+1) * x[u, last_hop, t]  ≤  M
```

**Critical caveat (verbatim from `relaxation.py` docstring).** The LP relaxation
returns the minimum *expected* completion time across units. This is NOT a true
makespan bound. With fractional `x`, the LP can produce values below the link-load
lower bound `LB`. Do **not** assume `M_LP ≥ LB`; the ordering is only
`M_LP ≤ M_opt`. A makespan-tight LP relaxation would require a different (more
expensive) formulation; that is out of scope for v1.

```
LB ≤ M_opt    (always)
M_LP ≤ M_opt  (LP relaxation is a lower bound on ILP)
M_LP vs. LB   (no guaranteed ordering; M_LP can be below LB)
```

Invoke as:

```python
from twisted_analysis.lp.relaxation import lp_relax_lower_bound

M_LP = lp_relax_lower_bound(t, r, list(w.flows), T_upper=w.lower_bound * 4)
```

## Variable Count and Complexity

For the binary-searched horizon `T` and `H` total hops across all units:

```
variables ≈ H × T
```

where `H = Σ_f (m × len(path(f)))`.

| Topology | N | Flows | H (m=1) | Approx variables at T=LB×4 |
|---|---|---|---|---|
| 2×4 | 8 | 56 | ~112 | ~1,800 |
| 4×8 | 32 | 992 | ~3,000 | ~90,000 |
| 4×4×8 | 128 | 16,256 | ~80,000 | ~27,000,000 |

- **2×4**: ILP tractable in seconds.
- **4×8**: ILP "best effort"; solve with symmetry reduction; LP relaxation always
  reported.
- **4×4×8**: LP relaxation only; full ILP is not tractable.

## Symmetric Scheduling ILP (Translational Orbits)

**File:** `twisted_analysis/lp/symmetric.py`

### Motivation

The full ILP scales as `O(H × T)` variables. For 4×8, `H ≈ 3,000` and
`T ≈ 84` under ILP routing, yielding ~252,000 binary variables — marginally tractable
but slow. Translational symmetry reduces this by a factor of `N` (32 for 4×8).

### Orbit Definition

The `N` translations of the network are automorphisms: if we shift every node index
by a constant vector `v` (mod topology shape), the link structure is preserved. Any
feasible schedule `x[u, i, t]` can be translated to give another feasible schedule
of equal makespan. We collect all units whose `(src, dst)` pair belongs to the same
translated orbit into one **orbit class**.

### Variables

For each orbit class `o`, hop index `i`, and time step `t`:

```
y[o, i, t] ∈ {0, 1}
```

Interpretation: every unit in orbit `o` traverses hop `i` of its (translated) path
at step `t`. Within a single orbit, all `N` translated units fire at the same
relative step.

### Constraints

**1. Fire-once.** Each orbit fires each hop exactly once:

```
∀ o, i:   Σ_t y[o, i, t] == 1
```

**2. Causal order.** Same store-and-forward rule as the full ILP, applied per orbit:

```
∀ o, i, s:   Σ_{t=0}^{s} y[o, i+1, t]  ≤  Σ_{t=0}^{s-1} y[o, i, t]
```

**3. Edge-orbit capacity.** At each time step, the directed links visited by orbit
`o` at hop `i` are determined by the canonical unit's path. Because of translational
symmetry, each orbit's hop covers a disjoint set of `N` directed links (one per
translated copy). The capacity constraint on any single directed link is therefore:

```
∀ e, t:   Σ_{(o,i) : canonical_path(o)[i] = e} y[o, i, t]  ≤  1
```

The number of orbits equals the number of distinct `(src, dst)` pairs with `src = 0`
(i.e., `N - 1`), a factor-`N` reduction over the full ILP.

### Symmetry Assumption

The reduction is valid when the topology has full translational symmetry (all `N`
translations are automorphisms). This holds for all topologies in the `{S, 2S}`
family. If ILP routing produces a non-symmetric path assignment (different canonical
paths for different translated copies of the same pair), the symmetric ILP does not
apply; in that case fall back to the full ILP with symmetry-breaking constraints.

### Solve Performance

| Topology | Orbits | Variables at T=LB | Solve time |
|----------|--------|--------------------|------------|
| 2×4      | 7      | ~126               | <1s        |
| 4×8      | 31     | ~2,604             | ~14s       |
| 4×4×8    | 127    | ~47,000            | not attempted |

The 4×8 symmetric ILP solves in ~14 s with CBC and achieves `makespan = 21 = LB`
under ILP routing — confirming the lower bound is tight and the previous gap was an
artifact of DOR's tie-breaking, not an intrinsic limit of the topology.

## See Also

- [algorithm.md](algorithm.md) — the cost model this ILP encodes.
- [schedules.md](schedules.md) — how the LP assignment becomes an executable schedule.
- [results.md](results.md) — ILP results for 2×4 and 4×8; LP relaxation notes for 4×4×8.
- [topology.md](topology.md) — ILPRouter and translational symmetry of the routing LP.
