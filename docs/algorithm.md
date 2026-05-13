# Algorithm: Cost Model and Lower Bound

## Cost Model

Time is discrete integer steps. Every directed link has capacity 1 flow-unit per
step; full-duplex links are treated as independent directed edges. The network uses
**store-and-forward** semantics: a unit of flow `f` may traverse link `e_{i+1}` at
step `s+1` only if it traversed `e_i` at step `s`. At most one flow-unit crosses any
directed link per step. Per-hop latency (alpha) is omitted; message latency equals
path length in steps, dominated by bandwidth for large messages.

The AllToAll workload is `N × (N-1)` flows — each of the `N = prod(slice)` nodes
sends a message of size `m` units to every other node. Default `m = 1`.

## Lower Bound (LB)

**Definition.** Given a fixed routing table mapping every `(src, dst)` pair to a
single path, define:

```
load(e) = m * |{ f : e ∈ path(f) }|
LB      = max_e load(e)
```

**Theorem.** Any feasible schedule has `makespan ≥ LB`.

**Proof.**  
Pick any directed edge `e` and let `L = load(e)`. In any feasible schedule, at most
one flow-unit crosses `e` per step (capacity 1). There are `L` units that must cross
`e` (each of the `L = m × (number of flows whose path includes e)` units must
traverse `e` exactly once). Therefore at least `L` steps must elapse before all
units crossing `e` are delivered, so `makespan ≥ L`. Since this holds for every
edge `e`, taking the maximum gives `makespan ≥ max_e load(e) = LB`. QED.

**Note on tightness.** The lower bound is not necessarily achieved. Whether `LB` is
achievable — i.e., whether a schedule exists with `makespan = LB` — is exactly the
question this project investigates. The ILP (see [lp_formulation.md](lp_formulation.md))
finds the optimal makespan; the ratio `M_opt / LB` measures the intrinsic gap from
routing alone.

## Worked Example: (2,4) with m=1

Topology has `N = 8` nodes; the AllToAll workload has `8 × 7 = 56` flows.

Under dimension-order routing (largest dim first), link loads take values in
`{1, 2, 3, 4}`. The bottleneck edges all have load 4 and are the dim-1 links in the
`+1` direction (the 8 forward links around the two length-4 rings):

| Edge (u → v, dim, dir) | load |
|---|---|
| `(0,0) → (0,1)`, dim=1, dir=+1 | 4 |
| `(0,1) → (0,2)`, dim=1, dir=+1 | 4 |
| `(0,2) → (0,3)`, dim=1, dir=+1 | 4 |
| `(0,3) → (0,0)`, dim=1, dir=+1 | 4 |
| `(1,0) → (1,1)`, dim=1, dir=+1 | 4 |
| … (symmetrically) | 4 |

Representative non-bottleneck loads:

| Edge | load |
|---|---|
| `(0,0) → (1,0)`, dim=0, dir=+1 | 3 |
| `(0,0) → (0,3)`, dim=1, dir=-1 | 2 |
| `(0,0) → (1,2)`, dim=0, dir=-1 | 1 |

So `LB = 4`. The ILP achieves exactly `makespan = 4` (ratio 1.00); see
[results.md](results.md).

To reproduce (with DOR routing, which produces LB = 4 on 2×4 — switch to
`ILPRouter` for LB = 3):

```python
from twisted_analysis.topology import Topology, DORRouter
from twisted_analysis.model import AllToAll

t = Topology(slice=(2, 4))
r = DORRouter(t)
w = AllToAll(t, r, 1)
print(sorted(set(w.link_load.values())))  # [1, 2, 3, 4]
print('LB=', w.lower_bound)               # LB= 4
print('bottleneck edges:', w.bottleneck_edges()[:3])
```

## Routing and Load Attribution

The link load depends entirely on the routing table, not on the schedule. For the
twisted-torus DOR, the smaller-dim wraparound displaces the larger dim (see
[topology.md](topology.md)), so DOR paths in the smaller dim can cross larger-dim
links as part of the twist. This coupling is why some links accumulate higher load
than on a standard torus.

## See Also

- [topology.md](topology.md) — neighbor function, DOR routing table construction.
- [lp_formulation.md](lp_formulation.md) — ILP encoding of the same cost model.
- [results.md](results.md) — measured `LB`, `makespan`, and `ratio` per topology.
