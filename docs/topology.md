# Topology: Twisted-Torus Neighbor Function and DOR Routing

## Neighbor Function

The canonical neighbor function (verbatim from `twisted_analysis/topology/lattice.py`):

```python
def neighbor(self, node: Node, dim: int, dir: int) -> Node:
    assert len(node) == self.ndim
    assert 0 <= dim < self.ndim
    assert dir in (-1, 1)
    new = list(node)
    new[dim] += dir
    wrapped = new[dim] < 0 or new[dim] >= self.slice[dim]
    if wrapped:
        shift = self.slice[dim]
        new = [(new[i] + shift) % self.slice[i] for i in range(self.ndim)]
    return tuple(new)
```

**Key rule.** When a step in dimension `d` crosses the boundary (coordinate goes
below 0 or reaches `slice[d]`), *every* coordinate is shifted by `slice[d]` modulo
its own size:

```
new[i] = (new[i] + slice[d]) % slice[i]   for all i
```

This applies to both forward (`dir=+1`) and backward (`dir=-1`) wraps.

### Shape constraint

All topologies must satisfy `∀i. slice[i] ∈ {S, 2S}` where `S = min(slice)`.
`Topology.__post_init__` enforces this.

## Wrap Semantics on R × 2R Topologies

For `slice = (R, 2R)` (dim 0 has size `R`, dim 1 has size `2R`):

| Wrap dim | shift applied | Effect on other dim |
|---|---|---|
| dim 0 (size `R`) | shift = R | dim-1 coord shifts by `R mod 2R = R` → **half-shift twist** |
| dim 1 (size `2R`) | shift = 2R | dim-0 coord shifts by `2R mod R = 0` → plain torus wrap |

Wrapping in the smaller dimension *shifts* the larger-dim coordinate by R = half of
its extent. This is the defining "twist". Wrapping in the larger dimension has no
cross-dim effect.

## Worked Traces: (2, 4)

`slice=(2,4)`, S=2, so dim 0 has size 2 (smaller), dim 1 has size 4 (larger).

**No-wrap steps (within-boundary):**

```
neighbor((0,0), dim=0, dir=+1) = (1, 0)   # 0+1=1 < 2, no wrap
neighbor((0,0), dim=1, dir=+1) = (0, 1)   # 0+1=1 < 4, no wrap
neighbor((1,0), dim=1, dir=+1) = (1, 1)   # in-plane
```

**Dim-0 wrap (twist):** stepping from row 1 in the +1 direction wraps back to row 0
and shifts the dim-1 coordinate by `slice[0]=2`:

```
neighbor((1, 0), dim=0, dir=+1):
    new = [2, 0] -> wrapped (2 >= 2)
    shift = 2
    new = [(2+2)%2, (0+2)%4] = [0, 2]
    result = (0, 2)

neighbor((1, 2), dim=0, dir=+1):
    new = [2, 2] -> wrapped
    new = [(2+2)%2, (2+2)%4] = [0, 0]
    result = (0, 0)
```

**Dim-0 backward wrap (also twist):** stepping from row 0 in the -1 direction:

```
neighbor((0, 0), dim=0, dir=-1):
    new = [-1, 0] -> wrapped (-1 < 0)
    shift = 2
    new = [(-1+2)%2, (0+2)%4] = [1, 2]
    result = (1, 2)
```

**Dim-1 wrap (plain torus):** stepping past col 3 or before col 0 does not shift
the row coordinate:

```
neighbor((0, 3), dim=1, dir=+1):
    new = [0, 4] -> wrapped (4 >= 4)
    shift = 4
    new = [(0+4)%2, (4+4)%4] = [0, 0]
    result = (0, 0)

neighbor((1, 3), dim=1, dir=+1):
    new = [1, 4] -> wrapped
    new = [(1+4)%2, (4+4)%4] = [1, 0]
    result = (1, 0)
```

The dim-1 rings are independent plain tori; no twist.

## ASCII Diagram: (2, 4) Links

Nodes arranged as 2 rows × 4 columns. Horizontal edges are dim-1 (within-row);
vertical edges are dim-0 (cross-row). Wrap edges are marked with arrows showing
where they land.

```
Row 0:  (0,0) -- (0,1) -- (0,2) -- (0,3) -[wrap]-> (0,0)  [plain torus]
          |                                    |
       [dim-0 links; wrap shifts col by 2]     |
          |                                    |
Row 1:  (1,0) -- (1,1) -- (1,2) -- (1,3) -[wrap]-> (1,0)  [plain torus]

Dim-0 wrap links (shown for +1 direction; carry the half-shift twist):
  (1,0) --+1--> (0,2)     (1,2) --+1--> (0,0)
  (1,1) --+1--> (0,3)     (1,3) --+1--> (0,1)
```

## Twist Orientation Symmetry

**Property.** For the smaller-dim wrap, both the forward (`dir=+1`) and backward
(`dir=-1`) wraps apply the same shift magnitude `+slice[d]`. Because
`+slice[d] mod 2*slice[d] == -slice[d] mod 2*slice[d]` (both equal `slice[d]`),
wrapping forward and wrapping backward from a node land in the same column of the
longer dim. Formally:

```
(x + slice[d]) % (2*slice[d])  ==  (x - slice[d]) % (2*slice[d])  ==  x XOR slice[d]
```

for `x ∈ [0, 2*slice[d])`. The implementation uses `+shift` unconditionally, so
both wrap directions are handled by the same arithmetic.

Consequence for routing: the DOR router cannot choose a different "twist column"
by going backward versus forward across the smaller dim; both arrive at the same
cross-dim offset. The router enumerates all candidate displacement vectors that
actually reach `dst` when walked in dim order.

## Worked Traces: (4, 4, 8)

`slice=(4,4,8)`, S=4. Dims 0 and 1 have size 4 (smaller); dim 2 has size 8 (larger).

Wrapping in dim 0 shifts all coords by 4: dim-1 shifts by `4 mod 4 = 0`; dim-2
shifts by `4 mod 8 = 4`. So a dim-0 wrap applies a **half-shift twist** to the
size-8 dimension.

Wrapping in dim 1 shifts all coords by 4: dim-0 shifts by `4 mod 4 = 0`; dim-2
shifts by `4 mod 8 = 4`. Same twist into dim 2.

Wrapping in dim 2 shifts all coords by 8: dim-0 shifts by `8 mod 4 = 0`; dim-1
shifts by `8 mod 4 = 0`. Plain torus wrap.

Example dim-0 wrap:

```
neighbor((3, 1, 5), dim=0, dir=+1):
    new = [4, 1, 5] -> wrapped (4 >= 4)
    shift = 4
    new = [(4+4)%4, (1+4)%4, (5+4)%8] = [0, 1, 1]
    result = (0, 1, 1)
```

## DOR Routing Table

The router resolves displacements in decreasing dim-size order (largest dim first).
For each `(src, dst)` pair it:

1. Enumerates candidate step vectors `(δ_0, δ_1, ...)` — one step per required
   coordinate offset — that reach `dst` when walked in dim order on the topology.
2. Picks the minimum hop-count candidate; tie-breaks prefer no-wrap, then `+dir`.
3. Walks the chosen displacement in fixed dim order, producing a path of directed
   links.

Because the twist couples dimensions, a step in dim 0 can change the dim-1 and
dim-2 coordinates at the wrap boundary. The router accounts for this by simulating
the walk rather than computing the displacement algebraically.

**Validation.** `len(DOR.path(s, d)) == BFS.dist(s, d)` for every `(s, d)` on every
topology (asserted in `tests/test_router.py`).

## ILPRouter (Load-Balanced Minimal Routing)

**Class:** `ILPRouter` in `twisted_analysis/topology/ilp_router.py`

### Motivation

DOR tie-breaking is lexicographic and can concentrate traffic on a subset of minimal
paths, raising the max-link load `LB`. The ILPRouter solves an LP to distribute each
`(src, dst)` flow across its set of minimal paths so that the max directed-link load
is minimized. This is the standard load-balanced minimal routing approach (ported from
the btowles framework).

### Formulation

Let `P(s,d)` be the set of hop-minimal paths from `s` to `d`. For each path
`p ∈ P(s,d)` introduce a variable `λ[s,d,p] ≥ 0` with `Σ_p λ[s,d,p] = 1`. The
LP minimizes the max directed-link load:

```
minimize  L
subject to:
    Σ_{p ∋ e} λ[s,d,p]  ≤  L     ∀ directed link e (summed over all (s,d) flows)
    Σ_p λ[s,d,p]         = 1      ∀ (s, d)
    λ[s,d,p]             ≥ 0
```

The resulting `LB = L*` is the routing lower bound under load-balanced minimal routing.

### Translational Symmetry Reduction

On an `N`-node topology, the N translations `(s,d) → (s+v, d+v)` (mod topology) are
automorphisms. All flows from a given canonical origin `s=0` cover the full orbit via
the symmetry, so the LP only needs variables for the `N-1` pairs with `src = 0`
(canonical origin). The resulting `λ` values are then replicated to all `N` translated
copies. This reduces the LP variable count by a factor of `N`.

### Impact on Lower Bound

ILP routing reduces `LB` relative to DOR on every topology:

| Topology | DOR LB | ILP LB | Reduction |
|----------|--------|--------|-----------|
| 2×4      | 4      | 3      | 25%       |
| 4×8      | 26     | 21     | 19%       |
| 4×4×8    | 86     | 74     | 14%       |

The reduction is most pronounced on smaller topologies where DOR's lexicographic
tie-breaking has fewer paths to choose from. On 2×4, ILP routing makes the `LB`
achievable by the symmetric ILP scheduler (makespan = LB = 3; see
[lp_formulation.md](lp_formulation.md)).

## See Also

- [algorithm.md](algorithm.md) — how link load is computed from paths.
- [schedules.md](schedules.md) — how the routing table determines phase structure.
- [lp_formulation.md](lp_formulation.md) — symmetric scheduling ILP formulation.
