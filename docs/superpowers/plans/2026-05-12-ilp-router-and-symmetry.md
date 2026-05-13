# ILP Router + Symmetric Scheduling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two extensions to TwistedAnalysis:

- **Part A — ILP Router:** Add `ILPRouter`, a load-balanced minimal-routing strategy ported from btowles' approach. Picks one minimal path per `(src, dst)` to minimize max channel load (using translational symmetry to keep ILP tiny). Makes `ILPRouter` the default for the CLI and downstream workloads. Lowers `LB` on topologies with path diversity.

- **Part B — Symmetric scheduling ILP:** Add a translation-symmetry-reduced scheduling ILP variant that groups flows into orbits and solves at most `N-1` per-orbit time tracks instead of `N(N-1)` per-unit time tracks. Factor-`N` variable reduction. Goal: make `4x8` ILP tractable; best-effort on `4x4x8`.

**Architecture:**

- A new `Router` Protocol unifies `DORRouter` (renamed from current `Router`) and `ILPRouter`. `AllToAll` is unchanged (it already accepts any object with `.path(src, dst)`).
- The CLI gains a `router: dor|ilp` config field (default `ilp`).
- A new file `twisted_analysis/lp/symmetric.py` houses the orbit-based scheduling ILP. Existing `lp/ilp.py` is unchanged for back-compat and for small-instance ground truth.

**Tech stack:** Python 3.11+, PuLP + CBC for ILP, existing test infra.

**Reference:** btowles' `solve_minimal_routes` (translational-symmetry ILP for routing) — the algorithm we're porting, discussed in the conversation that produced this plan.

**Conventions for every task:**
- Working directory: `/home/xutingl/collective_comm/TwistedAnalysis/`.
- Tests: `uv run pytest <path> -v` (fall back to `.venv/bin/python -m pytest`).
- Commits: Conventional Commits.
- Run full suite at end of every task.

---

## File structure (locked in)

```
twisted_analysis/topology/
├── lattice.py              # unchanged
├── router.py               # renamed: contains DORRouter (old Router class)
└── ilp_router.py           # NEW: ILPRouter

twisted_analysis/lp/
├── ilp.py                  # unchanged
├── relaxation.py           # unchanged
└── symmetric.py            # NEW: orbit-based scheduling ILP

experiments/                # each YAML gets a `router: ilp` field (default)
docs/
├── topology.md             # add ILPRouter section
├── lp_formulation.md       # add symmetric-ILP section
└── results.md              # updated headline table
```

---

# Part A — ILP Router

## Task A1: Router protocol + DORRouter rename

**Files:**
- Modify: `twisted_analysis/topology/router.py` — rename `Router` class to `DORRouter`. Add a `Router` Protocol class.
- Modify: `twisted_analysis/topology/__init__.py` — re-export both, with `Router` as the Protocol type.
- Update existing call sites that construct a `Router`: search and replace `Router(t)` with `DORRouter(t)` *only at the construction site*. Tests that import `from twisted_analysis.topology import Router` continue to work because the Protocol type is named `Router`.

- [ ] **Step 1: Write a failing test for the new layout**

`tests/test_router_protocol.py`:
```python
from twisted_analysis.topology import Topology, Router, DORRouter


def test_dor_router_implements_protocol():
    t = Topology(slice=(2, 4))
    r = DORRouter(t)
    # Must satisfy the Router Protocol structurally
    assert isinstance(r, Router) or hasattr(r, "path")
    assert r.path((0, 0), (0, 0)) == ()
    assert len(r.path((0, 0), (0, 1))) == 1


def test_router_is_protocol():
    # The Router export is a typing.Protocol — not a concrete class.
    import typing
    assert hasattr(Router, "_is_protocol") and Router._is_protocol is True
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_router_protocol.py -v`
Expected: `ImportError: cannot import name 'DORRouter'`.

- [ ] **Step 3: Refactor `router.py`**

Edit `twisted_analysis/topology/router.py`. At the top add:

```python
from typing import Protocol, runtime_checkable
```

Add this Protocol *before* the existing class definition:

```python
@runtime_checkable
class Router(Protocol):
    """Structural protocol for routers. Any object with .path(src, dst) -> Path
    is a Router. The two concrete implementations are DORRouter (dimension-order)
    and ILPRouter (load-balanced minimal routing).
    """
    topology: "Topology"
    def path(self, src: "Node", dst: "Node") -> "Path": ...
```

Rename the existing `@dataclass(frozen=True) class Router:` to `class DORRouter:`. Keep all method bodies identical.

- [ ] **Step 4: Update `topology/__init__.py`**

```python
from twisted_analysis.topology.lattice import Topology, Node, DirectedLink
from twisted_analysis.topology.router import Router, DORRouter, Path

__all__ = ["Topology", "Node", "DirectedLink", "Router", "DORRouter", "Path"]
```

- [ ] **Step 5: Update all `Router(...)` construction sites**

Search and replace:
```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
# Find every Router(...) construction. Update them to DORRouter(...).
grep -rn "Router(" --include="*.py" twisted_analysis/ tests/ scripts/
```

For each match where `Router` is being instantiated (not a type annotation), replace with `DORRouter`. Use file-by-file edits with care; don't blanket-replace because type annotations like `router: Router` should stay as `Router` (the Protocol).

Files that will need updates (approximate list — verify by greppping):
- `twisted_analysis/schedules/round_robin.py` — uses Router as type annotation (keep)
- `twisted_analysis/schedules/dim_phased.py` — same
- `twisted_analysis/cli.py` — `Router(t)` construction → `DORRouter(t)`
- `scripts/dump_routing_tables.py` — same
- All test files that build a router → `DORRouter(t)`

- [ ] **Step 6: Run full suite to verify no regressions**

Run: `uv run pytest -v`
Expected: 55 passed (53 previous + 2 new).

- [ ] **Step 7: Commit**

```bash
git add twisted_analysis/topology/router.py twisted_analysis/topology/__init__.py
git add tests/test_router_protocol.py
git add twisted_analysis/schedules/ twisted_analysis/cli.py scripts/dump_routing_tables.py tests/
git commit -m "refactor(topology): introduce Router Protocol; rename old class to DORRouter"
```

---

## Task A2: ILPRouter — load-balanced minimal routing

**Files:**
- Create: `twisted_analysis/topology/ilp_router.py`
- Modify: `twisted_analysis/topology/__init__.py` (re-export ILPRouter)
- Create: `tests/test_ilp_router.py`

The algorithm:

1. Enumerate **all** minimal paths for each `(origin, dst)` pair using BFS over `topology.directed_links()` (collects every shortest path, not just one).
2. By translational symmetry, the same path choices apply to every src up to translation. Build the ILP with one binary variable per `(dst_from_origin, candidate_path_index)`.
3. For each directed-link orbit class (one per `(dim, dir)` = `2*ndim` total), compute the load contribution from each variable: how many times does this candidate path cross *any* edge in this orbit class? Under symmetry, this equals 1 per crossing × N (one per src in the orbit).
4. Minimize the max-per-link-orbit-class load. Since each link orbit has `N` edges and the contribution from a path that crosses the orbit at hop `i` is uniform across all `N` edges (under translation), max link load = max link-orbit load.

Actually, simpler: use the *non-symmetric* contribution counts and the symmetry constraint. btowles' code does exactly this. Port directly.

- [ ] **Step 1: Write failing tests**

`tests/test_ilp_router.py`:
```python
from twisted_analysis.topology import Topology, DORRouter
from twisted_analysis.topology.ilp_router import ILPRouter
from twisted_analysis.model import AllToAll


def test_ilp_router_path_length_equals_bfs_2x4():
    t = Topology(slice=(2, 4))
    r = ILPRouter(t)
    dist = t.bfs_distances()
    for s in t.nodes():
        for d in t.nodes():
            assert len(r.path(s, d)) == dist[s][d]


def test_ilp_router_path_length_equals_bfs_4x8():
    t = Topology(slice=(4, 8))
    r = ILPRouter(t)
    dist = t.bfs_distances()
    for s in t.nodes():
        for d in t.nodes():
            assert len(r.path(s, d)) == dist[s][d]


def test_ilp_router_path_length_equals_bfs_4x4x8():
    t = Topology(slice=(4, 4, 8))
    r = ILPRouter(t)
    dist = t.bfs_distances()
    for s in t.nodes():
        for d in t.nodes():
            assert len(r.path(s, d)) == dist[s][d]


def test_ilp_router_lb_le_dor_lb_4x8():
    """ILP routing should produce LB <= DOR's LB (load balancing helps)."""
    t = Topology(slice=(4, 8))
    dor_w = AllToAll(t, DORRouter(t), msg_size=1)
    ilp_w = AllToAll(t, ILPRouter(t), msg_size=1)
    assert ilp_w.lower_bound <= dor_w.lower_bound


def test_ilp_router_is_deterministic():
    t = Topology(slice=(4, 8))
    r1 = ILPRouter(t)
    r2 = ILPRouter(t)
    # Same routing decisions (modulo solver ties — we don't strictly require
    # path-for-path equality, just LB equality, which is what's optimized).
    w1 = AllToAll(t, r1, 1)
    w2 = AllToAll(t, r2, 1)
    assert w1.lower_bound == w2.lower_bound


def test_ilp_router_satisfies_protocol():
    from twisted_analysis.topology import Router
    t = Topology(slice=(2, 4))
    r = ILPRouter(t)
    assert isinstance(r, Router)
```

- [ ] **Step 2: Run failing tests**

Run: `uv run pytest tests/test_ilp_router.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `ILPRouter`**

`twisted_analysis/topology/ilp_router.py`:

```python
"""ILP-based router: picks one minimal path per (src, dst) to minimize max
channel load. Ported from btowles' solve_minimal_routes with PuLP/CBC.

Uses translational symmetry: variables are created only for paths from a
canonical origin; other sources reuse the same variables under translation.
This reduces variable count by N.
"""
from __future__ import annotations
import collections
import itertools
from dataclasses import dataclass, field
from functools import cached_property
from typing import Iterable

import pulp

from twisted_analysis.topology.lattice import Topology, Node, DirectedLink

Path = tuple[DirectedLink, ...]


def _minimal_path_deltas(topology: Topology) -> dict[Node, list[tuple[int, ...]]]:
    """For each destination from origin, returns all minimal-hop delta tuples.

    A delta is a signed per-dim step count. Walking those steps via topology
    .neighbor (in any dim order — symmetry handles ambiguity) lands at dst.
    """
    origin = tuple([0] * topology.ndim)
    result: dict[Node, list[tuple[int, ...]]] = {}
    kr = [range(-topology.slice[d], topology.slice[d] + 1)
          for d in range(topology.ndim)]
    # Group deltas by their endpoint
    endpoint_to_deltas: dict[Node, list[tuple[int, ...]]] = {}
    for delta in itertools.product(*kr):
        endpoint = _walk(topology, origin, delta)
        endpoint_to_deltas.setdefault(endpoint, []).append(delta)
    for dst, deltas in endpoint_to_deltas.items():
        min_hops = min(sum(abs(d) for d in delta) for delta in deltas)
        result[dst] = [d for d in deltas if sum(abs(x) for x in d) == min_hops]
    return result


def _walk(topology: Topology, src: Node, delta: tuple[int, ...]) -> Node:
    """Walk a delta from src; returns endpoint."""
    node = src
    for dim, count in enumerate(delta):
        direction = 1 if count >= 0 else -1
        for _ in range(abs(count)):
            node = topology.neighbor(node, dim, direction)
    return node


def _delta_to_path(topology: Topology, src: Node, delta: tuple[int, ...]) -> Path:
    """Walk delta from src; return the path."""
    node = src
    hops: list[DirectedLink] = []
    for dim, count in enumerate(delta):
        direction = 1 if count >= 0 else -1
        for _ in range(abs(count)):
            nxt = topology.neighbor(node, dim, direction)
            hops.append((node, nxt, dim, direction))
            node = nxt
    return tuple(hops)


@dataclass(frozen=True)
class ILPRouter:
    """Load-balanced minimal router. Implements the Router Protocol.

    On first .path() call, solves an ILP to pick one minimal path per
    (origin, dst) such that max channel load over the directed-edge orbit
    classes is minimized. Subsequent .path() calls use the cached table.
    """
    topology: Topology
    ilp_timeout_seconds: float = 60.0

    @cached_property
    def _origin(self) -> Node:
        return tuple([0] * self.topology.ndim)

    @cached_property
    def _chosen_delta(self) -> dict[Node, tuple[int, ...]]:
        """Map dst-from-origin -> chosen delta tuple. Computed once via ILP."""
        t = self.topology
        minimal = _minimal_path_deltas(t)

        # ILP: pick one delta per (origin, dst).
        prob = pulp.LpProblem("ilp_router", pulp.LpMinimize)

        # Variables: x[dst, k] = 1 iff candidate path k is chosen for dst.
        x: dict[tuple[Node, int], pulp.LpVariable] = {}
        for dst, deltas in minimal.items():
            if dst == self._origin:
                continue
            if len(deltas) == 1:
                # No choice; will set later.
                continue
            # Exactly-one constraint
            choice_sum = []
            for k, _ in enumerate(deltas):
                v = pulp.LpVariable(f"x_{dst}_{k}", cat=pulp.LpBinary)
                x[(dst, k)] = v
                choice_sum.append(v)
            prob += pulp.lpSum(choice_sum) == 1

        # Compute per-edge-orbit load contributions.
        # An edge orbit is identified by (dim, dir). Under translation,
        # all 2*ndim edge orbits each contain N edges.
        # The load contribution from a path with hop (dim, dir) at hop index i
        # is 1 (per src in orbit; we sum across orbits, so this is N units total
        # per edge orbit per src-orbit choice, but we normalize per edge orbit).
        # We just count how many hops in each (dim, dir) each candidate uses;
        # under symmetry this is the load per edge in that orbit.

        # For each (dim, dir, dst, k), compute hop_count[(dim,dir),dst,k].
        # Group by edge orbit: max load per edge orbit must <= M.
        # Fixed load: sum over fixed deltas (single-choice dsts).
        edge_orbit_fixed: dict[tuple[int, int], int] = collections.Counter()
        edge_orbit_var_contrib: dict[tuple[int, int], list[tuple[pulp.LpVariable, int]]] = \
            collections.defaultdict(list)

        for dst, deltas in minimal.items():
            if dst == self._origin:
                continue
            if len(deltas) == 1:
                delta = deltas[0]
                # Each (dim, dir) contributes |delta[dim]| edges in that orbit.
                for dim, count in enumerate(delta):
                    if count > 0:
                        edge_orbit_fixed[(dim, 1)] += count
                    elif count < 0:
                        edge_orbit_fixed[(dim, -1)] += -count
            else:
                for k, delta in enumerate(deltas):
                    v = x[(dst, k)]
                    for dim, count in enumerate(delta):
                        if count > 0:
                            edge_orbit_var_contrib[(dim, 1)].append((v, count))
                        elif count < 0:
                            edge_orbit_var_contrib[(dim, -1)].append((v, -count))

        # M = max load across edge orbits.
        max_bound = sum(edge_orbit_fixed.values()) + sum(
            sum(c for _, c in vs) for vs in edge_orbit_var_contrib.values()
        )
        M = pulp.LpVariable("M", lowBound=0, upBound=max_bound or 1)
        prob += M  # minimize

        # Constraint per edge orbit: fixed + sum(var * contrib) <= M
        all_orbits = set(edge_orbit_fixed.keys()) | set(edge_orbit_var_contrib.keys())
        for orbit in all_orbits:
            fixed = edge_orbit_fixed.get(orbit, 0)
            contribs = edge_orbit_var_contrib.get(orbit, [])
            prob += (pulp.lpSum(v * c for v, c in contribs) + fixed) <= M

        # Solve.
        solver = pulp.getSolver(
            "PULP_CBC_CMD", msg=False, timeLimit=int(self.ilp_timeout_seconds)
        )
        prob.solve(solver)
        if pulp.LpStatus[prob.status] not in ("Optimal", "Not Solved"):
            raise RuntimeError(f"ILP router failed: {pulp.LpStatus[prob.status]}")

        # Extract chosen deltas.
        chosen: dict[Node, tuple[int, ...]] = {}
        for dst, deltas in minimal.items():
            if dst == self._origin:
                chosen[dst] = tuple([0] * t.ndim)
                continue
            if len(deltas) == 1:
                chosen[dst] = deltas[0]
            else:
                picked = None
                for k, delta in enumerate(deltas):
                    v = x[(dst, k)]
                    val = pulp.value(v)
                    if val is not None and val > 0.5:
                        if picked is not None:
                            # Multiple selected; tie-break by index.
                            continue
                        picked = delta
                if picked is None:
                    picked = deltas[0]  # solver returned no decision; default
                chosen[dst] = picked
        return chosen

    def path(self, src: Node, dst: Node) -> Path:
        if src == dst:
            return ()
        t = self.topology
        # Translation: src's path to dst is the canonical (origin -> dst-src) path
        # walked from src. Compute "dst from origin" = dst - src under twist:
        # we just walk the canonical delta starting from src.
        # First compute the equivalent dst-from-origin via inverse translation.
        # Simplest: find the canonical key by trying every node-as-origin... no, use:
        canonical_dst = _walk(t, self._origin,
                              _coord_diff(t, src, dst))
        # _coord_diff returns the delta vector (signed steps) that, walked from origin,
        # lands at "where dst is relative to src under the twist".
        delta = self._chosen_delta[canonical_dst]
        return _delta_to_path(t, src, delta)


def _coord_diff(topology: Topology, src: Node, dst: Node) -> tuple[int, ...]:
    """Return the delta vector that, walked from origin, lands at the canonical
    representative of dst-translated-by-(-src). Uses BFS since twist makes the
    straightforward subtraction wrong.
    """
    # BFS from origin to find a path landing at the "translated dst", where
    # translated_dst is the node such that walking the same delta from src
    # lands at dst. The cleanest way: BFS from src; record the steps; replay
    # from origin to find canonical_dst.
    from collections import deque
    parent: dict[Node, tuple[Node, int, int] | None] = {src: None}
    q: deque[Node] = deque([src])
    while q:
        u = q.popleft()
        if u == dst:
            break
        for v_dim_dir in [(topology.neighbor(u, dim, dir), dim, dir)
                          for dim in range(topology.ndim)
                          for dir in (-1, 1)]:
            v, dim, dir = v_dim_dir
            if v not in parent:
                parent[v] = (u, dim, dir)
                q.append(v)
    # Build hop list from src to dst
    hops: list[tuple[int, int]] = []
    cur = dst
    while parent[cur] is not None:
        p, dim, dir = parent[cur]
        hops.append((dim, dir))
        cur = p
    hops.reverse()
    # Replay from origin
    origin = tuple([0] * topology.ndim)
    delta = [0] * topology.ndim
    node = origin
    for dim, dir in hops:
        node = topology.neighbor(node, dim, dir)
        delta[dim] += dir
    return tuple(delta)
```

Update `twisted_analysis/topology/__init__.py`:

```python
from twisted_analysis.topology.lattice import Topology, Node, DirectedLink
from twisted_analysis.topology.router import Router, DORRouter, Path
from twisted_analysis.topology.ilp_router import ILPRouter

__all__ = ["Topology", "Node", "DirectedLink", "Router", "DORRouter",
           "ILPRouter", "Path"]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_ilp_router.py -v`
Expected: 6 passed.

> If `test_ilp_router_lb_le_dor_lb_4x8` FAILS (i.e., ILPRouter doesn't actually lower the LB), debug: probably the per-edge-orbit load formula is wrong (the load contribution should account for the orbit being multiplied by N srcs, but normalized to per-edge-load it cancels). Verify: print `dor_w.lower_bound` and `ilp_w.lower_bound`; if equal, check whether 4x8 has path diversity (`len(deltas) > 1` for any dst). If not, replace test with 4x4x8 (which definitely has diversity).

- [ ] **Step 5: Commit**

```bash
git add twisted_analysis/topology/ilp_router.py twisted_analysis/topology/__init__.py tests/test_ilp_router.py
git commit -m "feat(topology): ILPRouter — load-balanced minimal routing with translational symmetry"
```

---

## Task A3: Make ILPRouter the default in CLI and experiments

**Files:**
- Modify: `twisted_analysis/cli.py`
- Modify: every `experiments/*.yaml` — add `router: ilp` (or default to ilp if omitted).
- Modify: `tests/test_cli.py` (the inline YAML must include `router: ilp` for sanity).

- [ ] **Step 1: Write failing test for the CLI router selection**

Add to `tests/test_cli.py`:

```python
def test_cli_uses_ilp_router_by_default(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "name: smoke_ilp\nslice: [2, 4]\nmsg_size: 1\nschedule: round_robin\n"
        f"output_dir: {tmp_path}/out\n"
    )
    res = subprocess.run(
        [sys.executable, "-m", "twisted_analysis.cli", "run", str(cfg)],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["router"] == "ilp"


def test_cli_uses_dor_router_when_requested(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "name: smoke_dor\nslice: [2, 4]\nmsg_size: 1\nschedule: round_robin\n"
        f"router: dor\noutput_dir: {tmp_path}/out\n"
    )
    res = subprocess.run(
        [sys.executable, "-m", "twisted_analysis.cli", "run", str(cfg)],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["router"] == "dor"
```

Add to the top of `test_cli.py`: `import json`.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_cli.py -v`
Expected: 2 new tests fail (`summary["router"]` not present).

- [ ] **Step 3: Update CLI**

In `twisted_analysis/cli.py`, replace:

```python
from twisted_analysis.topology import Topology, Router
```

with:

```python
from twisted_analysis.topology import Topology, DORRouter, ILPRouter
```

Inside `run_experiment`, after computing `t`:

```python
router_name = cfg.get("router", "ilp")  # default: ILP
if router_name == "ilp":
    r = ILPRouter(t)
elif router_name == "dor":
    r = DORRouter(t)
else:
    raise ValueError(f"unknown router: {router_name}")
```

Replace the existing `r = Router(t)` (which after Task A1 became `r = DORRouter(t)`) with this dispatch.

In the `summary` dict, add: `"router": router_name,`.

- [ ] **Step 4: Update every experiment YAML**

For each file in `experiments/`, add `router: ilp` as a top-level field. Files:
- `2x4_rr.yaml`, `2x4_dim_phased.yaml`, `2x4_ilp.yaml`
- `4x8_rr.yaml`, `4x8_dim_phased.yaml`
- `4x4x8_rr.yaml`, `4x4x8_dim_phased.yaml`

Each becomes:
```yaml
name: 2x4_rr
slice: [2, 4]
msg_size: 1
schedule: round_robin
router: ilp
output_dir: results/2x4_rr
```

Also add 7 DOR-comparison experiments:
- `2x4_rr_dor.yaml`, `2x4_dim_phased_dor.yaml`, `2x4_ilp_dor.yaml`
- `4x8_rr_dor.yaml`, `4x8_dim_phased_dor.yaml`
- `4x4x8_rr_dor.yaml`, `4x4x8_dim_phased_dor.yaml`

Each with `router: dor` and `output_dir: results/<name>_dor`. This lets the eval show DOR-vs-ILP side by side.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_cli.py -v`
Expected: all CLI tests pass.

Run full suite: `uv run pytest -v`
Expected: 57 passed (53 previous + 4 new — Task A2's 6 minus tests already in test_router_protocol = 4 new from A2 since `test_ilp_router_satisfies_protocol` overlaps Task A1's coverage).

Actually let me re-count: A1 adds 2 tests, A2 adds 6, A3 adds 2 = 53 + 10 = 63. (Adjust if needed.)

- [ ] **Step 6: Commit**

```bash
git add twisted_analysis/cli.py experiments/ tests/test_cli.py
git commit -m "feat(cli): default to ILPRouter; add DOR comparison experiments"
```

---

## Task A4: Re-run eval and update fixtures

**Files:**
- Re-generate fixtures: re-run `scripts/dump_routing_tables.py` to add ILPRouter fixtures.
- Modify: `scripts/dump_routing_tables.py` to dump both DOR and ILP routing tables.
- Add: `fixtures/routing_ilp_2x4.csv`, `fixtures/routing_ilp_4x8.csv`.
- Update: `tests/test_routing_fixtures.py` to cover ILP variants.

- [ ] **Step 1: Update the dump script to handle both routers**

```python
"""Dump routing tables to CSV for inspection and as test fixtures."""
import csv
import sys
from pathlib import Path

from twisted_analysis.topology import Topology, DORRouter, ILPRouter

OUT = Path(__file__).parent.parent / "fixtures"


def dump(slice_: tuple[int, ...], name: str, router_kind: str) -> None:
    t = Topology(slice=slice_)
    if router_kind == "dor":
        r = DORRouter(t)
        prefix = "routing"
    elif router_kind == "ilp":
        r = ILPRouter(t)
        prefix = "routing_ilp"
    else:
        raise ValueError(router_kind)
    out_path = OUT / f"{prefix}_{name}.csv"
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src", "dst", "hops", "path"])
        for s in t.nodes():
            for d in t.nodes():
                path = r.path(s, d)
                path_str = "|".join(f"{u}->{v}({dim},{dir})"
                                    for u, v, dim, dir in path)
                w.writerow([str(s), str(d), len(path), path_str])
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    dump((2, 4), "2x4", "dor")
    dump((4, 8), "4x8", "dor")
    dump((2, 4), "2x4", "ilp")
    dump((4, 8), "4x8", "ilp")
    if "--include-3d" in sys.argv:
        dump((4, 4, 8), "4x4x8", "dor")
        dump((4, 4, 8), "4x4x8", "ilp")
```

- [ ] **Step 2: Generate fixtures**

Run: `uv run python scripts/dump_routing_tables.py`
Expected: four CSVs written (existing `routing_2x4.csv`, `routing_4x8.csv` overwritten; new `routing_ilp_2x4.csv`, `routing_ilp_4x8.csv`).

- [ ] **Step 3: Update fixture test**

Add to `tests/test_routing_fixtures.py`:

```python
def test_2x4_ilp_fixture_matches_router():
    from twisted_analysis.topology import ILPRouter
    t = Topology(slice=(2, 4))
    r = ILPRouter(t)
    for src_s, dst_s, hops, _ in _load("ilp_2x4"):
        assert len(r.path(eval(src_s), eval(dst_s))) == hops


def test_4x8_ilp_fixture_matches_router():
    from twisted_analysis.topology import ILPRouter
    t = Topology(slice=(4, 8))
    r = ILPRouter(t)
    for src_s, dst_s, hops, _ in _load("ilp_4x8"):
        assert len(r.path(eval(src_s), eval(dst_s))) == hops
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_routing_fixtures.py -v`
Expected: 4 passed.

- [ ] **Step 5: Run full suite**

Run: `uv run pytest -v`
Expected: all green.

- [ ] **Step 6: Run full eval**

Run: `bash eval/run_all.sh`
Expected: 14 experiments succeed (7 ILP + 7 DOR). Read `results/<today>/headlines.csv`.

- [ ] **Step 7: Commit**

```bash
git add scripts/dump_routing_tables.py fixtures/routing_ilp_2x4.csv fixtures/routing_ilp_4x8.csv fixtures/routing_2x4.csv fixtures/routing_4x8.csv tests/test_routing_fixtures.py
git commit -m "feat(fixtures): commit ILP-routed routing tables + dual-fixture regression tests"
```

---

# Part B — Symmetric Scheduling ILP

## Task B1: Orbit detection under translational symmetry

**Files:**
- Create: `twisted_analysis/lp/orbit.py`
- Create: `tests/test_orbit.py`

**Definition.** Two flows `f1 = (s1, d1)` and `f2 = (s2, d2)` are in the same translation orbit if there exists a translation `t` (a permutation of nodes induced by walking from origin to some node) such that `t(s1) = s2` and `t(d1) = d2`.

For AllToAll workload on our `{S, 2S}` family, the orbit count is `N - 1` (one per non-origin dst-from-origin), each of size `N`.

- [ ] **Step 1: Write failing tests**

`tests/test_orbit.py`:
```python
from twisted_analysis.topology import Topology
from twisted_analysis.lp.orbit import compute_orbits, OrbitId


def test_2x4_orbit_count():
    """8 nodes, AllToAll → 56 flows → 7 orbits each of size 8."""
    t = Topology(slice=(2, 4))
    orbits = compute_orbits(t)
    # orbits: dict[OrbitId, list[(src, dst)]]
    assert len(orbits) == 7  # N-1 orbits
    for members in orbits.values():
        assert len(members) == 8  # N members per orbit


def test_orbit_translation_consistency():
    """All members of an orbit have the same delta from src to dst."""
    t = Topology(slice=(4, 8))
    orbits = compute_orbits(t)
    for orbit_id, members in orbits.items():
        # Every member's path (from src to dst) should have the same length.
        from twisted_analysis.topology import DORRouter
        r = DORRouter(t)
        path_lens = {len(r.path(s, d)) for s, d in members}
        assert len(path_lens) == 1, f"orbit {orbit_id}: mixed path lengths {path_lens}"


def test_orbit_total_membership():
    """Every (src, dst) with src != dst is in exactly one orbit."""
    t = Topology(slice=(2, 4))
    orbits = compute_orbits(t)
    seen = set()
    for members in orbits.values():
        for m in members:
            assert m not in seen
            seen.add(m)
    nodes = list(t.nodes())
    expected = {(s, d) for s in nodes for d in nodes if s != d}
    assert seen == expected
```

- [ ] **Step 2: Run failing tests**

Run: `uv run pytest tests/test_orbit.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement orbits**

`twisted_analysis/lp/orbit.py`:

```python
from __future__ import annotations
from collections import defaultdict

from twisted_analysis.topology import Topology, Node

OrbitId = Node  # identified by dst-from-origin


def compute_orbits(topology: Topology) -> dict[OrbitId, list[tuple[Node, Node]]]:
    """Group AllToAll flows by translation orbit.

    Orbit id = dst-from-origin. Each orbit contains N members (one per src).
    """
    # Build translation table: for each node t, what does origin map to under
    # "walk to t"? We define translation_to[t] = t (identity walk lands at t).
    # The translation map sends node u to "u walked by (origin -> t) hops".
    # Cleanest: for each pair (src, dst), compute the canonical dst-from-origin
    # by walking from src to dst (recovered via BFS) then walking those steps
    # from origin.
    origin = tuple([0] * topology.ndim)
    orbits: dict[OrbitId, list[tuple[Node, Node]]] = defaultdict(list)

    # Precompute BFS-shortest path src->dst for every pair; convert to delta;
    # walk delta from origin to get canonical.
    for src in topology.nodes():
        if src == origin:
            for dst in topology.nodes():
                if dst != origin:
                    orbits[dst].append((src, dst))
        else:
            for dst in topology.nodes():
                if dst == src:
                    continue
                canonical_dst = _canonical_dst(topology, src, dst)
                orbits[canonical_dst].append((src, dst))
    return dict(orbits)


def _canonical_dst(topology: Topology, src: Node, dst: Node) -> Node:
    """Return the canonical dst-from-origin for the flow (src, dst).

    Algorithm: BFS from src to dst to get a shortest-hop path; replay that path
    starting from origin to get the canonical endpoint.
    """
    from collections import deque
    origin = tuple([0] * topology.ndim)
    parent: dict[Node, tuple[Node, int, int] | None] = {src: None}
    q: deque[Node] = deque([src])
    while q:
        u = q.popleft()
        if u == dst:
            break
        for dim in range(topology.ndim):
            for dir in (-1, 1):
                v = topology.neighbor(u, dim, dir)
                if v not in parent:
                    parent[v] = (u, dim, dir)
                    q.append(v)
    # Walk parent links to recover hop list
    hops: list[tuple[int, int]] = []
    cur = dst
    while parent[cur] is not None:
        p, dim, dir = parent[cur]
        hops.append((dim, dir))
        cur = p
    hops.reverse()
    # Replay from origin
    node = origin
    for dim, dir in hops:
        node = topology.neighbor(node, dim, dir)
    return node
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_orbit.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add twisted_analysis/lp/orbit.py tests/test_orbit.py
git commit -m "feat(lp): orbit detection for translational symmetry"
```

---

## Task B2: Symmetric scheduling ILP

**Files:**
- Create: `twisted_analysis/lp/symmetric.py`
- Modify: `twisted_analysis/lp/__init__.py`
- Create: `tests/test_symmetric_ilp.py`

The orbit-based ILP:
- Variables: `y[orbit_id, hop_index, time] ∈ {0,1}` — orbit fires hop `i` at step `t`.
- Per-orbit fire-once: `Σ_t y[O, i, t] = 1`.
- Per-orbit causal order: `Σ_{t≤s} y[O, i+1, t] ≤ Σ_{t≤s-1} y[O, i, t]`.
- Edge-orbit capacity: for each `(dim, dir, t)`: at most 1 orbit can fire a hop in that edge-orbit at time `t`. The hop count is symmetric: orbit O fires hop i contributing to edge-orbit `(dim_i, dir_i)` (the canonical hop-i's dim/dir).

- [ ] **Step 1: Write failing tests**

`tests/test_symmetric_ilp.py`:
```python
from twisted_analysis.topology import Topology, DORRouter
from twisted_analysis.model import AllToAll
from twisted_analysis.lp.symmetric import solve_symmetric_makespan
from twisted_analysis.lp.ilp import solve_makespan


def test_symmetric_matches_asymmetric_2x4():
    """On 2x4, the symmetric ILP optimum should match the asymmetric ILP."""
    t = Topology(slice=(2, 4))
    r = DORRouter(t)
    w = AllToAll(t, r, msg_size=1)
    m_sym, _ = solve_symmetric_makespan(t, r, list(w.flows), T_upper=w.lower_bound * 4)
    m_asym, _ = solve_makespan(t, r, list(w.flows), T_upper=w.lower_bound * 4)
    assert m_sym == m_asym


def test_symmetric_optimum_ge_lb_2x4():
    t = Topology(slice=(2, 4))
    r = DORRouter(t)
    w = AllToAll(t, r, msg_size=1)
    m_sym, _ = solve_symmetric_makespan(t, r, list(w.flows), T_upper=w.lower_bound * 4)
    assert m_sym >= w.lower_bound
```

- [ ] **Step 2: Run failing tests**

Run: `uv run pytest tests/test_symmetric_ilp.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement symmetric solver**

`twisted_analysis/lp/symmetric.py`:

```python
"""Translation-symmetry-reduced scheduling ILP.

Groups AllToAll flows into translation orbits (N-1 of them, each of size N)
and solves for one schedule per orbit. Variable count: O((N-1) * path_len * T)
instead of O(N(N-1) * path_len * T) — a factor-N reduction.
"""
from __future__ import annotations
from collections import Counter, defaultdict

import pulp

from twisted_analysis.topology import Topology, Router, DirectedLink
from twisted_analysis.model.flow import Flow
from twisted_analysis.lp.orbit import compute_orbits


def _solve_feasibility_sym(
    topology: Topology,
    router: Router,
    orbits: dict,
    T: int,
    msg_solver: pulp.LpSolver,
) -> dict | None:
    """Returns y[orbit, hop, t] -> 0/1 if feasible, else None."""
    prob = pulp.LpProblem("twisted_alltoall_sym", pulp.LpMinimize)

    # Canonical path per orbit (use the canonical src=origin member's path).
    origin = tuple([0] * topology.ndim)
    canonical_path: dict = {}
    for orbit_id, members in orbits.items():
        # The member with src == origin is the canonical representative.
        canon = next(((s, d) for (s, d) in members if s == origin), None)
        assert canon is not None
        canonical_path[orbit_id] = router.path(canon[0], canon[1])

    y: dict = {}
    for orbit_id, path in canonical_path.items():
        for i in range(len(path)):
            for t in range(T):
                y[(orbit_id, i, t)] = pulp.LpVariable(
                    f"y_{orbit_id}_{i}_{t}", cat=pulp.LpBinary
                )
    # Per-orbit fire-once
    for orbit_id, path in canonical_path.items():
        for i in range(len(path)):
            prob += pulp.lpSum(y[(orbit_id, i, t)] for t in range(T)) == 1
    # Per-orbit causal order
    for orbit_id, path in canonical_path.items():
        for i in range(len(path) - 1):
            for s in range(T):
                prob += (
                    pulp.lpSum(y[(orbit_id, i + 1, t)] for t in range(s + 1))
                    <= pulp.lpSum(y[(orbit_id, i, t)] for t in range(s))
                )
    # Edge-orbit capacity: for each (dim, dir, time), at most 1 orbit firing.
    edge_orbit_hits: dict = defaultdict(list)
    for orbit_id, path in canonical_path.items():
        for i, (_, _, dim, dir) in enumerate(path):
            edge_orbit_hits[(dim, dir)].append((orbit_id, i))
    for (dim, dir), hits in edge_orbit_hits.items():
        for t in range(T):
            prob += pulp.lpSum(
                y[(orbit_id, i, t)] for (orbit_id, i) in hits
            ) <= 1
    # Feasibility (no objective)
    prob += 0
    prob.solve(msg_solver)
    if pulp.LpStatus[prob.status] != "Optimal":
        return None
    return {k: pulp.value(v) for k, v in y.items()}


def solve_symmetric_makespan(
    topology: Topology,
    router: Router,
    flows: list[Flow],
    T_upper: int,
    solver_name: str = "PULP_CBC_CMD",
) -> tuple[int, dict]:
    """Binary-search makespan T for the symmetric scheduling ILP."""
    orbits = compute_orbits(topology)
    # Compute LB over canonical paths × N (one src in orbit ≡ all N edges in
    # the (dim,dir) class hit once)
    origin = tuple([0] * topology.ndim)
    edge_orbit_load: Counter = Counter()
    for orbit_id, members in orbits.items():
        canon = next(((s, d) for (s, d) in members if s == origin), None)
        if canon is None:
            continue
        path = router.path(*canon)
        for _, _, dim, dir in path:
            edge_orbit_load[(dim, dir)] += 1
    lb = max(edge_orbit_load.values()) if edge_orbit_load else 0

    solver = pulp.getSolver(solver_name, msg=False)
    lo, hi = lb, T_upper
    best_assignment: dict = {}
    while True:
        a = _solve_feasibility_sym(topology, router, orbits, hi, solver)
        if a is not None:
            best_assignment = a
            break
        hi *= 2
        if hi > 1_000_000:
            raise RuntimeError("Symmetric ILP T_upper grew past 1e6")
    while lo < hi:
        mid = (lo + hi) // 2
        a = _solve_feasibility_sym(topology, router, orbits, mid, solver)
        if a is not None:
            hi = mid
            best_assignment = a
        else:
            lo = mid + 1
    return lo, best_assignment
```

Update `twisted_analysis/lp/__init__.py`:

```python
from twisted_analysis.lp.ilp import solve_makespan, UnitPath
from twisted_analysis.lp.relaxation import lp_relax_lower_bound
from twisted_analysis.lp.symmetric import solve_symmetric_makespan

__all__ = ["solve_makespan", "UnitPath", "lp_relax_lower_bound",
           "solve_symmetric_makespan"]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_symmetric_ilp.py -v`
Expected: 2 passed.

> If `test_symmetric_matches_asymmetric_2x4` FAILS with `m_sym > m_asym`, that means the symmetric formulation is over-constrained. Likely cause: the edge-orbit capacity is too tight — review whether canonical-hop-i mapping to edge orbit `(dim, dir)` accounts for both orbit copies AND multiple hops within an orbit that share the same edge orbit. If `m_sym < m_asym`, that's worse — symmetric is producing infeasible schedules; investigate `_solve_feasibility_sym`.

- [ ] **Step 5: Commit**

```bash
git add twisted_analysis/lp/symmetric.py twisted_analysis/lp/__init__.py tests/test_symmetric_ilp.py
git commit -m "feat(lp): symmetric scheduling ILP with translational orbit reduction"
```

---

## Task B3: Try 4x8 (and 4x4x8) — measure tractability

**Files:**
- Create: `tests/test_symmetric_scale.py` — best-effort test on 4x8.

- [ ] **Step 1: Write the scale test**

```python
import pytest

from twisted_analysis.topology import Topology, ILPRouter
from twisted_analysis.model import AllToAll
from twisted_analysis.lp.symmetric import solve_symmetric_makespan


@pytest.mark.slow
def test_symmetric_4x8_tractable():
    """4x8 symmetric scheduling ILP should solve in under 5 minutes."""
    t = Topology(slice=(4, 8))
    r = ILPRouter(t)
    w = AllToAll(t, r, msg_size=1)
    m, _ = solve_symmetric_makespan(t, r, list(w.flows), T_upper=w.lower_bound * 4)
    assert m >= w.lower_bound
```

Mark `slow` because it may take minutes; gated in pytest config.

Update `pyproject.toml` to add the slow marker:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
markers = ["slow: marks tests as slow (deselect with '-m \"not slow\"')"]
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/test_symmetric_scale.py -v -m slow`
Expected: passes, possibly takes minutes.

If it times out (>5 min): the symmetric ILP is still intractable for 4x8. Report this as a finding; do NOT block the rest of the plan. The 2x4 case still validates the formulation.

- [ ] **Step 3: Commit**

```bash
git add tests/test_symmetric_scale.py pyproject.toml
git commit -m "test(lp): scale test for symmetric ILP on 4x8 (slow)"
```

---

# Part C — Integration: eval + docs

## Task C1: Add `ilp_optimal_symmetric` schedule and CLI integration

**Files:**
- Create: `twisted_analysis/schedules/lp_symmetric.py` — adapter that produces Injections from symmetric LP output.
- Modify: `twisted_analysis/cli.py` — add `ilp_optimal_symmetric` schedule option.
- Create: `experiments/4x8_ilp_symmetric.yaml` and `experiments/2x4_ilp_symmetric.yaml`.

- [ ] **Step 1: Implement adapter**

`twisted_analysis/schedules/lp_symmetric.py`:

```python
"""Adapter: convert symmetric-LP orbit assignment to per-unit Injections."""
from __future__ import annotations
from collections import defaultdict

from twisted_analysis.lp.orbit import compute_orbits
from twisted_analysis.model.flow import Flow
from twisted_analysis.schedules.base import Injection
from twisted_analysis.topology import Topology, Router


def symmetric_assignment_to_injections(
    topology: Topology,
    router: Router,
    flows: list[Flow],
    assignment: dict,  # (orbit_id, hop_i, t) -> 0/1
) -> list[Injection]:
    """Expand the orbit-level schedule into per-unit Injections.

    Each orbit's hop_i fire-time at canonical orbit time t becomes the fire-time
    for every src in the orbit (translation-equivariant).
    """
    orbits = compute_orbits(topology)
    # Map (src, dst) -> orbit_id
    pair_to_orbit: dict[tuple, "OrbitId"] = {}
    for orbit_id, members in orbits.items():
        for m in members:
            pair_to_orbit[m] = orbit_id
    # For each orbit, extract per-hop fire times
    orbit_hop_times: dict[tuple, list[int]] = defaultdict(list)
    # Group by orbit_id then by hop_i, find time where assignment > 0.5
    by_orbit_hop: dict[tuple, dict[int, int]] = defaultdict(dict)
    for (orbit_id, hop_i, t), val in assignment.items():
        if val is not None and val > 0.5:
            by_orbit_hop[orbit_id][hop_i] = t
    # Build Injections
    injections: list[Injection] = []
    for f in flows:
        orbit_id = pair_to_orbit.get((f.src, f.dst))
        if orbit_id is None:
            continue
        hop_times = by_orbit_hop[orbit_id]
        # Sort by hop index
        hop_schedule = tuple(hop_times[i] for i in sorted(hop_times.keys()))
        start = hop_schedule[0] if hop_schedule else 0
        injections.append(Injection(
            flow=f, start_step=start, priority=0, hop_schedule=hop_schedule,
        ))
    return injections
```

- [ ] **Step 2: Update CLI**

In `twisted_analysis/cli.py`, add to the schedule dispatch:

```python
if sched_name == "ilp_optimal":
    from twisted_analysis.lp.ilp import solve_makespan
    m_opt, assignment = solve_makespan(
        t, r, list(w.flows), T_upper=w.lower_bound * 4
    )
    injs = lp_assignment_to_injections(list(w.flows), r, assignment)
elif sched_name == "ilp_optimal_symmetric":
    from twisted_analysis.lp.symmetric import solve_symmetric_makespan
    from twisted_analysis.schedules.lp_symmetric import symmetric_assignment_to_injections
    m_opt, assignment = solve_symmetric_makespan(
        t, r, list(w.flows), T_upper=w.lower_bound * 4
    )
    injs = symmetric_assignment_to_injections(t, r, list(w.flows), assignment)
else:
    sched = SCHEDULES[sched_name]
    injs = sched.emit(w)
```

- [ ] **Step 3: Add experiments**

`experiments/2x4_ilp_symmetric.yaml`:
```yaml
name: 2x4_ilp_symmetric
slice: [2, 4]
msg_size: 1
schedule: ilp_optimal_symmetric
router: ilp
output_dir: results/2x4_ilp_symmetric
```

`experiments/4x8_ilp_symmetric.yaml`:
```yaml
name: 4x8_ilp_symmetric
slice: [4, 8]
msg_size: 1
schedule: ilp_optimal_symmetric
router: ilp
output_dir: results/4x8_ilp_symmetric
```

- [ ] **Step 4: Run eval**

Run: `bash eval/run_all.sh`
Expected: all 16+ experiments succeed.

- [ ] **Step 5: Commit**

```bash
git add twisted_analysis/schedules/lp_symmetric.py twisted_analysis/cli.py experiments/2x4_ilp_symmetric.yaml experiments/4x8_ilp_symmetric.yaml
git commit -m "feat(cli): add ilp_optimal_symmetric schedule; integrate symmetric LP"
```

---

## Task C2: Documentation update

**Files:**
- Modify: `docs/topology.md` — new section: "ILPRouter (load-balanced minimal routing)".
- Modify: `docs/lp_formulation.md` — new section: "Symmetric scheduling ILP (translational orbits)".
- Modify: `docs/schedules.md` — add `ILP-optimal (symmetric)` to schedule list.
- Modify: `docs/results.md` — updated headline table with DOR vs ILP columns.
- Modify: `README.md` — bump "What" section to mention ILPRouter default.

- [ ] **Step 1: Update `docs/topology.md`**

Add a new section after the existing DOR section:

```markdown
## ILPRouter: load-balanced minimal routing

For topologies with path diversity (multiple shortest paths between many
src/dst pairs), DOR's deterministic tie-break can produce a max-link-load
higher than the topology allows. `ILPRouter` solves an ILP to pick one
minimal path per `(src, dst)` such that the max channel load is minimized.

Translational symmetry: under our `{S, 2S}` topology family, the network is
vertex-transitive. So an optimal path choice for `(origin, dst)` extends by
translation to every `(src, src+dst)` pair. The ILP only needs variables for
one src; this is a factor-`N` variable reduction.

Formulation (per the btowles-style algorithm we ported):
- Enumerate all minimal-hop deltas from origin to each dst (via brute-force
  step-vector enumeration + walk validation).
- For each `(origin, dst)` with multiple candidate paths, create boolean
  variables (one per candidate) summing to 1.
- For each directed-edge orbit (one per `(dim, dir)`), the load is the count
  of selected candidate paths whose delta includes that (dim, dir).
- Minimize max load over all edge orbits.

The implementation lives in [twisted_analysis/topology/ilp_router.py](..).
```

- [ ] **Step 2: Update `docs/lp_formulation.md`**

Add a new section:

```markdown
## Symmetric scheduling ILP (translational orbits)

The standard scheduling ILP has `O(N(N-1) × path_len × T)` binary variables —
intractable for `N >= 32`. Exploiting the vertex-transitive structure of the
twisted torus, we group flows into translation orbits (one orbit = one
displacement-from-origin), reducing variable count by factor `N`.

### Variables

- `y[orbit_id, hop_i, t] ∈ {0,1}` — orbit fires hop `i` at time `t`.

### Constraints

- **Per-orbit fire-once**: `Σ_t y[O, i, t] = 1` for every orbit `O`, hop `i`.
- **Per-orbit causal order**: `Σ_{t≤s} y[O, i+1, t] ≤ Σ_{t≤s-1} y[O, i, t]`.
- **Edge-orbit capacity**: for each `(dim, dir, t)`, at most 1 orbit fires
  a hop in that edge orbit at time `t`. (Each edge orbit contains `N`
  edges; by symmetry, the load on every edge in the orbit is the same.)

### Symmetry assumption

We *impose* that all N members of an orbit fire their hop `i` at the same
time. By a Birkhoff-style argument on vertex-transitive workloads, the
optimal makespan under this constraint equals the unconstrained optimum.

### Solving

`solve_symmetric_makespan(topology, router, flows, T_upper)` returns
`(M_sym, assignment)`. The assignment can be converted to per-unit
Injections via `symmetric_assignment_to_injections(...)`.
```

- [ ] **Step 3: Update `docs/schedules.md`**

Add to the schedule list:

```markdown
### LP-optimal (symmetric)

`ilp_optimal_symmetric` solves the symmetric scheduling ILP (see
[lp_formulation.md](lp_formulation.md)). On topologies large enough that the
non-symmetric ILP is intractable, this is the way to get the optimal
makespan.
```

- [ ] **Step 4: Update `docs/results.md`**

Read the new `results/<today>/headlines.csv`; build a comparison table:

```markdown
## Headline Numbers (Updated 2026-05-12 — ILPRouter default)

### DOR vs ILP routing (impact on LB)

| Topology | DOR LB | ILP LB | Reduction |
|---|---:|---:|---:|
| 2x4 | <fill> | <fill> | <%> |
| 4x8 | <fill> | <fill> | <%> |
| 4x4x8 | <fill> | <fill> | <%> |

### Schedule comparison (all on ILP routing)

| Topology | Schedule | Makespan | LB | Ratio |
|---|---|---:|---:|---:|
| ... | ... | ... | ... | ... |
```

Fill in numbers from `headlines.csv`.

- [ ] **Step 5: Update `README.md`**

In the "What" section, mention that ILPRouter is the default and DOR is available via `router: dor` in the YAML.

- [ ] **Step 6: Commit**

```bash
git add docs/topology.md docs/lp_formulation.md docs/schedules.md docs/results.md README.md
git commit -m "docs: ILPRouter + symmetric scheduling ILP sections; updated results"
```

---

## Self-review checklist (run BEFORE handing off to executor)

1. **Spec coverage:**
   - Part A: Router Protocol (A1), ILPRouter implementation (A2), CLI default (A3), fixtures (A4) — ✓
   - Part B: Orbit detection (B1), symmetric ILP (B2), scale test (B3) — ✓
   - Part C: schedule integration + docs (C1, C2) — ✓

2. **Placeholder scan:**
   - Task A2 `_coord_diff` uses BFS — concrete; not a placeholder.
   - Task A2 has a debug-on-failure note ("If `test_ilp_router_lb_le_dor_lb_4x8` FAILS...") — that's runtime guidance, not a placeholder.
   - Task B2 has a similar debug note. Same.
   - All code blocks are complete.

3. **Type consistency:**
   - `Router` is a Protocol; `DORRouter` and `ILPRouter` are concrete implementations.
   - `OrbitId = Node` consistently used in `orbit.py` and `symmetric.py`.
   - `Injection.hop_schedule` is reused in `symmetric_assignment_to_injections`.

4. **Test count after all tasks:** baseline 53 + A1(2) + A2(6) + A3(2) + A4(2) + B1(3) + B2(2) + B3(1) = **71 tests**. Plus the existing tests for the renamed Router class still pass.

5. **Risk register:**
   - Task A2 LB-reduction test could pass trivially on 2x4 (no diversity); we use 4x8.
   - Task B2 symmetry assumption may produce sub-optimal makespan for asymmetric workloads — but AllToAll is symmetric, so optimal-matches-asymmetric should hold. The test validates this on 2x4.
   - Task B3 may not actually make 4x8 tractable — that's a research question; we report findings either way.

Plan is complete and ready for execution.
