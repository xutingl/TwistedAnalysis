# TwistedAnalysis — AllToAll Gap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python package + simulator + ILP solver + evaluation harness that quantifies the AllToAll performance gap on twisted-torus topologies (2x4, 4x8, 4x4x8) under fixed DOR routing, comparing analytical lower bound, LP/ILP optimum, and two heuristic schedules.

**Architecture:** A layered library `twisted_analysis/` with five modules — `topology`, `model`, `schedules`, `simulator`, `lp` — plus `viz` and a CLI. Experiments are YAML configs; reproduced via `eval/run_all.sh`. TDD throughout for topology/model/simulator; oracle-based validation for the LP layer.

**Tech Stack:** Python 3.11+, `uv` for env management, `pytest` for testing, `PuLP` (CBC backend) for ILP, `numpy`, `matplotlib` for plots, `pyyaml` for experiment configs.

**Reference spec:** [docs/superpowers/specs/2026-05-12-twisted-torus-alltoall-design.md](../specs/2026-05-12-twisted-torus-alltoall-design.md)

**Conventions for every task:**
- Working directory: `/home/xutingl/collective_comm/TwistedAnalysis/`.
- Run tests with `uv run pytest <path> -v` (fall back to `.venv/bin/python -m pytest` if `uv run` re-syncs unexpectedly).
- Commit messages use Conventional Commits prefixes: `feat:`, `test:`, `chore:`, `docs:`, `fix:`.
- Never amend commits.

---

## File structure (locked in here)

```
twisted_analysis/
├── __init__.py
├── topology/
│   ├── __init__.py        # re-exports Topology, Router
│   ├── lattice.py         # Topology class, neighbor(), links(), BFS
│   └── router.py          # DOR router, routing-table dump/load
├── model/
│   ├── __init__.py        # re-exports AllToAll, Flow
│   ├── flow.py            # Flow dataclass + AllToAll workload + link-load + LB
│   └── bounds.py          # bottleneck reporting, bisection check (Task 16)
├── schedules/
│   ├── __init__.py
│   ├── base.py            # Injection, ScheduleResult, Schedule protocol
│   ├── round_robin.py     # Latin-square schedule
│   ├── dim_phased.py      # Dimension-ordered phased schedule
│   └── lp_optimal.py      # extracts schedule from LP result
├── simulator/
│   ├── __init__.py
│   ├── engine.py          # step-sync engine
│   └── instrumentation.py # Gantt, heatmap, idle-trace
├── lp/
│   ├── __init__.py
│   ├── ilp.py             # time-indexed ILP with binary search on T
│   └── relaxation.py      # LP relaxation
├── viz/
│   ├── __init__.py
│   ├── load_histogram.py
│   ├── gantt.py
│   └── heatmap.py
└── cli.py                 # `python -m twisted_analysis run experiments/2x4_rr.yaml`

experiments/               # YAML per experiment
eval/run_all.sh
fixtures/                  # committed routing tables, expected loads
tests/                     # mirrors twisted_analysis/ layout
docs/                      # algorithm.md, topology.md, schedules.md, lp_formulation.md, evaluation.md, results.md
```

---

### Task 1: Project bootstrap

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `twisted_analysis/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "twisted-analysis"
version = "0.1.0"
description = "AllToAll performance-gap analysis on twisted-torus topologies"
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "pulp>=2.8",
    "matplotlib>=3.8",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-xdist>=3.5"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["twisted_analysis"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
```

- [ ] **Step 2: Write `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
results/
*.egg-info/
build/
dist/
.DS_Store
.coverage
```

- [ ] **Step 3: Create empty package init files**

`twisted_analysis/__init__.py`:
```python
"""TwistedAnalysis: AllToAll gap analysis on twisted-torus topologies."""
__version__ = "0.1.0"
```

`tests/__init__.py`: empty file.

- [ ] **Step 4: Write smoke test**

`tests/test_smoke.py`:
```python
import twisted_analysis

def test_package_imports():
    assert twisted_analysis.__version__ == "0.1.0"
```

- [ ] **Step 5: Create venv and install**

Run:
```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
uv venv
uv pip install -e ".[dev]"
```

- [ ] **Step 6: Run smoke test**

Run: `uv run pytest tests/test_smoke.py -v`
Expected: 1 passed.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore twisted_analysis/ tests/
git commit -m "chore: bootstrap project (pyproject, venv, smoke test)"
```

Do NOT commit `uv.lock` yet (it will be regenerated repeatedly during early dev); add it on a later, stable task.

---

### Task 2: Topology — `neighbor()` ported from reference

**Files:**
- Create: `twisted_analysis/topology/__init__.py`
- Create: `twisted_analysis/topology/lattice.py`
- Create: `tests/test_topology_neighbor.py`

- [ ] **Step 1: Write failing tests**

`tests/test_topology_neighbor.py`:
```python
import pytest
from twisted_analysis.topology.lattice import Topology

def test_2x4_inplane_step():
    t = Topology(slice=(2, 4))
    assert t.neighbor((0, 0), dim=1, dir=1) == (0, 1)
    assert t.neighbor((0, 1), dim=1, dir=-1) == (0, 0)

def test_2x4_smalldim_wrap_is_half_shift():
    t = Topology(slice=(2, 4))
    # (1, 0) +dim0 wraps: shift every coord by slice[0]=2
    assert t.neighbor((1, 0), dim=0, dir=1) == (0, 2)
    # Backward wrap from (0, 0)
    assert t.neighbor((0, 0), dim=0, dir=-1) == (1, 2)

def test_2x4_bigdim_wrap_has_no_effective_twist():
    t = Topology(slice=(2, 4))
    assert t.neighbor((0, 3), dim=1, dir=1) == (0, 0)
    assert t.neighbor((0, 0), dim=1, dir=-1) == (0, 3)

def test_4x8_smalldim_wrap_shifts_by_4():
    t = Topology(slice=(4, 8))
    assert t.neighbor((3, 0), dim=0, dir=1) == (0, 4)
    assert t.neighbor((0, 0), dim=0, dir=-1) == (3, 4)

def test_4x4x8_both_small_dims_twist_into_big():
    t = Topology(slice=(4, 4, 8))
    assert t.neighbor((3, 0, 0), dim=0, dir=1) == (0, 0, 4)
    assert t.neighbor((0, 3, 0), dim=1, dir=1) == (0, 0, 4)
    assert t.neighbor((0, 0, 7), dim=2, dir=1) == (0, 0, 0)

def test_assert_S_or_2S_only():
    with pytest.raises(AssertionError):
        Topology(slice=(2, 6))  # 6 is neither S=2 nor 2S=4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_topology_neighbor.py -v`
Expected: ModuleNotFoundError on `twisted_analysis.topology.lattice`.

- [ ] **Step 3: Implement `Topology`**

`twisted_analysis/topology/lattice.py`:
```python
from __future__ import annotations
from dataclasses import dataclass

Node = tuple[int, ...]


@dataclass(frozen=True)
class Topology:
    """A twisted-torus topology with shape `slice` (all sizes in {S, 2S})."""
    slice: tuple[int, ...]

    def __post_init__(self) -> None:
        s_min = min(self.slice)
        assert all(s in (s_min, 2 * s_min) for s in self.slice), (
            f"slice {self.slice} violates the {{S, 2S}} family"
        )

    @property
    def n_nodes(self) -> int:
        n = 1
        for s in self.slice:
            n *= s
        return n

    def neighbor(self, node: Node, dim: int, dir: int) -> Node:
        assert len(node) == len(self.slice)
        assert 0 <= dim < len(self.slice)
        assert dir in (-1, 1)

        new = list(node)
        new[dim] += dir
        wrapped = new[dim] < 0 or new[dim] >= self.slice[dim]
        if wrapped:
            shift = self.slice[dim]
            new = [(new[i] + shift) % self.slice[i] for i in range(len(new))]
        return tuple(new)
```

- [ ] **Step 4: Create `__init__.py`**

`twisted_analysis/topology/__init__.py`:
```python
from twisted_analysis.topology.lattice import Topology, Node

__all__ = ["Topology", "Node"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_topology_neighbor.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add twisted_analysis/topology/ tests/test_topology_neighbor.py
git commit -m "feat(topology): port reference twisted-torus neighbor function"
```

---

### Task 3: Topology — links() enumeration and BFS distance

**Files:**
- Modify: `twisted_analysis/topology/lattice.py`
- Create: `tests/test_topology_links_bfs.py`

- [ ] **Step 1: Write failing tests**

`tests/test_topology_links_bfs.py`:
```python
from twisted_analysis.topology.lattice import Topology


def test_2x4_node_iteration():
    t = Topology(slice=(2, 4))
    nodes = list(t.nodes())
    assert len(nodes) == 8
    assert (0, 0) in nodes and (1, 3) in nodes


def test_2x4_link_count():
    # 8 nodes, each has 2 dims * 2 dirs = 4 directed neighbors → 32 directed edges.
    t = Topology(slice=(2, 4))
    links = list(t.directed_links())
    assert len(links) == 8 * 4


def test_2x4_link_endpoints_consistent():
    t = Topology(slice=(2, 4))
    for u, v, dim, dir in t.directed_links():
        assert t.neighbor(u, dim, dir) == v


def test_2x4_bfs_distance_symmetry():
    t = Topology(slice=(2, 4))
    dist = t.bfs_distances()
    for u in t.nodes():
        for v in t.nodes():
            assert dist[u][v] == dist[v][u]


def test_2x4_bfs_zero_to_self():
    t = Topology(slice=(2, 4))
    dist = t.bfs_distances()
    for u in t.nodes():
        assert dist[u][u] == 0


def test_4x4x8_bfs_known_pair():
    # (0,0,0) → (3,0,0): one backward wrap on dim 0 → (3, 0, 4). Then need (-, 0, -4).
    # Or four forward dim-0 steps: (1,0,0) → ... → (3,0,0). 3 hops.
    # Compare against BFS truth.
    t = Topology(slice=(4, 4, 8))
    dist = t.bfs_distances()
    assert dist[(0, 0, 0)][(3, 0, 0)] == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_topology_links_bfs.py -v`
Expected: AttributeError on `nodes` / `directed_links` / `bfs_distances`.

- [ ] **Step 3: Extend `Topology`**

Append to `twisted_analysis/topology/lattice.py`:
```python
from collections import deque
from itertools import product


def _expand_methods():
    pass  # placeholder so the diff reads cleanly; remove if you prefer
```

Actually replace `Topology` with the extended version below (full file):

```python
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from itertools import product
from typing import Iterator

Node = tuple[int, ...]
DirectedLink = tuple[Node, Node, int, int]  # (u, v, dim, dir)


@dataclass(frozen=True)
class Topology:
    """A twisted-torus topology with shape `slice` (all sizes in {S, 2S})."""
    slice: tuple[int, ...]

    def __post_init__(self) -> None:
        s_min = min(self.slice)
        assert all(s in (s_min, 2 * s_min) for s in self.slice), (
            f"slice {self.slice} violates the {{S, 2S}} family"
        )

    @property
    def n_nodes(self) -> int:
        n = 1
        for s in self.slice:
            n *= s
        return n

    @property
    def ndim(self) -> int:
        return len(self.slice)

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

    def nodes(self) -> Iterator[Node]:
        yield from product(*(range(s) for s in self.slice))

    def directed_links(self) -> Iterator[DirectedLink]:
        for u in self.nodes():
            for dim in range(self.ndim):
                for dir in (-1, 1):
                    v = self.neighbor(u, dim, dir)
                    yield (u, v, dim, dir)

    def bfs_distances(self) -> dict[Node, dict[Node, int]]:
        adj: dict[Node, list[Node]] = {u: [] for u in self.nodes()}
        for u, v, _, _ in self.directed_links():
            adj[u].append(v)
        result: dict[Node, dict[Node, int]] = {}
        for src in self.nodes():
            dist = {src: 0}
            q: deque[Node] = deque([src])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        q.append(v)
            result[src] = dist
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_topology_links_bfs.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add twisted_analysis/topology/lattice.py tests/test_topology_links_bfs.py
git commit -m "feat(topology): node iteration, directed links, BFS distances"
```

---

### Task 4: Router — DOR with twist-aware shortest-path

**Files:**
- Create: `twisted_analysis/topology/router.py`
- Modify: `twisted_analysis/topology/__init__.py`
- Create: `tests/test_router.py`

The router resolves each `(src, dst)` to one path. Because the smaller-dim wrap shifts the larger dim, we enumerate signed wrap choices per dim (the displacement vectors `(δ_0, ..., δ_{d-1})` that satisfy `src + δ ≡ dst` under twist), pick the candidate with minimum `Σ |δ_i|` (shortest hop count), tie-break by preferring no-wrap then `+dir`, and walk in fixed dim order (default: largest dim first).

- [ ] **Step 1: Write failing tests**

`tests/test_router.py`:
```python
from twisted_analysis.topology.lattice import Topology
from twisted_analysis.topology.router import Router


def test_2x4_self_path_is_empty():
    t = Topology(slice=(2, 4))
    r = Router(t)
    assert r.path((0, 0), (0, 0)) == ()


def test_2x4_one_hop_inplane():
    t = Topology(slice=(2, 4))
    r = Router(t)
    p = r.path((0, 0), (0, 1))
    assert len(p) == 1
    u, v, dim, dir = p[0]
    assert u == (0, 0) and v == (0, 1)


def test_2x4_twist_shortcut_one_hop():
    # (0, 0) -> (1, 2) is a single backward wrap on dim 0 (since slice[0]=2 shifts dim 1 by 2).
    t = Topology(slice=(2, 4))
    r = Router(t)
    p = r.path((0, 0), (1, 2))
    assert len(p) == 1


def test_dor_path_length_equals_bfs_distance_2x4():
    t = Topology(slice=(2, 4))
    r = Router(t)
    dist = t.bfs_distances()
    for s in t.nodes():
        for d in t.nodes():
            assert len(r.path(s, d)) == dist[s][d], (
                f"DOR path {s}->{d} length {len(r.path(s, d))} != BFS {dist[s][d]}"
            )


def test_dor_path_length_equals_bfs_distance_4x8():
    t = Topology(slice=(4, 8))
    r = Router(t)
    dist = t.bfs_distances()
    for s in t.nodes():
        for d in t.nodes():
            assert len(r.path(s, d)) == dist[s][d]


def test_dor_path_length_equals_bfs_distance_4x4x8():
    t = Topology(slice=(4, 4, 8))
    r = Router(t)
    dist = t.bfs_distances()
    for s in t.nodes():
        for d in t.nodes():
            assert len(r.path(s, d)) == dist[s][d]


def test_router_is_deterministic():
    t = Topology(slice=(4, 8))
    r1 = Router(t)
    r2 = Router(t)
    for s in t.nodes():
        for d in t.nodes():
            assert r1.path(s, d) == r2.path(s, d)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_router.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `Router`**

`twisted_analysis/topology/router.py`:
```python
from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property
from itertools import product

from twisted_analysis.topology.lattice import Topology, Node, DirectedLink

# A "Hop" is one directed link: (u, v, dim, dir).
Path = tuple[DirectedLink, ...]


@dataclass(frozen=True)
class Router:
    """Twist-aware DOR router. Resolves dims in order of decreasing slice size.

    For each (src, dst), enumerates candidate wrap choices, picks the minimum
    hop count with deterministic tie-break (no-wrap > wrap, +dir > -dir).
    """
    topology: Topology

    @cached_property
    def _dim_order(self) -> tuple[int, ...]:
        # Resolve largest dim first (configurable later).
        return tuple(sorted(range(self.topology.ndim),
                            key=lambda d: -self.topology.slice[d]))

    def _candidate_displacements(self, src: Node, dst: Node) -> list[tuple[int, ...]]:
        """Enumerate per-dim signed wrap counts and resulting (δ_0, ..., δ_{d-1}).

        For each dim d we choose a wrap count w_d ∈ {-1, 0, +1} (single-wrap regime,
        sufficient for our {S, 2S} slices where diameter ≤ 2 per dim). The induced
        displacement is computed in coordinate space directly by simulating the
        twist via Topology.neighbor.
        """
        # Brute-force: BFS to compute the actual shortest paths via topology BFS
        # gives the *length*, but we still need a deterministic *path*. Use a
        # constructive algorithm: try all 3^ndim wrap-count combinations, compute
        # endpoint, keep those that land on `dst`, score by total |steps|.
        t = self.topology
        candidates: list[tuple[int, tuple[int, ...]]] = []
        for wraps in product((-1, 0, 1), repeat=t.ndim):
            # Compute the endpoint reached by taking, for each dim, the displacement
            # (dst[d] - src[d]) mod slice[d] adjusted for `wraps[d]` (which means
            # +wraps[d] extra full wraps in dim d). This is annoying with twist;
            # easier to simulate. We compute each dim's signed step count:
            steps_per_dim: list[int] = []
            for d in range(t.ndim):
                # Base in-plane delta in {-slice[d]+1, ..., slice[d]-1}
                base = dst[d] - src[d]
                steps_per_dim.append(base + wraps[d] * t.slice[d])
            # Total hop count under this wrap choice
            total = sum(abs(s) for s in steps_per_dim)
            candidates.append((total, tuple(steps_per_dim)))
        # Filter to feasible (twist must reconcile when we actually walk it).
        # Validate by simulating the walk; only the candidates that reach dst are kept.
        feasible: list[tuple[int, tuple[int, ...]]] = []
        for total, steps in candidates:
            endpoint = self._walk_endpoint(src, steps)
            if endpoint == dst:
                feasible.append((total, steps))
        if not feasible:
            # Fallback: BFS-shortest path reconstruction (handles 3D-with-twist edge cases).
            return [self._bfs_steps(src, dst)]
        # Tie-break: smaller total; then prefer (steps_per_dim) lexicographically
        # closest to "no-wrap": smaller |total wraps|, then prefer +dir over -dir.
        feasible.sort(key=lambda x: (
            x[0],
            sum(abs(s) > self.topology.slice[d] // 2
                for d, s in enumerate(x[1])),  # count of "wrap-direction" dims
            tuple(-s for s in x[1]),  # prefer + over -
        ))
        return [feasible[0][1]]

    def _walk_endpoint(self, src: Node, steps: tuple[int, ...]) -> Node:
        """Walk `steps[d]` signed in-plane steps in each dim in self._dim_order.
        Each step may wrap (and twist) at slice boundaries via Topology.neighbor.
        """
        node = src
        for dim in self._dim_order:
            count = steps[dim]
            direction = 1 if count >= 0 else -1
            for _ in range(abs(count)):
                node = self.topology.neighbor(node, dim, direction)
        return node

    def _bfs_steps(self, src: Node, dst: Node) -> tuple[int, ...]:
        """Recover a step vector via BFS — fallback for tricky 3D twist cases.
        Returns a `steps` tuple consistent with `_walk_endpoint(src, steps) == dst`.
        """
        # Simple: pick shortest BFS path and convert hops to per-dim signed counts
        # in dim-order. Since each BFS hop is (dim, dir), grouping by dim works.
        from collections import deque
        adj: dict[Node, list[tuple[Node, int, int]]] = {u: [] for u in self.topology.nodes()}
        for u, v, dim, dir in self.topology.directed_links():
            adj[u].append((v, dim, dir))
        parent: dict[Node, tuple[Node, int, int] | None] = {src: None}
        q: deque[Node] = deque([src])
        while q:
            u = q.popleft()
            if u == dst:
                break
            for v, dim, dir in adj[u]:
                if v not in parent:
                    parent[v] = (u, dim, dir)
                    q.append(v)
        # Walk parent links to build hop list
        hops: list[tuple[int, int]] = []
        cur = dst
        while parent[cur] is not None:
            p, dim, dir = parent[cur]
            hops.append((dim, dir))
            cur = p
        hops.reverse()
        # Convert to per-dim signed step counts; this may not respect _dim_order strictly,
        # but it's a fallback for rare cases.
        steps = [0] * self.topology.ndim
        for dim, dir in hops:
            steps[dim] += dir
        # Verify
        assert self._walk_endpoint(src, tuple(steps)) == dst, (
            f"BFS fallback failed for {src} -> {dst}"
        )
        return tuple(steps)

    def path(self, src: Node, dst: Node) -> Path:
        if src == dst:
            return ()
        [steps] = self._candidate_displacements(src, dst)
        # Construct the actual link sequence by walking in _dim_order
        node = src
        hops: list[DirectedLink] = []
        for dim in self._dim_order:
            count = steps[dim]
            direction = 1 if count >= 0 else -1
            for _ in range(abs(count)):
                nxt = self.topology.neighbor(node, dim, direction)
                hops.append((node, nxt, dim, direction))
                node = nxt
        assert node == dst, f"router walk landed at {node}, expected {dst}"
        return tuple(hops)
```

Update `twisted_analysis/topology/__init__.py`:
```python
from twisted_analysis.topology.lattice import Topology, Node, DirectedLink
from twisted_analysis.topology.router import Router, Path

__all__ = ["Topology", "Node", "DirectedLink", "Router", "Path"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_router.py -v`
Expected: 7 passed. If `test_dor_path_length_equals_bfs_distance_4x4x8` fails on a small number of pairs, the BFS fallback inside `_candidate_displacements` should kick in; tighten the candidate enumeration or accept the fallback.

- [ ] **Step 5: Commit**

```bash
git add twisted_analysis/topology/router.py twisted_analysis/topology/__init__.py tests/test_router.py
git commit -m "feat(topology): twist-aware DOR router with BFS-fallback"
```

---

### Task 5: Routing-table fixtures

**Files:**
- Create: `fixtures/routing_2x4.csv`
- Create: `fixtures/routing_4x8.csv`
- Create: `scripts/dump_routing_tables.py`
- Create: `tests/test_routing_fixtures.py`

- [ ] **Step 1: Write the dump script**

`scripts/dump_routing_tables.py`:
```python
"""Dump routing tables to CSV for inspection and as test fixtures."""
import csv
import sys
from pathlib import Path

from twisted_analysis.topology import Topology, Router

OUT = Path(__file__).parent.parent / "fixtures"


def dump(slice_: tuple[int, ...], name: str) -> None:
    t = Topology(slice=slice_)
    r = Router(t)
    out_path = OUT / f"routing_{name}.csv"
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src", "dst", "hops", "path"])
        for s in t.nodes():
            for d in t.nodes():
                path = r.path(s, d)
                path_str = "|".join(f"{u}->{v}({dim},{dir})" for u, v, dim, dir in path)
                w.writerow([str(s), str(d), len(path), path_str])
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    dump((2, 4), "2x4")
    dump((4, 8), "4x8")
    # 4x4x8 is large; gate behind explicit flag.
    if "--include-3d" in sys.argv:
        dump((4, 4, 8), "4x4x8")
```

- [ ] **Step 2: Generate the 2x4 and 4x8 fixtures**

Run: `uv run python scripts/dump_routing_tables.py`
Expected: Two CSVs written under `fixtures/`.

- [ ] **Step 3: Write a regression test that loads the fixtures**

`tests/test_routing_fixtures.py`:
```python
import csv
from pathlib import Path

from twisted_analysis.topology import Topology, Router

FIXT = Path(__file__).parent.parent / "fixtures"


def _load(name: str) -> list[tuple[str, str, int, str]]:
    rows = []
    with (FIXT / f"routing_{name}.csv").open() as f:
        r = csv.reader(f)
        next(r)  # header
        for row in r:
            rows.append((row[0], row[1], int(row[2]), row[3]))
    return rows


def test_2x4_fixture_matches_router():
    t = Topology(slice=(2, 4))
    r = Router(t)
    for src_s, dst_s, hops, path_str in _load("2x4"):
        src = eval(src_s)
        dst = eval(dst_s)
        path = r.path(src, dst)
        assert len(path) == hops


def test_4x8_fixture_matches_router():
    t = Topology(slice=(4, 8))
    r = Router(t)
    for src_s, dst_s, hops, _ in _load("4x8"):
        assert len(r.path(eval(src_s), eval(dst_s))) == hops
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_routing_fixtures.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/dump_routing_tables.py fixtures/routing_2x4.csv fixtures/routing_4x8.csv tests/test_routing_fixtures.py
git commit -m "feat(topology): commit DOR routing-table fixtures for 2x4, 4x8"
```

---

### Task 6: Flow / AllToAll / link-load / LB

**Files:**
- Create: `twisted_analysis/model/__init__.py`
- Create: `twisted_analysis/model/flow.py`
- Create: `tests/test_model.py`

- [ ] **Step 1: Write failing tests**

`tests/test_model.py`:
```python
from twisted_analysis.topology import Topology, Router
from twisted_analysis.model import AllToAll, Flow


def test_alltoall_flow_count_2x4():
    t = Topology(slice=(2, 4))
    r = Router(t)
    w = AllToAll(t, r, msg_size=1)
    assert len(w.flows) == 8 * 7  # N*(N-1)


def test_alltoall_flow_size():
    t = Topology(slice=(2, 4))
    r = Router(t)
    w = AllToAll(t, r, msg_size=4)
    assert all(f.size == 4 for f in w.flows)


def test_link_load_sums_to_total_hops_times_m():
    t = Topology(slice=(2, 4))
    r = Router(t)
    w = AllToAll(t, r, msg_size=1)
    total_load = sum(w.link_load.values())
    expected = sum(len(w.path(f)) for f in w.flows)
    assert total_load == expected


def test_lower_bound_is_max_link_load():
    t = Topology(slice=(2, 4))
    r = Router(t)
    w = AllToAll(t, r, msg_size=1)
    assert w.lower_bound == max(w.link_load.values())


def test_lower_bound_scales_with_msg_size():
    t = Topology(slice=(2, 4))
    r = Router(t)
    w1 = AllToAll(t, r, msg_size=1)
    w4 = AllToAll(t, r, msg_size=4)
    assert w4.lower_bound == 4 * w1.lower_bound


def test_bottleneck_edges_attain_lb():
    t = Topology(slice=(4, 8))
    r = Router(t)
    w = AllToAll(t, r, msg_size=1)
    for e in w.bottleneck_edges():
        assert w.link_load[e] == w.lower_bound
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_model.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement the model**

`twisted_analysis/model/flow.py`:
```python
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from functools import cached_property

from twisted_analysis.topology import Topology, Router, Node, Path, DirectedLink


@dataclass(frozen=True)
class Flow:
    src: Node
    dst: Node
    size: int  # units of payload


@dataclass(frozen=True)
class AllToAll:
    topology: Topology
    router: Router
    msg_size: int = 1

    @cached_property
    def flows(self) -> tuple[Flow, ...]:
        return tuple(
            Flow(s, d, self.msg_size)
            for s in self.topology.nodes()
            for d in self.topology.nodes()
            if s != d
        )

    @cached_property
    def _path_map(self) -> dict[Flow, Path]:
        return {f: self.router.path(f.src, f.dst) for f in self.flows}

    def path(self, flow: Flow) -> Path:
        return self._path_map[flow]

    @cached_property
    def link_load(self) -> dict[DirectedLink, int]:
        c: Counter[DirectedLink] = Counter()
        for f in self.flows:
            for e in self._path_map[f]:
                c[e] += f.size
        return dict(c)

    @cached_property
    def lower_bound(self) -> int:
        if not self.link_load:
            return 0
        return max(self.link_load.values())

    def bottleneck_edges(self) -> list[DirectedLink]:
        lb = self.lower_bound
        return [e for e, load in self.link_load.items() if load == lb]
```

`twisted_analysis/model/__init__.py`:
```python
from twisted_analysis.model.flow import Flow, AllToAll

__all__ = ["Flow", "AllToAll"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_model.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add twisted_analysis/model/ tests/test_model.py
git commit -m "feat(model): AllToAll workload, link load, lower bound"
```

---

### Task 7: Schedule base interface

**Files:**
- Create: `twisted_analysis/schedules/__init__.py`
- Create: `twisted_analysis/schedules/base.py`
- Create: `tests/test_schedules_base.py`

- [ ] **Step 1: Write failing tests**

`tests/test_schedules_base.py`:
```python
from twisted_analysis.schedules.base import Injection, ScheduleResult
from twisted_analysis.model.flow import Flow


def test_injection_dataclass():
    f = Flow((0, 0), (0, 1), 1)
    inj = Injection(flow=f, start_step=0)
    assert inj.flow == f and inj.start_step == 0


def test_schedule_result_ratio():
    res = ScheduleResult(name="test", makespan=20, lower_bound=10)
    assert res.ratio == 2.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_schedules_base.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement base**

`twisted_analysis/schedules/base.py`:
```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol

from twisted_analysis.model.flow import Flow, AllToAll


@dataclass(frozen=True)
class Injection:
    """One unit of a flow is injected onto its first hop at `start_step`."""
    flow: Flow
    start_step: int
    priority: int = 0  # lower = higher priority at link contention; default FIFO


@dataclass(frozen=True)
class ScheduleResult:
    name: str
    makespan: int
    lower_bound: int
    per_step_busy: tuple[int, ...] = ()
    idle_steps_on_bottleneck: dict[tuple, int] = field(default_factory=dict)

    @property
    def ratio(self) -> float:
        return self.makespan / self.lower_bound


class Schedule(Protocol):
    name: str
    def emit(self, workload: AllToAll) -> list[Injection]: ...
```

`twisted_analysis/schedules/__init__.py`:
```python
from twisted_analysis.schedules.base import Injection, ScheduleResult, Schedule

__all__ = ["Injection", "ScheduleResult", "Schedule"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_schedules_base.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add twisted_analysis/schedules/ tests/test_schedules_base.py
git commit -m "feat(schedules): base Schedule protocol + Injection + ScheduleResult"
```

---

### Task 8: Simulator engine

**Files:**
- Create: `twisted_analysis/simulator/__init__.py`
- Create: `twisted_analysis/simulator/engine.py`
- Create: `tests/test_simulator.py`

- [ ] **Step 1: Write failing tests**

`tests/test_simulator.py`:
```python
from twisted_analysis.topology import Topology, Router
from twisted_analysis.model import AllToAll, Flow
from twisted_analysis.schedules.base import Injection
from twisted_analysis.simulator.engine import Simulator


def test_single_flow_one_hop():
    t = Topology(slice=(2, 4))
    r = Router(t)
    f = Flow((0, 0), (0, 1), 1)
    sim = Simulator(t, r, [f])
    sim.inject(Injection(flow=f, start_step=0))
    makespan = sim.run()
    assert makespan == 1


def test_single_flow_multi_hop():
    t = Topology(slice=(2, 4))
    r = Router(t)
    f = Flow((0, 0), (0, 3), 1)
    sim = Simulator(t, r, [f])
    sim.inject(Injection(flow=f, start_step=0))
    makespan = sim.run()
    assert makespan == 3  # 3 hops along row


def test_msg_size_2_pipelines():
    # Two units, same path of length 2; pipelined → 3 steps total (S&F).
    t = Topology(slice=(2, 4))
    r = Router(t)
    f = Flow((0, 0), (0, 2), 2)
    sim = Simulator(t, r, [f])
    sim.inject(Injection(flow=f, start_step=0))
    makespan = sim.run()
    assert makespan == 3


def test_two_flows_contend_on_first_hop():
    # Both flows start at (0, 0) and must use the same first link.
    # Second flow waits one step.
    t = Topology(slice=(2, 4))
    r = Router(t)
    f1 = Flow((0, 0), (0, 1), 1)
    f2 = Flow((0, 0), (0, 1), 1)
    sim = Simulator(t, r, [f1, f2])
    sim.inject(Injection(flow=f1, start_step=0, priority=0))
    sim.inject(Injection(flow=f2, start_step=0, priority=1))
    makespan = sim.run()
    assert makespan == 2


def test_makespan_at_least_lower_bound():
    t = Topology(slice=(2, 4))
    r = Router(t)
    w = AllToAll(t, r, msg_size=1)
    sim = Simulator(t, r, list(w.flows))
    for f in w.flows:
        sim.inject(Injection(flow=f, start_step=0))
    makespan = sim.run()
    assert makespan >= w.lower_bound
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_simulator.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement simulator**

`twisted_analysis/simulator/engine.py`:
```python
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from itertools import count

from twisted_analysis.topology import Topology, Router, DirectedLink
from twisted_analysis.model.flow import Flow
from twisted_analysis.schedules.base import Injection


@dataclass
class _Unit:
    """One unit of payload from one flow."""
    flow: Flow
    path: tuple[DirectedLink, ...]
    next_hop_idx: int = 0
    priority: int = 0
    seq: int = 0  # tie-break for deterministic ordering

    @property
    def at_link(self) -> DirectedLink:
        return self.path[self.next_hop_idx]

    @property
    def delivered(self) -> bool:
        return self.next_hop_idx >= len(self.path)


class Simulator:
    """Step-synchronous, store-and-forward, capacity-1 simulator."""

    def __init__(self, topology: Topology, router: Router, flows: list[Flow]):
        self.topology = topology
        self.router = router
        self.flows = flows
        self.path_map = {f: router.path(f.src, f.dst) for f in flows}
        self.link_queue: dict[DirectedLink, deque[_Unit]] = {}
        self.pending_injections: list[Injection] = []
        self.units: list[_Unit] = []
        self._seq = count()
        self.busy_per_step: list[int] = []
        self.delivered_count = 0
        self.total_units = sum(f.size for f in flows)

    def inject(self, injection: Injection) -> None:
        self.pending_injections.append(injection)

    def _enqueue_injections(self, step: int) -> None:
        ready = [i for i in self.pending_injections if i.start_step == step]
        self.pending_injections = [i for i in self.pending_injections if i.start_step != step]
        for inj in ready:
            path = self.path_map[inj.flow]
            if not path:  # self-loop, instantly "delivered"
                self.delivered_count += inj.flow.size
                continue
            for _ in range(inj.flow.size):
                u = _Unit(
                    flow=inj.flow,
                    path=path,
                    priority=inj.priority,
                    seq=next(self._seq),
                )
                self.units.append(u)
                self.link_queue.setdefault(path[0], deque()).append(u)

    def run(self, max_steps: int = 1_000_000) -> int:
        step = 0
        while self.delivered_count < self.total_units and step < max_steps:
            self._enqueue_injections(step)
            busy = self._step()
            self.busy_per_step.append(busy)
            step += 1
        if self.delivered_count < self.total_units:
            raise RuntimeError(f"sim did not drain in {max_steps} steps")
        return step

    def _step(self) -> int:
        """Each link picks one unit; chosen units advance one hop. Return # active links."""
        # Selection: pick (priority, seq) minimum per link.
        chosen: list[tuple[DirectedLink, _Unit]] = []
        for link, q in self.link_queue.items():
            if not q:
                continue
            best = min(q, key=lambda u: (u.priority, u.seq))
            q.remove(best)
            chosen.append((link, best))
        # Advance.
        for link, u in chosen:
            u.next_hop_idx += 1
            if u.delivered:
                self.delivered_count += 1
            else:
                self.link_queue.setdefault(u.at_link, deque()).append(u)
        # Cleanup empty queues to keep iteration sane.
        self.link_queue = {k: v for k, v in self.link_queue.items() if v}
        return len(chosen)
```

`twisted_analysis/simulator/__init__.py`:
```python
from twisted_analysis.simulator.engine import Simulator

__all__ = ["Simulator"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_simulator.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add twisted_analysis/simulator/ tests/test_simulator.py
git commit -m "feat(simulator): step-sync store-and-forward engine"
```

---

### Task 9: Round-robin (Latin-square) schedule

**Files:**
- Create: `twisted_analysis/schedules/round_robin.py`
- Modify: `twisted_analysis/schedules/__init__.py`
- Create: `tests/test_round_robin.py`

- [ ] **Step 1: Write failing tests**

`tests/test_round_robin.py`:
```python
from twisted_analysis.topology import Topology, Router
from twisted_analysis.model import AllToAll
from twisted_analysis.schedules.round_robin import RoundRobinSchedule
from twisted_analysis.simulator import Simulator


def test_round_robin_emits_n_minus_1_phases_worth():
    t = Topology(slice=(2, 4))
    r = Router(t)
    w = AllToAll(t, r, msg_size=1)
    sched = RoundRobinSchedule()
    injections = sched.emit(w)
    # N*(N-1) = 56 flows, all injected (each exactly once)
    assert len(injections) == 56


def test_round_robin_makespan_at_least_lb():
    t = Topology(slice=(2, 4))
    r = Router(t)
    w = AllToAll(t, r, msg_size=1)
    sched = RoundRobinSchedule()
    sim = Simulator(t, r, list(w.flows))
    for inj in sched.emit(w):
        sim.inject(inj)
    makespan = sim.run()
    assert makespan >= w.lower_bound


def test_round_robin_phases_dont_overlap():
    # Phase r's flows all share the same start_step.
    t = Topology(slice=(2, 4))
    r = Router(t)
    w = AllToAll(t, r, msg_size=1)
    sched = RoundRobinSchedule()
    injs = sched.emit(w)
    by_start = {}
    for inj in injs:
        by_start.setdefault(inj.start_step, []).append(inj)
    # 7 phases (N-1 = 7), each with N=8 flows
    assert len(by_start) == 7
    assert all(len(v) == 8 for v in by_start.values())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_round_robin.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement**

`twisted_analysis/schedules/round_robin.py`:
```python
from __future__ import annotations
from dataclasses import dataclass

from twisted_analysis.model.flow import Flow, AllToAll
from twisted_analysis.schedules.base import Injection, Schedule
from twisted_analysis.simulator.engine import Simulator
from twisted_analysis.topology import Topology, Router


@dataclass
class RoundRobinSchedule:
    """Latin-square AllToAll: in phase r, node i sends to (i+r) mod N.

    Phases run back-to-back; each phase starts when the previous one drains
    (computed by a dry-run simulation per phase).
    """
    name: str = "round_robin"

    def emit(self, workload: AllToAll) -> list[Injection]:
        t = workload.topology
        r = workload.router
        nodes = list(t.nodes())
        N = len(nodes)
        # Flat id for each node based on iteration order
        idx_of = {n: i for i, n in enumerate(nodes)}
        injections: list[Injection] = []
        phase_start = 0
        for phase_r in range(1, N):
            phase_flows: list[Flow] = []
            for src_node in nodes:
                dst_node = nodes[(idx_of[src_node] + phase_r) % N]
                # Find the matching workload flow
                phase_flows.append(Flow(src_node, dst_node, workload.msg_size))
            for f in phase_flows:
                injections.append(Injection(flow=f, start_step=phase_start))
            # Dry-run to compute when this phase drains
            sim = Simulator(t, r, phase_flows)
            for f in phase_flows:
                sim.inject(Injection(flow=f, start_step=0))
            phase_makespan = sim.run()
            phase_start += phase_makespan
        return injections
```

Update `twisted_analysis/schedules/__init__.py`:
```python
from twisted_analysis.schedules.base import Injection, ScheduleResult, Schedule
from twisted_analysis.schedules.round_robin import RoundRobinSchedule

__all__ = ["Injection", "ScheduleResult", "Schedule", "RoundRobinSchedule"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_round_robin.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add twisted_analysis/schedules/round_robin.py twisted_analysis/schedules/__init__.py tests/test_round_robin.py
git commit -m "feat(schedules): round-robin (Latin-square) AllToAll schedule"
```

---

### Task 10: Dimension-ordered phased schedule

**Files:**
- Create: `twisted_analysis/schedules/dim_phased.py`
- Modify: `twisted_analysis/schedules/__init__.py`
- Create: `tests/test_dim_phased.py`

- [ ] **Step 1: Write failing tests**

`tests/test_dim_phased.py`:
```python
from twisted_analysis.topology import Topology, Router
from twisted_analysis.model import AllToAll
from twisted_analysis.schedules.dim_phased import DimPhasedSchedule
from twisted_analysis.simulator import Simulator


def test_2x4_dim_phased_two_phases():
    t = Topology(slice=(2, 4))
    r = Router(t)
    w = AllToAll(t, r, msg_size=1)
    sched = DimPhasedSchedule()
    injections = sched.emit(w)
    starts = sorted({inj.start_step for inj in injections})
    # Phase 1 (along longer dim) at step 0, Phase 2 (along shorter dim) at later step
    assert len(starts) == 2 and starts[0] == 0


def test_2x4_dim_phased_makespan_at_least_lb():
    # DimPhased covers only the one-dim-diff subset of pairs, so we build the
    # simulator from the injected flow set (not the full AllToAll workload).
    t = Topology(slice=(2, 4))
    r = Router(t)
    w = AllToAll(t, r, msg_size=1)
    sched = DimPhasedSchedule()
    injs = sched.emit(w)
    flows = list({inj.flow for inj in injs})
    sim = Simulator(t, r, flows)
    for inj in injs:
        sim.inject(inj)
    assert sim.run() >= 1  # at least one step; partial-coverage workload


def test_4x4x8_dim_phased_three_phases():
    t = Topology(slice=(4, 4, 8))
    r = Router(t)
    w = AllToAll(t, r, msg_size=1)
    sched = DimPhasedSchedule()
    injections = sched.emit(w)
    starts = sorted({inj.start_step for inj in injections})
    assert len(starts) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_dim_phased.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement**

`twisted_analysis/schedules/dim_phased.py`:
```python
from __future__ import annotations
from dataclasses import dataclass

from twisted_analysis.model.flow import Flow, AllToAll
from twisted_analysis.schedules.base import Injection, Schedule
from twisted_analysis.simulator.engine import Simulator


@dataclass
class DimPhasedSchedule:
    """Dimension-ordered phased AllToAll: one phase per dim, largest-dim first.

    Phase d's flows: every (src, dst) pair that differs only in dim d.
    Each phase uses only dim-d links; phases don't contend.
    """
    name: str = "dim_phased"

    def emit(self, workload: AllToAll) -> list[Injection]:
        t = workload.topology
        r = workload.router
        # Phase ordering: largest dim first (default DOR order)
        dim_order = sorted(range(t.ndim), key=lambda d: -t.slice[d])
        injections: list[Injection] = []
        phase_start = 0
        for d in dim_order:
            phase_flows: list[Flow] = []
            for src in t.nodes():
                for dst in t.nodes():
                    if src == dst:
                        continue
                    # Only pairs that differ exactly in dim d
                    if all(src[i] == dst[i] for i in range(t.ndim) if i != d) \
                            and src[d] != dst[d]:
                        phase_flows.append(Flow(src, dst, workload.msg_size))
            for f in phase_flows:
                injections.append(Injection(flow=f, start_step=phase_start))
            # Dry-run for phase makespan
            sim = Simulator(t, r, phase_flows)
            for f in phase_flows:
                sim.inject(Injection(flow=f, start_step=0))
            phase_makespan = sim.run()
            phase_start += phase_makespan
        return injections
```

Update `twisted_analysis/schedules/__init__.py`:
```python
from twisted_analysis.schedules.base import Injection, ScheduleResult, Schedule
from twisted_analysis.schedules.round_robin import RoundRobinSchedule
from twisted_analysis.schedules.dim_phased import DimPhasedSchedule

__all__ = [
    "Injection", "ScheduleResult", "Schedule",
    "RoundRobinSchedule", "DimPhasedSchedule",
]
```

> **IMPORTANT — coverage caveat.** This `DimPhasedSchedule` is a *partial-coverage* schedule: it only emits flows for `(src, dst)` pairs that differ in *exactly one* dimension. It does **not** complete a full AllToAll. We use it as a diagnostic schedule that isolates per-dim link load. Two implications you MUST honor downstream:
>
> 1. When running this schedule through the simulator, build the simulator from the schedule's *emitted* flows, not from `w.flows`. The CLI in Task 15 follows this rule.
> 2. Comparing `M_DimPhased` to `LB(full AllToAll)` is not directly meaningful. In `docs/results.md` and `docs/schedules.md`, report this caveat explicitly and report `M_DimPhased` against `LB(subset)` — the max link load over only the emitted flows' paths.
>
> A full multi-stage transpose-style DimPhased schedule is out of scope for v1 and listed in `docs/schedules.md` as future work.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dim_phased.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add twisted_analysis/schedules/dim_phased.py twisted_analysis/schedules/__init__.py tests/test_dim_phased.py
git commit -m "feat(schedules): dimension-ordered phased AllToAll (one-dim-diff pairs)"
```

---

### Task 11: ILP formulation

**Files:**
- Create: `twisted_analysis/lp/__init__.py`
- Create: `twisted_analysis/lp/ilp.py`
- Create: `tests/test_ilp.py`

- [ ] **Step 1: Write failing tests**

`tests/test_ilp.py`:
```python
import pytest

from twisted_analysis.topology import Topology, Router
from twisted_analysis.model import AllToAll, Flow
from twisted_analysis.lp.ilp import solve_makespan


def test_single_flow_optimum_equals_path_length():
    t = Topology(slice=(2, 4))
    r = Router(t)
    f = Flow((0, 0), (0, 2), 1)
    workload = AllToAll(t, r, msg_size=1)
    # Override flows to a single one for this test
    workload_single = type(workload)(t, r, msg_size=1)
    # Use a tiny custom solve
    m_opt, _ = solve_makespan(t, r, [f], T_upper=8)
    assert m_opt == 2  # path length


def test_two_contending_flows_optimum_is_2():
    t = Topology(slice=(2, 4))
    r = Router(t)
    f1 = Flow((0, 0), (0, 1), 1)
    f2 = Flow((0, 0), (0, 1), 1)
    m_opt, _ = solve_makespan(t, r, [f1, f2], T_upper=8)
    assert m_opt == 2


def test_optimum_ge_lower_bound_2x4():
    t = Topology(slice=(2, 4))
    r = Router(t)
    w = AllToAll(t, r, msg_size=1)
    # Bound T_upper generously: LB * 4
    m_opt, _ = solve_makespan(t, r, list(w.flows), T_upper=4 * w.lower_bound)
    assert m_opt >= w.lower_bound
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ilp.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement ILP**

`twisted_analysis/lp/ilp.py`:
```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

import pulp

from twisted_analysis.topology import Topology, Router, DirectedLink
from twisted_analysis.model.flow import Flow


@dataclass
class UnitPath:
    """One unit of payload with its own path. Multiple units may share a flow."""
    unit_id: int
    flow: Flow
    path: tuple[DirectedLink, ...]


def _unroll_units(router: Router, flows: Iterable[Flow]) -> list[UnitPath]:
    units: list[UnitPath] = []
    uid = 0
    for f in flows:
        p = router.path(f.src, f.dst)
        for _ in range(f.size):
            units.append(UnitPath(uid, f, p))
            uid += 1
    return units


def _solve_feasibility(
    units: list[UnitPath], T: int, msg_solver: pulp.LpSolver
) -> dict[tuple[int, int, int], float] | None:
    """Returns assignment x[unit, i, t] -> {0, 1} if feasible, else None."""
    prob = pulp.LpProblem("twisted_alltoall", pulp.LpMinimize)
    x: dict[tuple[int, int, int], pulp.LpVariable] = {}
    for u in units:
        for i in range(len(u.path)):
            for t in range(T):
                x[(u.unit_id, i, t)] = pulp.LpVariable(
                    f"x_{u.unit_id}_{i}_{t}", cat=pulp.LpBinary
                )
    # Per-hop fire-once
    for u in units:
        for i in range(len(u.path)):
            prob += pulp.lpSum(x[(u.unit_id, i, t)] for t in range(T)) == 1
    # Causal order
    for u in units:
        for i in range(len(u.path) - 1):
            for s in range(T):
                prob += (
                    pulp.lpSum(x[(u.unit_id, i + 1, t)] for t in range(s + 1))
                    <= pulp.lpSum(x[(u.unit_id, i, t)] for t in range(s))
                )
    # Link capacity
    edge_to_uses: dict[DirectedLink, list[tuple[int, int]]] = {}
    for u in units:
        for i, e in enumerate(u.path):
            edge_to_uses.setdefault(e, []).append((u.unit_id, i))
    for e, uses in edge_to_uses.items():
        for t in range(T):
            prob += pulp.lpSum(x[(uid, i, t)] for uid, i in uses) <= 1
    # Trivial objective (feasibility)
    prob += 0
    prob.solve(msg_solver)
    if pulp.LpStatus[prob.status] != "Optimal":
        return None
    return {k: pulp.value(v) for k, v in x.items()}


def solve_makespan(
    topology: Topology,
    router: Router,
    flows: list[Flow],
    T_upper: int,
    solver_name: str = "PULP_CBC_CMD",
) -> tuple[int, dict[tuple[int, int, int], float]]:
    """Binary search on makespan T. Returns (optimal_T, optimal_assignment).

    Lower bound for the search = max link load.
    """
    units = _unroll_units(router, flows)
    if not units:
        return 0, {}
    # Compute LB
    from collections import Counter
    c: Counter[DirectedLink] = Counter()
    for u in units:
        for e in u.path:
            c[e] += 1
    lb = max(c.values()) if c else 0
    solver = pulp.getSolver(solver_name, msg=False)
    lo, hi = lb, T_upper
    best_assignment: dict[tuple[int, int, int], float] = {}
    # First confirm hi is feasible (else widen)
    while True:
        a = _solve_feasibility(units, hi, solver)
        if a is not None:
            best_assignment = a
            break
        hi *= 2
        if hi > 1_000_000:
            raise RuntimeError("ILP T_upper grew past 1e6; check formulation")
    # Binary search
    while lo < hi:
        mid = (lo + hi) // 2
        a = _solve_feasibility(units, mid, solver)
        if a is not None:
            hi = mid
            best_assignment = a
        else:
            lo = mid + 1
    return lo, best_assignment
```

`twisted_analysis/lp/__init__.py`:
```python
from twisted_analysis.lp.ilp import solve_makespan, UnitPath

__all__ = ["solve_makespan", "UnitPath"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ilp.py -v`
Expected: 3 passed. (The 2x4 AllToAll test may take a few seconds.)

- [ ] **Step 5: Commit**

```bash
git add twisted_analysis/lp/ tests/test_ilp.py
git commit -m "feat(lp): time-indexed ILP with binary-search makespan"
```

---

### Task 12: LP relaxation

**Files:**
- Create: `twisted_analysis/lp/relaxation.py`
- Modify: `twisted_analysis/lp/__init__.py`
- Create: `tests/test_lp_relaxation.py`

- [ ] **Step 1: Write failing tests**

`tests/test_lp_relaxation.py`:
```python
from twisted_analysis.topology import Topology, Router
from twisted_analysis.model import AllToAll
from twisted_analysis.lp.relaxation import lp_relax_lower_bound


def test_lp_relax_is_ge_link_lb_2x4():
    t = Topology(slice=(2, 4))
    r = Router(t)
    w = AllToAll(t, r, msg_size=1)
    m_lp = lp_relax_lower_bound(t, r, list(w.flows), T_upper=w.lower_bound * 4)
    assert m_lp >= w.lower_bound - 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_lp_relaxation.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement relaxation**

`twisted_analysis/lp/relaxation.py`:
```python
from __future__ import annotations

import pulp

from twisted_analysis.topology import Topology, Router, DirectedLink
from twisted_analysis.model.flow import Flow
from twisted_analysis.lp.ilp import _unroll_units


def lp_relax_lower_bound(
    topology: Topology,
    router: Router,
    flows: list[Flow],
    T_upper: int,
    solver_name: str = "PULP_CBC_CMD",
) -> float:
    """LP relaxation: minimize makespan T s.t. flow-conservation + capacity (fractional).

    Returns the optimal T_LP, which satisfies LB <= T_LP <= ILP optimum.
    """
    units = _unroll_units(router, flows)
    if not units:
        return 0.0
    T = T_upper
    prob = pulp.LpProblem("twisted_alltoall_relax", pulp.LpMinimize)
    M = pulp.LpVariable("M", lowBound=0)
    x: dict[tuple[int, int, int], pulp.LpVariable] = {}
    for u in units:
        for i in range(len(u.path)):
            for t in range(T):
                x[(u.unit_id, i, t)] = pulp.LpVariable(
                    f"x_{u.unit_id}_{i}_{t}", lowBound=0, upBound=1
                )
    # Per-hop fire-once
    for u in units:
        for i in range(len(u.path)):
            prob += pulp.lpSum(x[(u.unit_id, i, t)] for t in range(T)) == 1
    # Causal order
    for u in units:
        for i in range(len(u.path) - 1):
            for s in range(T):
                prob += (
                    pulp.lpSum(x[(u.unit_id, i + 1, t)] for t in range(s + 1))
                    <= pulp.lpSum(x[(u.unit_id, i, t)] for t in range(s))
                )
    # Link capacity
    edge_to_uses: dict[DirectedLink, list[tuple[int, int]]] = {}
    for u in units:
        for i, e in enumerate(u.path):
            edge_to_uses.setdefault(e, []).append((u.unit_id, i))
    for e, uses in edge_to_uses.items():
        for t in range(T):
            prob += pulp.lpSum(x[(uid, i, t)] for uid, i in uses) <= 1
    # Makespan: last-hop firing time bounded by M
    for u in units:
        last = len(u.path) - 1
        prob += pulp.lpSum((t + 1) * x[(u.unit_id, last, t)] for t in range(T)) <= M
    prob += M
    prob.solve(pulp.getSolver(solver_name, msg=False))
    return float(pulp.value(M))
```

Update `twisted_analysis/lp/__init__.py`:
```python
from twisted_analysis.lp.ilp import solve_makespan, UnitPath
from twisted_analysis.lp.relaxation import lp_relax_lower_bound

__all__ = ["solve_makespan", "UnitPath", "lp_relax_lower_bound"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_lp_relaxation.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add twisted_analysis/lp/relaxation.py twisted_analysis/lp/__init__.py tests/test_lp_relaxation.py
git commit -m "feat(lp): LP relaxation lower bound"
```

---

### Task 13: LP-extracted schedule + simulator replay validation

**Files:**
- Create: `twisted_analysis/schedules/lp_optimal.py`
- Modify: `twisted_analysis/schedules/__init__.py`
- Create: `tests/test_lp_replay.py`

- [ ] **Step 1: Write failing tests**

`tests/test_lp_replay.py`:
```python
from twisted_analysis.topology import Topology, Router
from twisted_analysis.model import AllToAll
from twisted_analysis.lp.ilp import solve_makespan
from twisted_analysis.schedules.lp_optimal import lp_assignment_to_injections
from twisted_analysis.simulator import Simulator


def test_lp_assignment_replays_to_same_makespan_2x4():
    t = Topology(slice=(2, 4))
    r = Router(t)
    w = AllToAll(t, r, msg_size=1)
    m_opt, assignment = solve_makespan(t, r, list(w.flows), T_upper=w.lower_bound * 4)
    flows_list = list(w.flows)
    injs = lp_assignment_to_injections(flows_list, r, assignment)
    sim = Simulator(t, r, flows_list)
    for inj in injs:
        sim.inject(inj)
    sim_makespan = sim.run()
    # Simulator should reproduce the LP makespan exactly under the LP-derived priorities.
    assert sim_makespan == m_opt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_lp_replay.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement assignment-to-injections**

`twisted_analysis/schedules/lp_optimal.py`:
```python
from __future__ import annotations
from collections import defaultdict

from twisted_analysis.model.flow import Flow
from twisted_analysis.schedules.base import Injection
from twisted_analysis.topology import Router


def lp_assignment_to_injections(
    flows: list[Flow],
    router: Router,
    assignment: dict[tuple[int, int, int], float],
) -> list[Injection]:
    """Translate the LP `x[unit, i, t]` assignment into Injection records.

    The LP encodes per-unit per-hop firing times. For our purposes we only need
    each unit's first-hop firing time (= start_step) and a per-link priority
    derived from the LP's fire-step at each hop.
    """
    # Build unit_id -> Flow and Flow -> [unit_ids]
    unit_to_flow: dict[int, Flow] = {}
    uid = 0
    flow_to_uids: dict[Flow, list[int]] = defaultdict(list)
    for f in flows:
        for _ in range(f.size):
            unit_to_flow[uid] = f
            flow_to_uids[f].append(uid)
            uid += 1
    # Extract first-hop firing step per unit
    first_step: dict[int, int] = {}
    for (uid, i, t), val in assignment.items():
        if i == 0 and val > 0.5:
            first_step[uid] = t
    injections: list[Injection] = []
    for unit_id, f in unit_to_flow.items():
        start = first_step.get(unit_id, 0)
        injections.append(Injection(flow=f, start_step=start, priority=unit_id))
    return injections
```

Update `twisted_analysis/schedules/__init__.py`:
```python
from twisted_analysis.schedules.base import Injection, ScheduleResult, Schedule
from twisted_analysis.schedules.round_robin import RoundRobinSchedule
from twisted_analysis.schedules.dim_phased import DimPhasedSchedule
from twisted_analysis.schedules.lp_optimal import lp_assignment_to_injections

__all__ = [
    "Injection", "ScheduleResult", "Schedule",
    "RoundRobinSchedule", "DimPhasedSchedule",
    "lp_assignment_to_injections",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_lp_replay.py -v`
Expected: 1 passed.

> **Caveat.** If the LP places multiple units on the same link at the same step (it shouldn't, by capacity constraint), the per-unit `priority=unit_id` assignment provides a stable tie-break for the simulator. If replay diverges, the LP's per-step assignment can be fed to the simulator directly as a "forced selection" override; defer that to a follow-up if needed.

- [ ] **Step 5: Commit**

```bash
git add twisted_analysis/schedules/lp_optimal.py twisted_analysis/schedules/__init__.py tests/test_lp_replay.py
git commit -m "feat(schedules): LP-extracted optimal schedule + replay validation"
```

---

### Task 14: Instrumentation (idle-trace, Gantt log)

**Files:**
- Modify: `twisted_analysis/simulator/engine.py`
- Create: `twisted_analysis/simulator/instrumentation.py`
- Create: `tests/test_instrumentation.py`

- [ ] **Step 1: Write failing tests**

`tests/test_instrumentation.py`:
```python
from twisted_analysis.topology import Topology, Router
from twisted_analysis.model import AllToAll
from twisted_analysis.schedules.round_robin import RoundRobinSchedule
from twisted_analysis.simulator import Simulator
from twisted_analysis.simulator.instrumentation import collect_idle_trace, gantt_log


def test_idle_trace_returns_dict_keyed_by_edge():
    t = Topology(slice=(2, 4))
    r = Router(t)
    w = AllToAll(t, r, msg_size=1)
    sched = RoundRobinSchedule()
    sim = Simulator(t, r, list(w.flows), record_history=True)
    for inj in sched.emit(w):
        sim.inject(inj)
    sim.run()
    trace = collect_idle_trace(sim, bottleneck_edges=w.bottleneck_edges())
    assert all(isinstance(v, int) and v >= 0 for v in trace.values())


def test_gantt_log_has_one_row_per_unit_per_hop():
    t = Topology(slice=(2, 4))
    r = Router(t)
    w = AllToAll(t, r, msg_size=1)
    sched = RoundRobinSchedule()
    sim = Simulator(t, r, list(w.flows), record_history=True)
    for inj in sched.emit(w):
        sim.inject(inj)
    sim.run()
    log = gantt_log(sim)
    expected_rows = sum(len(sim.path_map[f]) for f in w.flows)
    assert len(log) == expected_rows
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_instrumentation.py -v`
Expected: TypeError on `record_history` or ModuleNotFoundError.

- [ ] **Step 3: Extend Simulator to record history**

In `twisted_analysis/simulator/engine.py`, change `__init__` and `_step`:

```python
class Simulator:
    def __init__(
        self,
        topology: Topology,
        router: Router,
        flows: list[Flow],
        record_history: bool = False,
    ):
        self.topology = topology
        self.router = router
        self.flows = flows
        self.path_map = {f: router.path(f.src, f.dst) for f in flows}
        self.link_queue: dict[DirectedLink, deque[_Unit]] = {}
        self.pending_injections: list[Injection] = []
        self.units: list[_Unit] = []
        self._seq = count()
        self.busy_per_step: list[int] = []
        self.delivered_count = 0
        self.total_units = sum(f.size for f in flows)
        self.record_history = record_history
        # When history enabled, store (step, link, flow, hop_index) per transmission
        self.history: list[tuple[int, DirectedLink, Flow, int]] = []
        # Per-link busy mask at each step (for idle-trace) — only populated if record_history
        self.link_busy: dict[DirectedLink, list[bool]] = {}

    def run(self, max_steps: int = 1_000_000) -> int:
        step = 0
        while self.delivered_count < self.total_units and step < max_steps:
            self._enqueue_injections(step)
            busy_links = self._step(step)
            self.busy_per_step.append(len(busy_links))
            if self.record_history:
                for e in self.topology.directed_links():
                    self.link_busy.setdefault(e, []).append(e in busy_links)
            step += 1
        if self.delivered_count < self.total_units:
            raise RuntimeError(f"sim did not drain in {max_steps} steps")
        return step

    def _step(self, step: int) -> set[DirectedLink]:
        chosen: list[tuple[DirectedLink, _Unit]] = []
        for link, q in self.link_queue.items():
            if not q:
                continue
            best = min(q, key=lambda u: (u.priority, u.seq))
            q.remove(best)
            chosen.append((link, best))
        busy_links = {link for link, _ in chosen}
        for link, u in chosen:
            if self.record_history:
                self.history.append((step, link, u.flow, u.next_hop_idx))
            u.next_hop_idx += 1
            if u.delivered:
                self.delivered_count += 1
            else:
                self.link_queue.setdefault(u.at_link, deque()).append(u)
        self.link_queue = {k: v for k, v in self.link_queue.items() if v}
        return busy_links
```

- [ ] **Step 4: Implement instrumentation helpers**

`twisted_analysis/simulator/instrumentation.py`:
```python
from __future__ import annotations
import csv
from pathlib import Path

from twisted_analysis.topology import DirectedLink
from twisted_analysis.simulator.engine import Simulator


def collect_idle_trace(
    sim: Simulator, bottleneck_edges: list[DirectedLink]
) -> dict[DirectedLink, int]:
    """For each bottleneck edge, count steps where the edge was idle but
    other links were still working (i.e. work was not finished)."""
    if not sim.record_history:
        raise ValueError("Simulator must be constructed with record_history=True")
    result: dict[DirectedLink, int] = {}
    total_steps = len(sim.busy_per_step)
    for e in bottleneck_edges:
        busy = sim.link_busy.get(e, [False] * total_steps)
        idle = sum(1 for b in busy if not b)
        result[e] = idle
    return result


def gantt_log(sim: Simulator) -> list[tuple[int, DirectedLink, str, int]]:
    """Return Gantt rows: (step, link, flow_repr, hop_index)."""
    return [(t, link, repr(flow), hop) for t, link, flow, hop in sim.history]


def write_gantt_csv(sim: Simulator, path: Path) -> None:
    rows = gantt_log(sim)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "link", "flow", "hop_index"])
        for r in rows:
            w.writerow(r)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_instrumentation.py -v`
Expected: 2 passed.

- [ ] **Step 6: Run full test suite to make sure nothing regressed**

Run: `uv run pytest -v`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add twisted_analysis/simulator/ tests/test_instrumentation.py
git commit -m "feat(simulator): per-link history + idle-trace + Gantt CSV"
```

---

### Task 15: CLI + experiment YAML

**Files:**
- Create: `twisted_analysis/cli.py`
- Create: `experiments/2x4_rr.yaml`
- Create: `experiments/2x4_dim_phased.yaml`
- Create: `experiments/2x4_ilp.yaml`
- Create: `experiments/4x8_rr.yaml`
- Create: `experiments/4x8_dim_phased.yaml`
- Create: `experiments/4x4x8_rr.yaml`
- Create: `experiments/4x4x8_dim_phased.yaml`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Define the YAML schema**

Each experiment file:
```yaml
name: 2x4_rr
slice: [2, 4]
msg_size: 1
schedule: round_robin     # one of: round_robin, dim_phased, ilp_optimal
output_dir: results/2x4_rr
```

Create `experiments/2x4_rr.yaml` with that content (with `name: 2x4_rr`, `schedule: round_robin`).

Create matching files for the other six experiments, varying `slice`, `name`, `schedule`, `output_dir`.

- [ ] **Step 2: Write failing CLI test**

`tests/test_cli.py`:
```python
import subprocess
import sys
from pathlib import Path


def test_cli_runs_2x4_rr(tmp_path):
    # Run CLI as a module; expect a summary file in tmp_path
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"name: smoke\nslice: [2, 4]\nmsg_size: 1\nschedule: round_robin\n"
        f"output_dir: {tmp_path}/out\n"
    )
    res = subprocess.run(
        [sys.executable, "-m", "twisted_analysis.cli", "run", str(cfg)],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
    out_dir = tmp_path / "out"
    assert (out_dir / "summary.json").exists()
```

- [ ] **Step 3: Implement CLI**

`twisted_analysis/cli.py`:
```python
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import yaml

from twisted_analysis.topology import Topology, Router
from twisted_analysis.model import AllToAll
from twisted_analysis.schedules.round_robin import RoundRobinSchedule
from twisted_analysis.schedules.dim_phased import DimPhasedSchedule
from twisted_analysis.schedules.lp_optimal import lp_assignment_to_injections
from twisted_analysis.simulator import Simulator
from twisted_analysis.simulator.instrumentation import collect_idle_trace, write_gantt_csv


SCHEDULES = {
    "round_robin": RoundRobinSchedule(),
    "dim_phased": DimPhasedSchedule(),
}


def run_experiment(cfg: dict) -> dict:
    slice_ = tuple(cfg["slice"])
    msg_size = cfg.get("msg_size", 1)
    sched_name = cfg["schedule"]
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    t = Topology(slice=slice_)
    r = Router(t)
    w = AllToAll(t, r, msg_size=msg_size)

    if sched_name == "ilp_optimal":
        from twisted_analysis.lp.ilp import solve_makespan
        m_opt, assignment = solve_makespan(
            t, r, list(w.flows), T_upper=w.lower_bound * 4
        )
        injs = lp_assignment_to_injections(list(w.flows), r, assignment)
    else:
        sched = SCHEDULES[sched_name]
        injs = sched.emit(w)

    # Build the simulator from the actual flows the schedule will inject — this
    # supports partial-coverage schedules (e.g. DimPhased) without hanging.
    sim_flows = list({inj.flow for inj in injs})
    sim = Simulator(t, r, sim_flows, record_history=True)
    for inj in injs:
        sim.inject(inj)
    makespan = sim.run()

    idle = collect_idle_trace(sim, w.bottleneck_edges())
    write_gantt_csv(sim, out_dir / "gantt.csv")

    summary = {
        "name": cfg["name"],
        "slice": list(slice_),
        "msg_size": msg_size,
        "schedule": sched_name,
        "lower_bound": w.lower_bound,
        "makespan": makespan,
        "ratio": makespan / w.lower_bound if w.lower_bound else 0.0,
        "bottleneck_edges": [list(map(list, [e[0], e[1]])) + [e[2], e[3]]
                              for e in w.bottleneck_edges()],
        "idle_steps_on_bottleneck": {
            str(e): v for e, v in idle.items()
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="twisted_analysis")
    sub = p.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("config", type=Path)
    args = p.parse_args(argv)
    if args.cmd == "run":
        cfg = yaml.safe_load(args.config.read_text())
        summary = run_experiment(cfg)
        print(json.dumps(summary, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test**

Run: `uv run pytest tests/test_cli.py -v`
Expected: 1 passed.

- [ ] **Step 5: Spot-check a real experiment**

Run: `uv run python -m twisted_analysis.cli run experiments/2x4_rr.yaml`
Inspect `results/2x4_rr/summary.json`.

- [ ] **Step 6: Commit**

```bash
git add twisted_analysis/cli.py experiments/ tests/test_cli.py
git commit -m "feat(cli): YAML experiment driver with summary.json output"
```

---

### Task 16: Bisection-bandwidth lower bound

**Files:**
- Create: `twisted_analysis/model/bounds.py`
- Modify: `twisted_analysis/model/__init__.py`
- Create: `tests/test_bounds.py`

- [ ] **Step 1: Write failing tests**

`tests/test_bounds.py`:
```python
from twisted_analysis.topology import Topology, Router
from twisted_analysis.model import AllToAll
from twisted_analysis.model.bounds import bisection_bound


def test_bisection_bound_2x4_positive():
    t = Topology(slice=(2, 4))
    r = Router(t)
    w = AllToAll(t, r, msg_size=1)
    bb = bisection_bound(w)
    assert bb >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_bounds.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement bisection lower bound**

`twisted_analysis/model/bounds.py`:
```python
from __future__ import annotations

from twisted_analysis.model.flow import AllToAll


def bisection_bound(workload: AllToAll) -> int:
    """A relaxed bisection-bandwidth lower bound.

    For each "cut" splitting nodes into halves A and B, count the number of
    flows crossing the cut. A valid bisection bound is ceil(cross / cut_edges),
    which must hold for every cut. We evaluate only the natural coordinate-
    aligned half-cuts (one per dim: split along the median plane of each axis).
    The true bisection bound (over all 2^N partitions) is >= this much.
    """
    t = workload.topology
    nodes = list(t.nodes())
    if len(nodes) < 2:
        return 0
    best = 0
    for d in range(t.ndim):
        threshold = t.slice[d] // 2
        A = {n_ for n_ in nodes if n_[d] < threshold}
        cross_flows = sum(
            workload.msg_size
            for f in workload.flows
            if (f.src in A) != (f.dst in A)
        )
        cut_edges = sum(
            1 for u, v, _, _ in t.directed_links() if (u in A) != (v in A)
        )
        if cut_edges > 0:
            best = max(best, -(-cross_flows // cut_edges))  # ceil division
    return best
```

Update `twisted_analysis/model/__init__.py`:
```python
from twisted_analysis.model.flow import Flow, AllToAll
from twisted_analysis.model.bounds import bisection_bound

__all__ = ["Flow", "AllToAll", "bisection_bound"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_bounds.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add twisted_analysis/model/bounds.py twisted_analysis/model/__init__.py tests/test_bounds.py
git commit -m "feat(model): bisection-bandwidth lower bound (coordinate cuts)"
```

---

### Task 17: Visualization helpers

**Files:**
- Create: `twisted_analysis/viz/__init__.py`
- Create: `twisted_analysis/viz/load_histogram.py`
- Create: `twisted_analysis/viz/gantt.py`
- Create: `twisted_analysis/viz/heatmap.py`
- Create: `tests/test_viz.py`

- [ ] **Step 1: Write failing tests**

`tests/test_viz.py`:
```python
import matplotlib
matplotlib.use("Agg")

from pathlib import Path

from twisted_analysis.topology import Topology, Router
from twisted_analysis.model import AllToAll
from twisted_analysis.viz.load_histogram import plot_load_histogram


def test_load_histogram_writes_png(tmp_path: Path):
    t = Topology(slice=(2, 4))
    r = Router(t)
    w = AllToAll(t, r, msg_size=1)
    out = tmp_path / "hist.png"
    plot_load_histogram(w, out)
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_viz.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement load histogram**

`twisted_analysis/viz/load_histogram.py`:
```python
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt

from twisted_analysis.model.flow import AllToAll


def plot_load_histogram(workload: AllToAll, out_path: Path) -> None:
    loads = list(workload.link_load.values())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(loads, bins=range(min(loads), max(loads) + 2))
    ax.set_xlabel("Link load (flow-units)")
    ax.set_ylabel("# directed links")
    ax.set_title(f"Link-load distribution: slice={workload.topology.slice}, "
                  f"m={workload.msg_size}, LB={workload.lower_bound}")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
```

`twisted_analysis/viz/gantt.py`:
```python
"""Minimal Gantt plotter: one row per link, time on x-axis, colored bars per flow."""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt

from twisted_analysis.simulator.engine import Simulator


def plot_gantt(sim: Simulator, out_path: Path, max_links: int = 32) -> None:
    if not sim.record_history:
        raise ValueError("Simulator needs record_history=True")
    links = list(sim.link_busy.keys())[:max_links]
    fig, ax = plt.subplots(figsize=(10, max(2, len(links) * 0.25)))
    for i, e in enumerate(links):
        busy = sim.link_busy[e]
        for t, b in enumerate(busy):
            if b:
                ax.barh(i, 1, left=t, height=0.8, color="C0")
    ax.set_yticks(range(len(links)))
    ax.set_yticklabels([str(e) for e in links], fontsize=6)
    ax.set_xlabel("Step")
    ax.set_title("Link usage Gantt")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
```

`twisted_analysis/viz/heatmap.py`:
```python
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from twisted_analysis.simulator.engine import Simulator


def plot_link_utilization_heatmap(sim: Simulator, out_path: Path) -> None:
    if not sim.record_history:
        raise ValueError("Simulator needs record_history=True")
    links = list(sim.link_busy.keys())
    T = len(sim.busy_per_step)
    mat = np.zeros((len(links), T), dtype=int)
    for i, e in enumerate(links):
        mat[i, :] = sim.link_busy[e][:T]
    fig, ax = plt.subplots(figsize=(10, max(3, len(links) * 0.1)))
    ax.imshow(mat, aspect="auto", cmap="Greys", interpolation="nearest")
    ax.set_xlabel("Step")
    ax.set_ylabel("Directed link")
    ax.set_title("Per-step link utilization")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
```

`twisted_analysis/viz/__init__.py`:
```python
from twisted_analysis.viz.load_histogram import plot_load_histogram
from twisted_analysis.viz.gantt import plot_gantt
from twisted_analysis.viz.heatmap import plot_link_utilization_heatmap

__all__ = ["plot_load_histogram", "plot_gantt", "plot_link_utilization_heatmap"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_viz.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add twisted_analysis/viz/ tests/test_viz.py
git commit -m "feat(viz): load histogram, Gantt, link-utilization heatmap"
```

---

### Task 18: eval/run_all.sh

**Files:**
- Create: `eval/run_all.sh`

- [ ] **Step 1: Write the runner**

`eval/run_all.sh`:
```bash
#!/usr/bin/env bash
# Runs every YAML experiment and aggregates summaries into a single table.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATE=$(date +%Y-%m-%d)
RESULTS_DIR="$ROOT/results/$DATE"
mkdir -p "$RESULTS_DIR"

PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
    echo "Run 'uv venv && uv pip install -e \".[dev]\"' first." >&2
    exit 1
fi

for cfg in "$ROOT"/experiments/*.yaml; do
    name=$(basename "$cfg" .yaml)
    out="$RESULTS_DIR/$name"
    echo "=== $name ==="
    # Patch output_dir for this run
    tmpcfg=$(mktemp --suffix=.yaml)
    sed "s|^output_dir:.*|output_dir: $out|" "$cfg" > "$tmpcfg"
    "$PY" -m twisted_analysis.cli run "$tmpcfg" > "$out.log" 2>&1 || {
        echo "  FAILED — see $out.log"; continue;
    }
    rm "$tmpcfg"
done

# Aggregate summaries
"$PY" - <<EOF
import json, pathlib, csv
root = pathlib.Path("$RESULTS_DIR")
rows = []
for s in sorted(root.rglob("summary.json")):
    rows.append(json.loads(s.read_text()))
with (root / "headlines.csv").open("w", newline="") as f:
    w = csv.writer(f)
    if rows:
        w.writerow(list(rows[0].keys()))
        for r in rows:
            w.writerow([json.dumps(v) if isinstance(v, (list, dict)) else v
                         for v in r.values()])
print(f"Wrote {root}/headlines.csv with {len(rows)} rows")
EOF
```

- [ ] **Step 2: Make executable**

```bash
chmod +x eval/run_all.sh
```

- [ ] **Step 3: Smoke run**

Run: `bash eval/run_all.sh`
Expected: For each experiment, a `<name>/summary.json` under `results/<today>/`, and a `headlines.csv` aggregator at the top of that dir.

- [ ] **Step 4: Commit**

```bash
git add eval/run_all.sh
git commit -m "chore(eval): single-command runner aggregating all experiment summaries"
```

---

### Task 19: Documentation

**Files:**
- Create: `docs/algorithm.md`
- Create: `docs/topology.md`
- Create: `docs/schedules.md`
- Create: `docs/lp_formulation.md`
- Create: `docs/evaluation.md`
- Create: `docs/results.md`
- Modify: `README.md`

- [ ] **Step 1: Write `docs/algorithm.md`** (cost model + LB proof)

Content:
- Restate the spec's cost model (step-sync, S&F, capacity 1).
- Give the LB proof in full: pick any directed edge `e`; let `L = load(e)`; in any feasible schedule, `e` transmits at most one unit per step; total `L` units must cross `e`; therefore `makespan ≥ L`; taking max over `e` gives `LB`. QED.
- Worked example: on `(2,4)` with `m=1`, list the link loads for 2-3 specific edges; identify the bottleneck.
- Cross-link to `lp_formulation.md` for the ILP encoding of the same model.

- [ ] **Step 2: Write `docs/topology.md`** (neighbor function semantics)

Content:
- Restate the neighbor function (verbatim from `lattice.py`).
- Worked traces of wrap on `(2,4)` and `(4,4,8)` (use the design spec's tables).
- Diagram (ASCII or matplotlib-rendered) of the `(2, 4)` topology.
- Document the "twist orientation symmetry" property: `+R mod 2R == -R mod 2R` because the wrap shifts by the *full* `slice[dim]` magnitude regardless of direction.

- [ ] **Step 3: Write `docs/schedules.md`** (each schedule explained)

Content:
- Latin-square: phase structure, formula `dst = (src + r) mod N`, phase makespan accumulation.
- Dim-phased: per-dim phase definition; coverage caveat (only one-dim-diff pairs); pseudocode.
- LP-optimal: how the LP assignment becomes Injections + per-link priorities.

- [ ] **Step 4: Write `docs/lp_formulation.md`** (ILP details)

Content:
- Variables, constraints, objective verbatim from the design spec §5.
- Binary search strategy on `T`; LP relaxation as bound-tightener.
- Complexity discussion: variable count ≈ `(total hops) × T`.
- Solver invocation example.

- [ ] **Step 5: Write `docs/evaluation.md`** (how to reproduce)

Content:
- Experiment matrix (lift from spec §7).
- `bash eval/run_all.sh` walkthrough.
- How to add a new experiment (YAML schema explained).

- [ ] **Step 6: Write `docs/results.md`** (placeholder)

Content:
- Initially: "Run `bash eval/run_all.sh` and inspect `results/<date>/headlines.csv`."
- Document a few headline numbers manually after the first full run (do this after Task 20).

- [ ] **Step 7: Rewrite top-level `README.md`**

Replace the existing content with:

```markdown
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
```

- [ ] **Step 8: Commit docs**

```bash
git add docs/ README.md
git commit -m "docs: algorithm, topology, schedules, LP, evaluation, results + README"
```

---

### Task 20: Headline results + final review

**Files:**
- Modify: `docs/results.md`
- Create: `results/2026-05-12/` (the first canonical run)

- [ ] **Step 1: Run full experiment suite**

Run: `bash eval/run_all.sh`
Expected: All experiments succeed; `results/<today>/headlines.csv` is populated.

- [ ] **Step 2: Inspect headlines**

Open `results/<today>/headlines.csv`. For each `(slice, schedule)`, record `lower_bound`, `makespan`, `ratio`.

- [ ] **Step 3: Fill in `docs/results.md`**

Replace placeholder content with a markdown table of headline numbers, formatted like:

```markdown
| slice | schedule | LB | makespan | ratio |
|---|---|---|---|---|
| (2,4) | round_robin | <LB> | <M> | <r> |
| (2,4) | dim_phased | <LB> | <M> | <r> |
| (2,4) | ilp_optimal | <LB> | <M> | <r> |
| (4,8) | round_robin | ... | ... | ... |
| ... | ... | ... | ... | ... |
```

Add a short qualitative summary: which schedule is closest to LB? On which topology is the gap largest? Is `M_opt == LB` on any instance?

- [ ] **Step 4: Final lint + test**

Run: `uv run pytest -v`
Expected: full suite green.

- [ ] **Step 5: Commit**

```bash
git add docs/results.md results/2026-05-12/
git commit -m "docs(results): first canonical run headlines (2026-05-12)"
```

---

## Self-review (run BEFORE handing off for execution)

The plan author runs this; the executor doesn't need to.

1. **Spec coverage** — every spec section maps to at least one task:
   - §1 problem + cost model → Task 6 (LB) + docs Task 19.
   - §2 topology + routing → Tasks 2–5.
   - §3 flow + LB → Task 6.
   - §4 schedules → Tasks 7, 9, 10, 13.
   - §5 LP/ILP → Tasks 11, 12.
   - §6 simulator → Tasks 8, 14.
   - §7 evaluation → Tasks 15, 18, 20.
   - §8 project structure → Task 1.
   - §9 defaults/open questions → docs Task 19; bisection bound in Task 16.

2. **Placeholder scan** — no "TBD"; no "add appropriate handling"; concrete code in every step. The ASCII diagram in `docs/topology.md` Task 19 Step 2 is described but not literally drawn; that's intentional — the writer chooses the diagram.

3. **Type consistency** — `Flow`, `Injection`, `DirectedLink`, `Path`, `Topology`, `Router`, `AllToAll`, `Simulator`, `ScheduleResult` are used consistently across tasks. Same module paths used everywhere.

4. **Ablations** — DOR dim-order alt and m-sweep ablation are deferred to a follow-up plan (out of v1 scope but listed in spec §7). Add a note in `docs/evaluation.md` Task 19 that these are future work.

---

## Execution

This plan is ready for either:

- **Subagent-driven execution** (recommended): fresh subagent per task, review between tasks.
- **Inline execution**: this session executes tasks under `superpowers:executing-plans`.

Pick at handoff time.
