# Router → Scheduler → Kernel Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the TwistedAnalysis pipeline into three composable stages — Router emits a routing-table JSON; Scheduler consumes that JSON and emits a schedule JSON; the Pallas kernel generator orchestrates both stages, persists the intermediate artifacts, and emits the kernel source.

**Architecture:** Three on-disk artifacts in well-known directories: `fixtures/routing_table_<slice>_<router>.json`, `fixtures/schedule_<slice>_<router>_<scheduler>.json`, and `pallas_kernel/outputs/_ragged_a2a_kernel_<scheduler>_<slice>.py`. Routing-table format mirrors the existing `routing_table_8x4x4_twist.json` matrix-of-paths shape (with `vc` dropped). Schedule format is a flat list of `{round, src, dst, path}` dicts using flat-IDs. The kernel generator gains two modes: generate-routing-table-from-router, OR load-existing-routing-table — and always persists both intermediates.

**Tech Stack:** Python 3.11, numpy, pulp (already-used ILP), pytest. No new third-party dependencies.

---

## File Structure

**New files (modules):**
- `twisted_analysis/io/__init__.py` — re-exports
- `twisted_analysis/io/coords.py` — flat-id ↔ tuple coordinate utilities (`flatten`, `unflatten`)
- `twisted_analysis/io/routing_table.py` — `save_routing_table`, `load_routing_table`, `RoutingTableRouter` (Router-protocol adapter)
- `twisted_analysis/io/schedule.py` — `save_schedule`, `load_schedule`, `schedule_from_orbit_greedy`

**New CLI scripts:**
- `scripts/generate_routing_table.py` — CLI: `(slice, router) → routing-table JSON` in `fixtures/`
- `scripts/generate_schedule.py` — CLI: `(routing-table JSON, slice, scheduler, order) → schedule JSON` in `fixtures/`

**New tests:**
- `tests/test_io_coords.py`
- `tests/test_io_routing_table.py`
- `tests/test_io_schedule.py`
- `tests/test_gen_orbit_greedy_kernel_pipeline.py` (CLI-level orchestration test)

**Modified files:**
- `pallas_kernel/gen_orbit_greedy_kernel.py` — accept `--routing-table FILE` OR generate; always persist routing-table + schedule alongside the kernel; build `_DEST_TABLE_NP` from the loaded schedule (not from `_canonical_paths` directly)
- `pallas_kernel/README.md` — describe the 3-stage workflow with the example invocation
- `README.md` (top-level) — update layout section to mention `twisted_analysis/io/`, `fixtures/`, and the pipeline scripts
- `scripts/dump_routing_tables.py` — leave as-is (legacy CSV for inspection); cross-link to the new JSON dumper

**Coordinate flattening convention** (matches `gen_orbit_greedy_kernel.py:78`, dim-0 most significant):
```
flat = c0 * prod(slice[1:]) + c1 * prod(slice[2:]) + ... + c_{n-1} * 1
```
For `slice=(4,4,8)`, node `(i, j, k)` → `flat = i*32 + j*8 + k`. Verified: node 42 = `(1, 1, 2)`.

**Routing-table on-disk format** (top-level: list of N rows, one per source flat-id; each row is a list of N cells, one per destination flat-id; each cell is `{"path": [{"node_id": <int>}, ...]}`. The first node in `path` is the source, the last is the destination, and consecutive node_ids correspond to single twisted-torus hops. `vc` is intentionally omitted — the loader tolerates it if present in input files but the writer never emits it):
```json
[
  [
    {"path": [{"node_id": 0}]},
    {"path": [{"node_id": 0}, {"node_id": 1}]},
    ...
  ],
  ...
]
```

**Schedule on-disk format** (flat list of dicts; one entry per (round, src, orbit-member) triple. Total rows = `N * (N-1)` for OrbitGreedy on a full AllToAll. `round` = OrbitGreedy hop-0 step `t_0^O` of the orbit src→dst belongs to):
```json
[
  {"round": 0, "src": 0, "dst": 1, "path": [0, 1]},
  {"round": 0, "src": 1, "dst": 2, "path": [1, 2]},
  ...
]
```

---

## Task 1: Coordinate Flattening Utilities

**Files:**
- Create: `twisted_analysis/io/__init__.py`
- Create: `twisted_analysis/io/coords.py`
- Create: `tests/test_io_coords.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_io_coords.py
"""Coordinate flatten/unflatten — must match the convention in
pallas_kernel/gen_orbit_greedy_kernel.py:78 (dim-0 most significant)."""
import pytest

from twisted_analysis.io.coords import flatten, unflatten


def test_flatten_4x4x8_node_42_is_1_1_2():
    # Convention: flat = i*32 + j*8 + k for slice=(4,4,8). Node 42 -> (1,1,2).
    assert flatten((1, 1, 2), (4, 4, 8)) == 42


def test_unflatten_4x4x8_42_is_1_1_2():
    assert unflatten(42, (4, 4, 8)) == (1, 1, 2)


def test_roundtrip_all_nodes_4x4x8():
    slice_ = (4, 4, 8)
    n = 4 * 4 * 8
    for i in range(4):
        for j in range(4):
            for k in range(8):
                f = flatten((i, j, k), slice_)
                assert 0 <= f < n
                assert unflatten(f, slice_) == (i, j, k)


def test_roundtrip_2d():
    slice_ = (2, 4)
    for i in range(2):
        for j in range(4):
            f = flatten((i, j), slice_)
            assert unflatten(f, slice_) == (i, j)


def test_flatten_validates_dim_count():
    with pytest.raises(ValueError):
        flatten((0, 0), (4, 4, 8))


def test_flatten_validates_in_range():
    with pytest.raises(ValueError):
        flatten((4, 0, 0), (4, 4, 8))  # i = slice[0] is out of range
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/xutingl/collective_comm/TwistedAnalysis && uv run pytest tests/test_io_coords.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'twisted_analysis.io'`

- [ ] **Step 3: Create empty package init**

Create `twisted_analysis/io/__init__.py`:

```python
from twisted_analysis.io.coords import flatten, unflatten

__all__ = ["flatten", "unflatten"]
```

- [ ] **Step 4: Implement coords**

Create `twisted_analysis/io/coords.py`:

```python
"""Flatten / unflatten device coordinates.

Convention (matches pallas_kernel/gen_orbit_greedy_kernel.py:78):
    flat = c0 * prod(slice[1:]) + c1 * prod(slice[2:]) + ... + c_{n-1} * 1
i.e. dim 0 is most significant. Node (i, j, k) on slice=(4,4,8) maps to
flat = i*32 + j*8 + k.
"""
from __future__ import annotations
from typing import Sequence


def _strides(slice_: Sequence[int]) -> tuple[int, ...]:
    n = len(slice_)
    out = [1] * n
    for d in range(n - 2, -1, -1):
        out[d] = out[d + 1] * slice_[d + 1]
    return tuple(out)


def flatten(node: Sequence[int], slice_: Sequence[int]) -> int:
    if len(node) != len(slice_):
        raise ValueError(
            f"node has {len(node)} dims; slice has {len(slice_)}"
        )
    for d, (c, s) in enumerate(zip(node, slice_)):
        if not (0 <= c < s):
            raise ValueError(f"coord {c} out of range [0, {s}) at dim {d}")
    strides = _strides(slice_)
    return sum(c * st for c, st in zip(node, strides))


def unflatten(flat: int, slice_: Sequence[int]) -> tuple[int, ...]:
    n = 1
    for s in slice_:
        n *= s
    if not (0 <= flat < n):
        raise ValueError(f"flat={flat} out of range [0, {n})")
    strides = _strides(slice_)
    out = []
    rem = flat
    for st in strides:
        out.append(rem // st)
        rem = rem % st
    return tuple(out)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /home/xutingl/collective_comm/TwistedAnalysis && uv run pytest tests/test_io_coords.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
git add twisted_analysis/io/__init__.py twisted_analysis/io/coords.py tests/test_io_coords.py
git commit -m "feat(io): add coordinate flatten/unflatten utilities"
```

---

## Task 2: Routing-Table I/O (save, load, RoutingTableRouter adapter)

**Files:**
- Create: `twisted_analysis/io/routing_table.py`
- Create: `tests/test_io_routing_table.py`
- Modify: `twisted_analysis/io/__init__.py` (re-export)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_io_routing_table.py
"""Routing-table save/load + RoutingTableRouter adapter.

On-disk shape (matches fixtures/routing_table_8x4x4_twist.json minus vc):
  list[N] of list[N] of {"path": [{"node_id": int}, ...]}
"""
from __future__ import annotations
import json
from pathlib import Path

import pytest

from twisted_analysis.io.routing_table import (
    save_routing_table, load_routing_table, RoutingTableRouter,
)
from twisted_analysis.topology import Topology, ILPRouter, DORRouter


def test_save_routing_table_shape_2x4(tmp_path: Path):
    t = Topology(slice=(2, 4))
    r = DORRouter(t)
    out = tmp_path / "rt.json"
    save_routing_table(t, r, out)
    raw = json.loads(out.read_text())
    n = t.n_nodes
    assert isinstance(raw, list) and len(raw) == n
    for row in raw:
        assert isinstance(row, list) and len(row) == n
        for cell in row:
            assert "path" in cell
            assert all("node_id" in node for node in cell["path"])
            assert all("vc" not in node for node in cell["path"])  # vc omitted


def test_save_routing_table_self_path_is_singleton(tmp_path: Path):
    t = Topology(slice=(2, 4))
    r = DORRouter(t)
    out = tmp_path / "rt.json"
    save_routing_table(t, r, out)
    raw = json.loads(out.read_text())
    # Every diagonal cell src==dst is a single-node path.
    for f in range(t.n_nodes):
        assert len(raw[f][f]["path"]) == 1
        assert raw[f][f]["path"][0]["node_id"] == f


def test_save_routing_table_path_first_is_src_last_is_dst(tmp_path: Path):
    t = Topology(slice=(2, 4))
    r = DORRouter(t)
    out = tmp_path / "rt.json"
    save_routing_table(t, r, out)
    raw = json.loads(out.read_text())
    for src in range(t.n_nodes):
        for dst in range(t.n_nodes):
            path = raw[src][dst]["path"]
            assert path[0]["node_id"] == src
            assert path[-1]["node_id"] == dst


def test_load_routing_table_returns_matrix_of_int_paths(tmp_path: Path):
    t = Topology(slice=(2, 4))
    r = DORRouter(t)
    out = tmp_path / "rt.json"
    save_routing_table(t, r, out)
    table = load_routing_table(out)
    n = t.n_nodes
    assert len(table) == n
    for src in range(n):
        assert len(table[src]) == n
        for dst in range(n):
            path = table[src][dst]
            assert isinstance(path, list)
            assert all(isinstance(x, int) for x in path)
            assert path[0] == src
            assert path[-1] == dst


def test_load_routing_table_tolerates_vc_field(tmp_path: Path):
    # Mimic the existing routing_table_8x4x4_twist.json shape (with vc).
    raw = [
        [
            {"path": [{"node_id": 0, "vc": -1}]},
            {"path": [{"node_id": 0, "vc": 0}, {"node_id": 1, "vc": -1}]},
        ],
        [
            {"path": [{"node_id": 1, "vc": 0}, {"node_id": 0, "vc": -1}]},
            {"path": [{"node_id": 1, "vc": -1}]},
        ],
    ]
    p = tmp_path / "rt_with_vc.json"
    p.write_text(json.dumps(raw))
    table = load_routing_table(p)
    assert table == [[[0], [0, 1]], [[1, 0], [1]]]


def test_routing_table_router_path_matches_source_router_2x4_dor():
    t = Topology(slice=(2, 4))
    r = DORRouter(t)
    table = [[None] * t.n_nodes for _ in range(t.n_nodes)]
    from twisted_analysis.io.coords import flatten
    for src in t.nodes():
        for dst in t.nodes():
            path = r.path(src, dst)
            nodes = [src] + [v for (_u, v, _, _) in path]
            table[flatten(src, t.slice)][flatten(dst, t.slice)] = [
                flatten(n, t.slice) for n in nodes
            ]
    rt_router = RoutingTableRouter(topology=t, table=table)
    for src in t.nodes():
        for dst in t.nodes():
            expected = r.path(src, dst)
            actual = rt_router.path(src, dst)
            assert actual == expected


def test_routing_table_router_path_matches_disk_roundtrip(tmp_path: Path):
    t = Topology(slice=(2, 4))
    r = ILPRouter(t)
    p = tmp_path / "rt.json"
    save_routing_table(t, r, p)
    table = load_routing_table(p)
    rt_router = RoutingTableRouter(topology=t, table=table)
    for src in t.nodes():
        for dst in t.nodes():
            assert rt_router.path(src, dst) == r.path(src, dst)


def test_routing_table_router_raises_on_non_neighbor_step(tmp_path: Path):
    t = Topology(slice=(2, 4))
    # Build a table where path 0 -> 7 jumps from 0 to 7 directly (illegal).
    table = [[[i] if i == j else [i, j] for j in range(t.n_nodes)]
             for i in range(t.n_nodes)]
    rt_router = RoutingTableRouter(topology=t, table=table)
    import pytest
    with pytest.raises(ValueError, match="not a neighbor"):
        rt_router.path((0, 0), (1, 3))  # flat 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/xutingl/collective_comm/TwistedAnalysis && uv run pytest tests/test_io_routing_table.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'twisted_analysis.io.routing_table'`

- [ ] **Step 3: Implement routing-table I/O**

Create `twisted_analysis/io/routing_table.py`:

```python
"""Routing-table on-disk I/O and RoutingTableRouter adapter.

On-disk format (matches fixtures/routing_table_8x4x4_twist.json shape, with
the `vc` field intentionally omitted). Top-level: list of N rows; each row is
a list of N cells; each cell is `{"path": [{"node_id": int}, ...]}`. The
first node_id is the source, the last is the destination, and consecutive
node_ids correspond to single twisted-torus hops under
`Topology.neighbor()`.

Loader tolerates a `vc` field if present (existing fixtures contain it).
Saver never emits it.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Sequence

from twisted_analysis.io.coords import flatten, unflatten
from twisted_analysis.topology import Topology
from twisted_analysis.topology.lattice import DirectedLink, Node
from twisted_analysis.topology.router import Path as RouterPath, Router


def save_routing_table(
    topology: Topology, router: Router, out_path: Path | str,
) -> None:
    """Compute paths from `router` for every (src, dst) and write JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nodes = list(topology.nodes())
    n = len(nodes)
    matrix: list[list[dict]] = [[{} for _ in range(n)] for _ in range(n)]
    for src in nodes:
        src_flat = flatten(src, topology.slice)
        for dst in nodes:
            dst_flat = flatten(dst, topology.slice)
            if src == dst:
                matrix[src_flat][dst_flat] = {
                    "path": [{"node_id": src_flat}]
                }
                continue
            path = router.path(src, dst)
            node_ids = [src_flat] + [
                flatten(v, topology.slice) for (_u, v, _, _) in path
            ]
            matrix[src_flat][dst_flat] = {
                "path": [{"node_id": nid} for nid in node_ids]
            }
    out_path.write_text(json.dumps(matrix, indent=2))


def load_routing_table(path: Path | str) -> list[list[list[int]]]:
    """Load a routing-table JSON file. Returns `table[src][dst] = [int, ...]`.

    Tolerates a `vc` field on path nodes if present (it is dropped).
    """
    raw = json.loads(Path(path).read_text())
    n = len(raw)
    table: list[list[list[int]]] = []
    for src in range(n):
        row = raw[src]
        if len(row) != n:
            raise ValueError(
                f"row {src} has length {len(row)}; expected {n}"
            )
        out_row = []
        for dst in range(n):
            cell = row[dst]
            if "path" not in cell:
                raise ValueError(f"cell [{src}][{dst}] missing 'path'")
            out_row.append([node["node_id"] for node in cell["path"]])
        table.append(out_row)
    return table


@dataclass(frozen=True)
class RoutingTableRouter:
    """Router-protocol adapter that serves paths from a loaded routing table.

    Paths in the table are sequences of flat-IDs; this adapter reconstructs
    the (u, v, dim, dir) DirectedLink sequence by looking up each consecutive
    flat-ID pair in the topology's neighbor map.
    """
    topology: Topology
    table: list[list[list[int]]]

    @cached_property
    def _flat_to_node(self) -> dict[int, Node]:
        slice_ = self.topology.slice
        return {flatten(n, slice_): n for n in self.topology.nodes()}

    @cached_property
    def _neighbor_lookup(self) -> dict[tuple[Node, Node], tuple[int, int]]:
        """Map (u, v) -> (dim, dir) for adjacent pairs."""
        out: dict[tuple[Node, Node], tuple[int, int]] = {}
        for u in self.topology.nodes():
            for dim in range(self.topology.ndim):
                for dir in (-1, 1):
                    v = self.topology.neighbor(u, dim, dir)
                    out[(u, v)] = (dim, dir)
        return out

    def path(self, src: Node, dst: Node) -> RouterPath:
        if src == dst:
            return ()
        slice_ = self.topology.slice
        src_flat = flatten(src, slice_)
        dst_flat = flatten(dst, slice_)
        flat_path = self.table[src_flat][dst_flat]
        if len(flat_path) < 2 or flat_path[0] != src_flat or flat_path[-1] != dst_flat:
            raise ValueError(
                f"table[{src_flat}][{dst_flat}] = {flat_path} does not connect "
                f"src={src_flat} to dst={dst_flat}"
            )
        f2n = self._flat_to_node
        nb = self._neighbor_lookup
        hops: list[DirectedLink] = []
        for i in range(len(flat_path) - 1):
            u = f2n[flat_path[i]]
            v = f2n[flat_path[i + 1]]
            if (u, v) not in nb:
                raise ValueError(
                    f"flat {flat_path[i]} -> {flat_path[i+1]} is not a neighbor pair"
                )
            dim, dir = nb[(u, v)]
            hops.append((u, v, dim, dir))
        return tuple(hops)
```

- [ ] **Step 4: Update io package init**

Edit `twisted_analysis/io/__init__.py`:

```python
from twisted_analysis.io.coords import flatten, unflatten
from twisted_analysis.io.routing_table import (
    save_routing_table, load_routing_table, RoutingTableRouter,
)

__all__ = [
    "flatten", "unflatten",
    "save_routing_table", "load_routing_table", "RoutingTableRouter",
]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /home/xutingl/collective_comm/TwistedAnalysis && uv run pytest tests/test_io_routing_table.py -v`
Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
git add twisted_analysis/io/__init__.py twisted_analysis/io/routing_table.py tests/test_io_routing_table.py
git commit -m "feat(io): add routing-table JSON save/load and RoutingTableRouter adapter"
```

---

## Task 3: Router CLI Script (`scripts/generate_routing_table.py`)

**Files:**
- Create: `scripts/generate_routing_table.py`
- Modify: `tests/test_io_routing_table.py` (add CLI smoke test)

- [ ] **Step 1: Add the failing CLI test**

Append to `tests/test_io_routing_table.py`:

```python
def test_cli_generate_routing_table_writes_file(tmp_path: Path):
    import subprocess
    import sys
    out = tmp_path / "rt_2x4_dor.json"
    res = subprocess.run(
        [
            sys.executable,
            "scripts/generate_routing_table.py",
            "--slice", "2,4",
            "--router", "dor",
            "--out", str(out),
        ],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr
    assert out.exists()
    table = load_routing_table(out)
    assert len(table) == 8 and len(table[0]) == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/xutingl/collective_comm/TwistedAnalysis && uv run pytest tests/test_io_routing_table.py::test_cli_generate_routing_table_writes_file -v`
Expected: FAIL with non-zero exit code (script does not exist).

- [ ] **Step 3: Implement the CLI**

Create `scripts/generate_routing_table.py`:

```python
"""Generate a routing-table JSON file for a {S, 2S}^n twisted-torus topology.

Output: matrix-of-paths JSON in the shape of fixtures/routing_table_8x4x4_twist.json
(with `vc` omitted). Default destination: fixtures/routing_table_<slice>_<router>.json.

Usage:
    python scripts/generate_routing_table.py --slice 4,4,8 --router ilp
    python scripts/generate_routing_table.py --slice 2,4 --router dor --out /tmp/rt.json
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Make `python scripts/generate_routing_table.py` work without install.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from twisted_analysis.io.routing_table import save_routing_table
from twisted_analysis.topology import Topology, DORRouter, ILPRouter


def _parse_slice(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def _build_router(name: str, topology: Topology):
    name = name.lower()
    if name == "ilp":
        return ILPRouter(topology=topology)
    if name == "dor":
        return DORRouter(topology=topology)
    raise ValueError(f"unknown router: {name!r} (choose ilp|dor)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Generate a routing-table JSON for a twisted-torus topology.",
    )
    p.add_argument("--slice", required=True,
                   help="Comma-separated slice, e.g. 4,4,8")
    p.add_argument("--router", default="ilp", choices=["ilp", "dor"])
    p.add_argument("--out", default=None,
                   help="Output path (default: ./fixtures/routing_table_<slice>_<router>.json)")
    args = p.parse_args(argv)

    slice_ = _parse_slice(args.slice)
    topology = Topology(slice=slice_)
    router = _build_router(args.router, topology)

    if args.out is None:
        slice_str = "x".join(str(s) for s in slice_)
        out_path = _HERE.parent / "fixtures" / f"routing_table_{slice_str}_{args.router}.json"
    else:
        out_path = Path(args.out)

    save_routing_table(topology, router, out_path)
    print(f"wrote {out_path} ({out_path.stat().st_size:,} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/xutingl/collective_comm/TwistedAnalysis && uv run pytest tests/test_io_routing_table.py::test_cli_generate_routing_table_writes_file -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
git add scripts/generate_routing_table.py tests/test_io_routing_table.py
git commit -m "feat(scripts): add generate_routing_table CLI"
```

---

## Task 4: Schedule I/O — save and load

**Files:**
- Create: `twisted_analysis/io/schedule.py`
- Create: `tests/test_io_schedule.py`
- Modify: `twisted_analysis/io/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_io_schedule.py
"""Schedule JSON save/load.

Format: list of dicts {"round": int, "src": int, "dst": int, "path": [int, ...]}.
"""
from __future__ import annotations
import json
from pathlib import Path

import pytest

from twisted_analysis.io.schedule import save_schedule, load_schedule


def test_save_schedule_writes_list_of_dicts(tmp_path: Path):
    entries = [
        {"round": 0, "src": 0, "dst": 1, "path": [0, 1]},
        {"round": 0, "src": 1, "dst": 0, "path": [1, 0]},
        {"round": 1, "src": 0, "dst": 2, "path": [0, 1, 2]},
    ]
    p = tmp_path / "sched.json"
    save_schedule(entries, p)
    raw = json.loads(p.read_text())
    assert raw == entries


def test_load_schedule_roundtrip(tmp_path: Path):
    entries = [
        {"round": 0, "src": 0, "dst": 42,
         "path": [0, 16, 32, 36, 40, 41, 42]},
    ]
    p = tmp_path / "sched.json"
    save_schedule(entries, p)
    out = load_schedule(p)
    assert out == entries


def test_save_schedule_validates_required_keys(tmp_path: Path):
    bad = [{"round": 0, "src": 0, "dst": 1}]  # missing 'path'
    with pytest.raises(ValueError, match="path"):
        save_schedule(bad, tmp_path / "x.json")


def test_save_schedule_validates_path_endpoints(tmp_path: Path):
    bad = [{"round": 0, "src": 0, "dst": 5, "path": [0, 1, 2]}]  # last != dst
    with pytest.raises(ValueError, match="dst"):
        save_schedule(bad, tmp_path / "x.json")


def test_save_schedule_creates_parent_dirs(tmp_path: Path):
    p = tmp_path / "nested" / "deep" / "sched.json"
    save_schedule(
        [{"round": 0, "src": 0, "dst": 1, "path": [0, 1]}],
        p,
    )
    assert p.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/xutingl/collective_comm/TwistedAnalysis && uv run pytest tests/test_io_schedule.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement schedule I/O**

Create `twisted_analysis/io/schedule.py`:

```python
"""Schedule on-disk I/O.

Format: list of dicts, one per (round, src, dst) triple. Each dict has at
least the keys: {"round": int, "src": int, "dst": int, "path": [int, ...]}.

`src` and `dst` are flat-IDs; `path` is the sequence of flat-IDs traversed
from src to dst (inclusive of both endpoints).
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, Mapping

ScheduleEntry = Mapping[str, object]
_REQUIRED = ("round", "src", "dst", "path")


def _validate(entries: Iterable[ScheduleEntry]) -> list[dict]:
    out = []
    for i, e in enumerate(entries):
        for k in _REQUIRED:
            if k not in e:
                raise ValueError(f"entry {i} missing required key {k!r}: {dict(e)}")
        path = e["path"]
        if not isinstance(path, list) or not path:
            raise ValueError(f"entry {i}: path must be non-empty list")
        if path[0] != e["src"]:
            raise ValueError(
                f"entry {i}: path[0]={path[0]} != src={e['src']}"
            )
        if path[-1] != e["dst"]:
            raise ValueError(
                f"entry {i}: path[-1]={path[-1]} != dst={e['dst']}"
            )
        out.append(dict(e))
    return out


def save_schedule(entries: Iterable[ScheduleEntry], out_path: Path | str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    validated = _validate(entries)
    out_path.write_text(json.dumps(validated, indent=2))


def load_schedule(path: Path | str) -> list[dict]:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list):
        raise ValueError(f"{path}: top-level must be a list")
    return _validate(raw)
```

- [ ] **Step 4: Update io package init**

Edit `twisted_analysis/io/__init__.py`:

```python
from twisted_analysis.io.coords import flatten, unflatten
from twisted_analysis.io.routing_table import (
    save_routing_table, load_routing_table, RoutingTableRouter,
)
from twisted_analysis.io.schedule import save_schedule, load_schedule

__all__ = [
    "flatten", "unflatten",
    "save_routing_table", "load_routing_table", "RoutingTableRouter",
    "save_schedule", "load_schedule",
]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /home/xutingl/collective_comm/TwistedAnalysis && uv run pytest tests/test_io_schedule.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
git add twisted_analysis/io/schedule.py twisted_analysis/io/__init__.py tests/test_io_schedule.py
git commit -m "feat(io): add schedule JSON save/load with endpoint validation"
```

---

## Task 5: Build OrbitGreedy schedule from a routing table

**Files:**
- Modify: `twisted_analysis/io/schedule.py` (add `schedule_from_orbit_greedy`)
- Create: `tests/test_io_schedule.py` additions (parametrized correctness test)

- [ ] **Step 1: Append the failing test**

Append to `tests/test_io_schedule.py`:

```python
import pytest


def test_schedule_from_orbit_greedy_2x4_dor():
    from twisted_analysis.io.schedule import schedule_from_orbit_greedy
    from twisted_analysis.io.coords import flatten
    from twisted_analysis.io.routing_table import (
        save_routing_table, load_routing_table,
    )
    from twisted_analysis.topology import Topology, DORRouter

    t = Topology(slice=(2, 4))
    r = DORRouter(t)

    # Build the routing table in-memory in the loaded shape.
    table = []
    for src in t.nodes():
        row = []
        for dst in t.nodes():
            if src == dst:
                row.append([flatten(src, t.slice)])
                continue
            path = r.path(src, dst)
            nodes = [src] + [v for (_u, v, _, _) in path]
            row.append([flatten(n, t.slice) for n in nodes])
        table.append(row)

    entries = schedule_from_orbit_greedy(t, table, order="lpt_tail_asc")

    # Full coverage: N * (N - 1) entries.
    n = t.n_nodes
    assert len(entries) == n * (n - 1)

    # Each entry's path is consistent with src/dst and uses int flat-IDs.
    for e in entries:
        assert e["path"][0] == e["src"]
        assert e["path"][-1] == e["dst"]
        assert isinstance(e["round"], int)

    # For each src, the destinations span every other device exactly once.
    by_src: dict[int, set[int]] = {}
    for e in entries:
        by_src.setdefault(e["src"], set()).add(e["dst"])
    for src_flat in range(n):
        assert by_src[src_flat] == set(range(n)) - {src_flat}


def test_schedule_from_orbit_greedy_invalid_order():
    from twisted_analysis.io.schedule import schedule_from_orbit_greedy
    from twisted_analysis.topology import Topology, DORRouter
    from twisted_analysis.io.routing_table import save_routing_table, load_routing_table

    t = Topology(slice=(2, 4))
    r = DORRouter(t)
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "rt.json"
        save_routing_table(t, r, p)
        table = load_routing_table(p)
        with pytest.raises(ValueError):
            schedule_from_orbit_greedy(t, table, order="bogus")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/xutingl/collective_comm/TwistedAnalysis && uv run pytest tests/test_io_schedule.py::test_schedule_from_orbit_greedy_2x4_dor -v`
Expected: FAIL with `ImportError: cannot import name 'schedule_from_orbit_greedy'`.

- [ ] **Step 3: Implement schedule_from_orbit_greedy**

Append to `twisted_analysis/io/schedule.py`:

```python
def schedule_from_orbit_greedy(
    topology, table, *, order: str = "lpt_tail_asc",
) -> list[dict]:
    """Run the OrbitGreedy scheduler against a routing table; return entries.

    For each orbit O firing hop-0 at OrbitGreedy step `t_0^O`, emit one entry
    per source `s`:
        {"round": t_0^O, "src": flat(s), "dst": flat(s + δ_O),
         "path": [flat-IDs along the canonical path translated to s]}

    Total entries: `N * (N - 1)` for a full AllToAll. Entries are sorted by
    (round, src) for determinism.
    """
    from twisted_analysis.io.coords import flatten
    from twisted_analysis.io.routing_table import RoutingTableRouter
    from twisted_analysis.lp.orbit import compute_orbits
    from twisted_analysis.schedules.orbit_greedy import (
        _emit_orbit_greedy, _ordered_orbits, _canonical_paths, _edge_orbit_load,
    )

    rt_router = RoutingTableRouter(topology=topology, table=table)

    canon = _canonical_paths(topology, rt_router)
    edge_load = _edge_orbit_load(canon)
    ordered = _ordered_orbits(canon, edge_load, order)

    assignment = _emit_orbit_greedy(topology, rt_router, order)
    # hop-0 firing time per orbit
    t0: dict = {}
    for (orbit_id, hop_i, t), _v in assignment.items():
        if hop_i == 0:
            t0[orbit_id] = t

    orbits = compute_orbits(topology)
    slice_ = topology.slice

    entries: list[dict] = []
    for orbit_id in ordered:
        round_t = t0[orbit_id]
        # `members` contains (src, dst) pairs for every source in this orbit.
        for (src, dst) in orbits[orbit_id]:
            src_flat = flatten(src, slice_)
            dst_flat = flatten(dst, slice_)
            entries.append({
                "round": int(round_t),
                "src": src_flat,
                "dst": dst_flat,
                "path": list(table[src_flat][dst_flat]),
            })
    entries.sort(key=lambda e: (e["round"], e["src"]))
    return entries
```

- [ ] **Step 4: Re-export the new function**

Edit `twisted_analysis/io/__init__.py` to add `schedule_from_orbit_greedy`:

```python
from twisted_analysis.io.coords import flatten, unflatten
from twisted_analysis.io.routing_table import (
    save_routing_table, load_routing_table, RoutingTableRouter,
)
from twisted_analysis.io.schedule import (
    save_schedule, load_schedule, schedule_from_orbit_greedy,
)

__all__ = [
    "flatten", "unflatten",
    "save_routing_table", "load_routing_table", "RoutingTableRouter",
    "save_schedule", "load_schedule", "schedule_from_orbit_greedy",
]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /home/xutingl/collective_comm/TwistedAnalysis && uv run pytest tests/test_io_schedule.py -v`
Expected: 7 passed (5 from Task 4 + 2 new)

- [ ] **Step 6: Commit**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
git add twisted_analysis/io/schedule.py twisted_analysis/io/__init__.py tests/test_io_schedule.py
git commit -m "feat(io): add schedule_from_orbit_greedy (routing-table -> schedule entries)"
```

---

## Task 6: Schedule Generator CLI (`scripts/generate_schedule.py`)

**Files:**
- Create: `scripts/generate_schedule.py`
- Modify: `tests/test_io_schedule.py` (CLI smoke test)

- [ ] **Step 1: Append the failing test**

Append to `tests/test_io_schedule.py`:

```python
def test_cli_generate_schedule_writes_file(tmp_path: Path):
    import subprocess
    import sys
    from twisted_analysis.io.routing_table import save_routing_table
    from twisted_analysis.topology import Topology, DORRouter

    t = Topology(slice=(2, 4))
    r = DORRouter(t)
    rt_path = tmp_path / "rt.json"
    save_routing_table(t, r, rt_path)

    out = tmp_path / "sched.json"
    res = subprocess.run(
        [
            sys.executable,
            "scripts/generate_schedule.py",
            "--routing-table", str(rt_path),
            "--slice", "2,4",
            "--scheduler", "orbit_greedy",
            "--order", "lpt_tail_asc",
            "--out", str(out),
        ],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr
    assert out.exists()
    entries = load_schedule(out)
    assert len(entries) == 8 * 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/xutingl/collective_comm/TwistedAnalysis && uv run pytest tests/test_io_schedule.py::test_cli_generate_schedule_writes_file -v`
Expected: FAIL with non-zero exit.

- [ ] **Step 3: Implement the CLI**

Create `scripts/generate_schedule.py`:

```python
"""Generate a schedule JSON from a routing table.

Currently supports only `--scheduler orbit_greedy`. Extend the dispatch dict
in `_run` to add more schedulers.

Usage:
    python scripts/generate_schedule.py \\
        --routing-table fixtures/routing_table_8x4x4_twist.json \\
        --slice 4,4,8 \\
        --scheduler orbit_greedy \\
        --order lpt_tail_asc
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Make `python scripts/generate_schedule.py` work without install.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from twisted_analysis.io.routing_table import load_routing_table
from twisted_analysis.io.schedule import save_schedule, schedule_from_orbit_greedy
from twisted_analysis.topology import Topology


def _parse_slice(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def _run(scheduler: str, topology: Topology, table: list, order: str) -> list[dict]:
    if scheduler == "orbit_greedy":
        return schedule_from_orbit_greedy(topology, table, order=order)
    raise ValueError(f"unknown scheduler: {scheduler!r}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Generate a schedule JSON from a routing-table JSON.",
    )
    p.add_argument("--routing-table", required=True, type=Path,
                   help="Path to routing-table JSON (matrix-of-paths shape)")
    p.add_argument("--slice", required=True,
                   help="Comma-separated slice, e.g. 4,4,8 — must match the table size")
    p.add_argument("--scheduler", default="orbit_greedy", choices=["orbit_greedy"])
    p.add_argument("--order", default="lpt_tail_asc",
                   choices=["lpt_tail_asc", "lpt", "spt", "tail_asc"])
    p.add_argument("--out", default=None,
                   help="Output path (default: ./fixtures/schedule_<slice>_<scheduler>_<order>.json)")
    args = p.parse_args(argv)

    slice_ = _parse_slice(args.slice)
    topology = Topology(slice=slice_)
    table = load_routing_table(args.routing_table)
    if len(table) != topology.n_nodes:
        raise SystemExit(
            f"routing table has {len(table)} sources; slice {slice_} expects {topology.n_nodes}"
        )

    entries = _run(args.scheduler, topology, table, args.order)

    if args.out is None:
        slice_str = "x".join(str(s) for s in slice_)
        out_path = _HERE.parent / "fixtures" / (
            f"schedule_{slice_str}_{args.scheduler}_{args.order}.json"
        )
    else:
        out_path = Path(args.out)

    save_schedule(entries, out_path)
    print(f"wrote {out_path} ({out_path.stat().st_size:,} bytes, "
          f"{len(entries):,} entries)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/xutingl/collective_comm/TwistedAnalysis && uv run pytest tests/test_io_schedule.py::test_cli_generate_schedule_writes_file -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
git add scripts/generate_schedule.py tests/test_io_schedule.py
git commit -m "feat(scripts): add generate_schedule CLI"
```

---

## Task 7: Refactor `gen_orbit_greedy_kernel.py` to orchestrate the full pipeline

**Files:**
- Modify: `pallas_kernel/gen_orbit_greedy_kernel.py`
- Create: `tests/test_gen_orbit_greedy_kernel_pipeline.py`

The kernel generator must:
1. Either load an existing routing-table JSON (via `--routing-table`) OR generate one (via `--router`) and save it to `fixtures/routing_table_<slice>_<router>.json`.
2. Generate a schedule from the routing table and save it to `fixtures/schedule_<slice>_<router>_<order>.json`.
3. Build the `_DEST_TABLE_NP` and `_ORBIT_STEPS` literals **from the loaded schedule** (rather than re-computing canonical paths). This guarantees the kernel reflects exactly what's on disk.

**Design notes for `_DEST_TABLE_NP` and `_ORBIT_STEPS` from a schedule:**

- Group schedule entries by `round` → ordered list of rounds `R = sorted(unique rounds)`.
- For each src, sort its entries by `round`. Use the resulting destination order as the columns of `_DEST_TABLE_NP[src, k]`. Because the OrbitGreedy hop-0 step `t_0^O` is the same for every source in orbit O, the column index `k` (after stable sort by round) corresponds to one orbit per column for every source — same as the existing per-orbit table.
- `_ORBIT_STEPS[t]` = list of column indices `k` whose hop-0 round equals `R[t]`. Since every source agrees on which `k` belongs to which round, derive this once from src=0's row.

- [ ] **Step 1: Write the failing pipeline test**

Create `tests/test_gen_orbit_greedy_kernel_pipeline.py`:

```python
"""End-to-end test for gen_orbit_greedy_kernel.py orchestration.

Verifies:
  * Generating from --slice + --router produces both intermediate files
    AND the kernel file.
  * Generating from an existing --routing-table reuses the table and
    produces a schedule + kernel.
  * The generated kernel parses as Python.
"""
from __future__ import annotations
import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "pallas_kernel/gen_orbit_greedy_kernel.py", *args],
        cwd=cwd, capture_output=True, text=True,
    )


def test_pipeline_from_router_writes_all_three_artifacts(tmp_path: Path):
    # We isolate output by overriding the three output paths.
    rt_out = tmp_path / "rt.json"
    sched_out = tmp_path / "sched.json"
    kernel_out = tmp_path / "kernel.py"
    res = _run([
        "--slice", "2,4",
        "--router", "dor",
        "--routing-table-out", str(rt_out),
        "--schedule-out", str(sched_out),
        "--out", str(kernel_out),
    ], cwd=REPO)
    assert res.returncode == 0, res.stderr
    assert rt_out.exists()
    assert sched_out.exists()
    assert kernel_out.exists()
    # Generated kernel parses.
    ast.parse(kernel_out.read_text())


def test_pipeline_from_existing_routing_table_does_not_overwrite(tmp_path: Path):
    # Generate a routing table, capture its mtime, then re-run with --routing-table.
    rt_in = tmp_path / "rt_in.json"
    sched_out = tmp_path / "sched.json"
    kernel_out = tmp_path / "kernel.py"

    # Pre-generate the routing table in a separate invocation.
    res0 = subprocess.run(
        [sys.executable, "scripts/generate_routing_table.py",
         "--slice", "2,4", "--router", "dor", "--out", str(rt_in)],
        cwd=REPO, capture_output=True, text=True,
    )
    assert res0.returncode == 0, res0.stderr
    rt_mtime_before = rt_in.stat().st_mtime_ns

    res = _run([
        "--slice", "2,4",
        "--routing-table", str(rt_in),
        "--schedule-out", str(sched_out),
        "--out", str(kernel_out),
    ], cwd=REPO)
    assert res.returncode == 0, res.stderr
    # Routing table is the input, not regenerated:
    assert rt_in.stat().st_mtime_ns == rt_mtime_before
    assert sched_out.exists()
    assert kernel_out.exists()
    ast.parse(kernel_out.read_text())


def test_pipeline_from_routing_table_8x4x4_twist_fixture(tmp_path: Path):
    """Exercise the example from the spec: load the existing 4x4x8 fixture
    and produce a schedule + kernel."""
    rt = REPO / "fixtures" / "routing_table_8x4x4_twist.json"
    assert rt.exists()
    sched_out = tmp_path / "sched_4x4x8.json"
    kernel_out = tmp_path / "kernel_4x4x8.py"
    res = _run([
        "--slice", "4,4,8",
        "--routing-table", str(rt),
        "--schedule-out", str(sched_out),
        "--out", str(kernel_out),
    ], cwd=REPO)
    assert res.returncode == 0, res.stderr
    assert sched_out.exists()
    assert kernel_out.exists()
    ast.parse(kernel_out.read_text())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/xutingl/collective_comm/TwistedAnalysis && uv run pytest tests/test_gen_orbit_greedy_kernel_pipeline.py -v`
Expected: FAIL — generator does not yet accept `--routing-table` / `--routing-table-out` / `--schedule-out`.

- [ ] **Step 3: Refactor the generator**

Replace `pallas_kernel/gen_orbit_greedy_kernel.py` with the version below. Key changes from the previous version:

- New CLI flags: `--routing-table FILE` (load), `--routing-table-out FILE` (save when generated), `--schedule-out FILE`, plus an exclusivity check between `--routing-table` and `--router`.
- New helper `_dest_table_and_orbit_steps_from_schedule(schedule, n)` that derives `_DEST_TABLE_NP[N, K]` and `_ORBIT_STEPS` from the schedule entries, replacing the previous `_build_dest_table` + `_hop0_steps` (which used `router.path` directly).
- The generator persists the routing table (when generated) and the schedule unconditionally before emitting the kernel.

```python
"""Generator for orbit-greedy P2P AllToAll Pallas TPU kernels.

Pipeline:
    [router OR --routing-table FILE] -> routing-table JSON
                  -> schedule_from_orbit_greedy -> schedule JSON
                  -> _DEST_TABLE_NP, _ORBIT_STEPS literals
                  -> kernel .py source

All three artifacts are written to disk before the kernel is emitted, so the
intermediate routing table and schedule can be inspected independently.

Usage (CLI):

    # Generate routing-table from a router:
    python pallas_kernel/gen_orbit_greedy_kernel.py --slice 4,4,8 --router ilp

    # Reuse an existing routing-table:
    python pallas_kernel/gen_orbit_greedy_kernel.py \\
        --slice 4,4,8 \\
        --routing-table fixtures/routing_table_8x4x4_twist.json

Default outputs:
    routing table: ./fixtures/routing_table_<slice>_<router>.json
    schedule:      ./fixtures/schedule_<slice>_<router>_<order>.json
    kernel:        ./pallas_kernel/outputs/_ragged_a2a_kernel_orbit_greedy_<slice>.py
"""
from __future__ import annotations
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Make `python pallas_kernel/gen_orbit_greedy_kernel.py` work without install.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from twisted_analysis.io.routing_table import (
    save_routing_table, load_routing_table,
)
from twisted_analysis.io.schedule import save_schedule, schedule_from_orbit_greedy
from twisted_analysis.topology import Topology, DORRouter, ILPRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dest_table_and_orbit_steps_from_schedule(
    schedule: list[dict], n: int,
) -> tuple[np.ndarray, list[list[int]]]:
    """Build _DEST_TABLE_NP[n, K] and _ORBIT_STEPS from schedule entries.

    Strategy:
      * Sort schedule by (round, src). For each src, the per-round destination
        sequence becomes the columns of _DEST_TABLE_NP[src].
      * The k-th column corresponds to one orbit; all sources agree on the
        round-of-column-k, so derive _ORBIT_STEPS once from src=0.
    """
    # Collect per-(src) ordered (round, dst) lists.
    by_src: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for e in schedule:
        by_src[e["src"]].append((e["round"], e["dst"]))
    if any(len(by_src[s]) != n - 1 for s in range(n)):
        raise RuntimeError(
            f"schedule does not cover full AllToAll: "
            f"each source needs {n - 1} entries"
        )
    for s in range(n):
        by_src[s].sort()  # by (round, dst); rounds may tie (different orbits at same step)

    K = n - 1
    table = np.zeros((n, K), dtype=np.int32)
    for src in range(n):
        for k, (_round, dst) in enumerate(by_src[src]):
            table[src, k] = dst

    # _ORBIT_STEPS: bucket column indices by the round of src=0.
    rounds_src0 = [r for (r, _d) in by_src[0]]
    by_step: dict[int, list[int]] = defaultdict(list)
    for k, r in enumerate(rounds_src0):
        by_step[r].append(k)
    orbit_steps = [by_step[r] for r in sorted(by_step.keys())]
    return table, orbit_steps


def _dest_table_literal(table: np.ndarray) -> str:
    """Compact module-level literal for an int32 [N, K] table."""
    rows = ["    [" + ", ".join(f"{v:3d}" for v in row) + "]"
            for row in table]
    return "np.array([\n" + ",\n".join(rows) + ",\n], dtype=np.int32)"


# ---------------------------------------------------------------------------
# Source builder (split out for testability)
# ---------------------------------------------------------------------------

def generate_kernel_source(
    *,
    slice_: tuple[int, ...],
    router_name_for_doc: str,
    order: str,
    per_step_barrier: bool,
    function_name: str | None,
    dest_table: np.ndarray,
    orbit_steps: list[list[int]],
) -> str:
    """Emit kernel source from already-computed dest_table + orbit_steps.

    Pure function: takes the schedule-derived arrays as inputs and emits the
    .py source string. Does NOT touch disk and does NOT call routers.
    """
    n_dim = len(slice_)
    N = 1
    for s in slice_:
        N *= s

    if function_name is None:
        slice_str = "_".join(str(s) for s in slice_)
        function_name = f"_ragged_a2a_kernel_orbit_greedy_{slice_str}"

    K = dest_table.shape[1]
    if K != N - 1:
        raise RuntimeError(f"Expected dest_table cols = {N - 1}; got {K}")
    makespan_hop0 = len(orbit_steps)

    L: list[str] = []

    # ---- header ------------------------------------------------------------
    L.append('"""Orbit-greedy P2P AllToAll Pallas TPU kernel.')
    L.append('')
    L.append('AUTO-GENERATED — DO NOT EDIT BY HAND.')
    L.append('')
    L.append(f'Topology:        slice={slice_}  (N={N} devices, ndim={n_dim})')
    L.append(f'Router:          {router_name_for_doc}')
    L.append(f'OrbitGreedy:     order={order!r}')
    L.append(f'Per-step barrier: {per_step_barrier}')
    L.append(f'Hop-0 steps:     {makespan_hop0}')
    L.append('')
    L.append('Generated from:  routing-table JSON + schedule JSON')
    L.append('"""')
    L.append('from __future__ import annotations')
    L.append('')
    L.append('import jax')
    L.append('import numpy as np')
    L.append('from jax import lax')
    L.append('from jax.experimental import pallas as pl')
    L.append('from jax.experimental.pallas import tpu as pltpu')
    L.append('')
    L.append('from megablox.collectives import ragged_collectives_utils  # type: ignore')
    L.append('')
    L.append('')
    L.append('# ----------------------------- baked schedule -------------------------------')
    L.append(f'_DEST_TABLE_NP = {_dest_table_literal(dest_table)}')
    L.append(f'assert _DEST_TABLE_NP.shape == ({N}, {K}), (')
    L.append('    f"_DEST_TABLE_NP shape mismatch: {_DEST_TABLE_NP.shape}"')
    L.append(')')
    L.append('')
    L.append(f'# _ORBIT_STEPS[t] = orbit indices firing at OrbitGreedy step t. '
             f'{makespan_hop0} steps total.')
    L.append('_ORBIT_STEPS = [')
    for t, step in enumerate(orbit_steps):
        L.append(f'    {step!r},  # step {t} ({len(step)} concurrent orbit(s))')
    L.append(']')
    L.append('')
    L.append('')

    # ---- function header --------------------------------------------------
    L.append(f'def {function_name}(')
    L.append('    input_offsets_ref,')
    L.append('    output_offsets_ref,')
    L.append('    sizes_ref,')
    L.append('    total_send_amount_ref,')
    L.append('    total_recv_amount_ref,')
    L.append('    num_packets_per_group_ref,')
    L.append('    dest_table_ref,  # int32[N, K] in SMEM — pass as extra pallas_call input')
    L.append('    x_ref,')
    L.append('    _,')
    L.append('    o_ref,')
    L.append('    scratch_ref,')
    L.append('    send_sem,')
    L.append('    recv_sem,')
    L.append('    scratch_sems,')
    L.append('    *,')
    L.append('    axis_name,')
    L.append('    transpose,')
    L.append('    packet_size,')
    L.append('    enable_checks: bool = False,')
    L.append('):')
    L.append(f'    """Orbit-greedy P2P AllToAll kernel for slice={slice_}."""')
    L.append('    assert scratch_ref is None')
    L.append('    del scratch_ref')
    L.append('    assert scratch_sems is None')
    L.append('    del scratch_sems')
    L.append('    assert not transpose, (')
    L.append('        "transpose=True not supported by orbit-greedy kernel; use reference."')
    L.append('    )')
    L.append('')
    L.append('    my_flat = jax.lax.axis_index(axis_name)')
    L.append('    axis_size_local = jax.lax.axis_size(axis_name)')
    L.append('')
    L.append('    num_groups = sizes_ref.shape[0]')
    L.append(f'    assert num_groups == {N}, (')
    L.append(f'        f"Expected num_groups={N} (uniform AllToAll on {slice_}); got {{num_groups}}"')
    L.append('    )')
    L.append('    groups_per_shard, r = divmod(num_groups, axis_size_local)')
    L.append('    assert r == 0 and groups_per_shard == 1, (')
    L.append('        "orbit-greedy kernel assumes 1 group per device"')
    L.append('    )')
    L.append('')
    L.append('    if axis_size_local > 1:')
    L.append('        ragged_collectives_utils.main_barrier(')
    L.append('            axis_name,')
    L.append('            barrier_type=ragged_collectives_utils.BarrierType.ALL_TO_ALL,')
    L.append('        )')
    L.append('')
    L.append('    num_packets = num_packets_per_group_ref[0]')
    L.append('')
    L.append('    def _issue_packet(packet_idx, group_idx, dst_device_id):')
    L.append('        size = lax.min(')
    L.append('            packet_size,')
    L.append('            lax.max(sizes_ref[group_idx] - packet_idx * packet_size, 0),')
    L.append('        )')
    L.append('        input_offset = input_offsets_ref[group_idx] + packet_idx * packet_size')
    L.append('        output_offset = output_offsets_ref[group_idx] + packet_idx * packet_size')
    L.append('')
    L.append('        @pl.when(size > 0)')
    L.append('        def _():')
    L.append('            if axis_size_local > 1:')
    L.append('                pltpu.make_async_remote_copy(')
    L.append('                    x_ref.at[pl.ds(input_offset, size)],')
    L.append('                    o_ref.at[pl.ds(output_offset, size)],')
    L.append('                    device_id=dst_device_id,')
    L.append('                    send_sem=send_sem,')
    L.append('                    recv_sem=recv_sem,')
    L.append('                ).start()')
    L.append('            else:')
    L.append('                pltpu.make_async_copy(')
    L.append('                    x_ref.at[pl.ds(input_offset, size)],')
    L.append('                    o_ref.at[pl.ds(output_offset, size)],')
    L.append('                    sem=send_sem,')
    L.append('                ).start()')
    L.append('')
    L.append('    def _self_body(packet_idx, _state):')
    L.append('        _issue_packet(packet_idx, my_flat, {axis_name: my_flat})')
    L.append('        return _state')
    L.append('')
    L.append('    jax.lax.fori_loop(0, num_packets, _self_body, None)')
    L.append('')

    if not per_step_barrier:
        L.append('    # ---- main orbit loop: packet outer, OrbitGreedy order inner ----')
        L.append(f'    _NUM_ORBITS = {K}')
        L.append('    def _body(i, _state):')
        L.append('        packet_idx = lax.div(i, _NUM_ORBITS)')
        L.append('        k = lax.rem(i, _NUM_ORBITS)')
        L.append('        dst_flat = dest_table_ref[my_flat, k]')
        L.append('        _issue_packet(packet_idx, dst_flat, {axis_name: dst_flat})')
        L.append('        return _state')
        L.append('')
        L.append('    jax.lax.fori_loop(0, _NUM_ORBITS * num_packets, _body, None)')
        L.append('')
        L.append('    send_amount = total_send_amount_ref[0]')
        L.append('    recv_amount = total_recv_amount_ref[0]')
        L.append('    pltpu.make_async_copy(')
        L.append('        o_ref.at[pl.ds(0, send_amount)],')
        L.append('        o_ref.at[pl.ds(0, send_amount)],')
        L.append('        send_sem,')
        L.append('    ).wait()')
        L.append('    if axis_size_local > 1:')
        L.append('        pltpu.make_async_copy(')
        L.append('            o_ref.at[pl.ds(0, recv_amount)],')
        L.append('            o_ref.at[pl.ds(0, recv_amount)],')
        L.append('            recv_sem,')
        L.append('        ).wait()')
    else:
        L.append('    _self_bytes = sizes_ref[my_flat]')
        L.append('    pltpu.make_async_copy(')
        L.append('        o_ref.at[pl.ds(0, _self_bytes)],')
        L.append('        o_ref.at[pl.ds(0, _self_bytes)],')
        L.append('        send_sem,')
        L.append('    ).wait()')
        L.append('    if axis_size_local > 1:')
        L.append('        pltpu.make_async_copy(')
        L.append('            o_ref.at[pl.ds(0, _self_bytes)],')
        L.append('            o_ref.at[pl.ds(0, _self_bytes)],')
        L.append('            recv_sem,')
        L.append('        ).wait()')
        L.append('')
        L.append('    def _issue_orbit(k):')
        L.append('        dst_flat = dest_table_ref[my_flat, k]')
        L.append('        dst_dev = {axis_name: dst_flat}')
        L.append('        def _pb(packet_idx, _state):')
        L.append('            _issue_packet(packet_idx, dst_flat, dst_dev)')
        L.append('            return _state')
        L.append('        jax.lax.fori_loop(0, num_packets, _pb, None)')
        L.append('')
        L.append('    def _drain_step(step_indices):')
        L.append('        cum = 0')
        L.append('        for k in step_indices:')
        L.append('            cum = cum + sizes_ref[dest_table_ref[my_flat, k]]')
        L.append('        pltpu.make_async_copy(')
        L.append('            o_ref.at[pl.ds(0, cum)],')
        L.append('            o_ref.at[pl.ds(0, cum)],')
        L.append('            send_sem,')
        L.append('        ).wait()')
        L.append('        if axis_size_local > 1:')
        L.append('            pltpu.make_async_copy(')
        L.append('                o_ref.at[pl.ds(0, cum)],')
        L.append('                o_ref.at[pl.ds(0, cum)],')
        L.append('                recv_sem,')
        L.append('            ).wait()')
        L.append('')
        for t, step in enumerate(orbit_steps):
            L.append(f'    # ---- OrbitGreedy step {t} ({len(step)} orbit(s)) ----')
            for k in step:
                L.append(f'    _issue_orbit({k})')
            L.append(f'    _drain_step({step!r})')
            L.append('')

    L.append('')
    L.append('def build_pallas_call_kwargs():')
    L.append('    """Helper for inserting _DEST_TABLE_NP as an extra pallas_call input."""')
    L.append('    import jax.numpy as jnp')
    L.append('    return {')
    L.append('        "dest_table": jnp.asarray(_DEST_TABLE_NP),')
    L.append('        "extra_in_spec": pl.BlockSpec(memory_space=pltpu.SMEM),')
    L.append('        "input_output_aliases_shift": 1,')
    L.append('    }')
    return "\n".join(L)


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------

def _parse_slice(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def _build_router(name: str, topology: Topology):
    name = name.lower()
    if name == "ilp":
        return ILPRouter(topology=topology), "ILP"
    if name == "dor":
        return DORRouter(topology=topology), "DOR"
    raise ValueError(f"unknown router: {name!r} (choose ilp|dor)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Generate orbit-greedy P2P Pallas kernel source via the "
                    "router -> scheduler -> kernel pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--slice", required=True,
                   help="Comma-separated slice, e.g. 4,4,8")
    p.add_argument("--routing-table", default=None, type=Path,
                   help="Load this routing-table JSON instead of generating one.")
    p.add_argument("--router", default=None, choices=["ilp", "dor"],
                   help="Router used to generate the routing table "
                        "(ignored if --routing-table is given). Default: ilp.")
    p.add_argument("--order", default="lpt_tail_asc",
                   choices=["lpt_tail_asc", "lpt", "spt", "tail_asc"])
    p.add_argument("--per-step-barrier", action="store_true",
                   help="Emit per-OrbitGreedy-step barriers (best-effort).")
    p.add_argument("--function-name", default=None)
    p.add_argument("--routing-table-out", default=None, type=Path,
                   help="Where to save the generated routing table "
                        "(default: ./fixtures/routing_table_<slice>_<router>.json). "
                        "Ignored if --routing-table is given.")
    p.add_argument("--schedule-out", default=None, type=Path,
                   help="Where to save the schedule "
                        "(default: ./fixtures/schedule_<slice>_<router_or_loaded>_<order>.json)")
    p.add_argument("--out", default=None, type=Path,
                   help="Output kernel path "
                        "(default: ./pallas_kernel/outputs/_ragged_a2a_kernel_orbit_greedy_<slice>.py)")
    args = p.parse_args(argv)

    slice_ = _parse_slice(args.slice)
    topology = Topology(slice=slice_)
    slice_slug = "x".join(str(s) for s in slice_)        # e.g. "4x4x8" — used in fixtures filenames
    slice_kern = "_".join(str(s) for s in slice_)        # e.g. "4_4_8" — used in kernel filename + function name
    fixtures = _HERE.parent / "fixtures"

    # Stage 1: routing table.
    if args.routing_table is not None:
        if args.router is not None:
            raise SystemExit(
                "Conflict: pass --routing-table OR --router, not both."
            )
        rt_path = args.routing_table
        router_slug = "loaded"           # filename slug
        router_doc = "loaded"            # display-name in kernel docstring
    else:
        router_slug = args.router or "ilp"
        router, router_disp = _build_router(router_slug, topology)
        rt_path = args.routing_table_out or (
            fixtures / f"routing_table_{slice_slug}_{router_slug}.json"
        )
        save_routing_table(topology, router, rt_path)
        print(f"[1/3] wrote routing table {rt_path}", file=sys.stderr)
        router_doc = f"{router_disp}Router"

    table = load_routing_table(rt_path)
    if len(table) != topology.n_nodes:
        raise SystemExit(
            f"routing table {rt_path} has {len(table)} sources; "
            f"slice {slice_} expects {topology.n_nodes}"
        )

    # Stage 2: schedule.
    schedule = schedule_from_orbit_greedy(topology, table, order=args.order)
    sched_path = args.schedule_out or (
        fixtures / f"schedule_{slice_slug}_{router_slug}_{args.order}.json"
    )
    save_schedule(schedule, sched_path)
    print(f"[2/3] wrote schedule     {sched_path}", file=sys.stderr)

    # Stage 3: kernel.
    dest_table, orbit_steps = _dest_table_and_orbit_steps_from_schedule(
        schedule, topology.n_nodes,
    )
    src = generate_kernel_source(
        slice_=slice_,
        router_name_for_doc=router_doc,
        order=args.order,
        per_step_barrier=args.per_step_barrier,
        function_name=args.function_name,
        dest_table=dest_table,
        orbit_steps=orbit_steps,
    )
    out_path = args.out or (
        _HERE / "outputs" / f"_ragged_a2a_kernel_orbit_greedy_{slice_kern}.py"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(src)
    print(f"[3/3] wrote kernel       {out_path} ({len(src):,} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the new pipeline test**

Run: `cd /home/xutingl/collective_comm/TwistedAnalysis && uv run pytest tests/test_gen_orbit_greedy_kernel_pipeline.py -v`
Expected: 3 passed

- [ ] **Step 5: Run the full test suite to confirm no regression**

Run: `cd /home/xutingl/collective_comm/TwistedAnalysis && uv run pytest -x`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
git add pallas_kernel/gen_orbit_greedy_kernel.py tests/test_gen_orbit_greedy_kernel_pipeline.py
git commit -m "refactor(kernel-gen): orchestrate router -> schedule -> kernel; persist intermediates"
```

---

## Task 8: Run the example end-to-end and persist the artifacts

**Files:**
- (No code changes — this task verifies the pipeline produces the expected artifacts and commits them.)

- [ ] **Step 1: Run the example invocation against the existing 4x4x8 fixture**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
uv run python pallas_kernel/gen_orbit_greedy_kernel.py \
    --slice 4,4,8 \
    --routing-table fixtures/routing_table_8x4x4_twist.json \
    --order lpt_tail_asc
```

Expected stderr:
```
[2/3] wrote schedule     fixtures/schedule_4x4x8_loaded_lpt_tail_asc.json
[3/3] wrote kernel       pallas_kernel/outputs/_ragged_a2a_kernel_orbit_greedy_4_4_8.py (...,... bytes)
```

- [ ] **Step 2: Verify the schedule has the expected entry count**

Run:
```bash
cd /home/xutingl/collective_comm/TwistedAnalysis && python3 -c "
import json
data = json.load(open('fixtures/schedule_4x4x8_loaded_lpt_tail_asc.json'))
print('entries:', len(data))
print('first:', data[0])
print('rounds:', len(set(e['round'] for e in data)))
"
```
Expected: `entries: 16256` (= 128 * 127), and `rounds:` between 21 and 30 (matches the OrbitGreedy hop-0 step count for 4x4x8 ILP).

- [ ] **Step 3: Verify the generated kernel parses**

Run:
```bash
cd /home/xutingl/collective_comm/TwistedAnalysis && python3 -c "
import ast
ast.parse(open('pallas_kernel/outputs/_ragged_a2a_kernel_orbit_greedy_4_4_8.py').read())
print('OK')
"
```
Expected: `OK`

- [ ] **Step 4: Commit the example artifacts**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
git add fixtures/schedule_4x4x8_loaded_lpt_tail_asc.json \
        pallas_kernel/outputs/_ragged_a2a_kernel_orbit_greedy_4_4_8.py
git commit -m "fixture: example schedule and kernel from routing_table_8x4x4_twist.json"
```

---

## Task 9: Update READMEs to reflect the new workflow

**Files:**
- Modify: `README.md` (top-level)
- Modify: `pallas_kernel/README.md`

- [ ] **Step 1: Update the top-level README layout section**

Edit `README.md` — replace the `## Layout` section (lines 50-60 in the current file) with this content (preserve everything else verbatim):

```markdown
## Layout

- `twisted_analysis/topology/` — twisted-torus lattice, DOR router, ILPRouter.
- `twisted_analysis/io/` — routing-table and schedule JSON I/O + flat-id utilities.
- `twisted_analysis/model/` — AllToAll workload, link load, lower bound.
- `twisted_analysis/schedules/` — RoundRobin, XLA, DimPhased, OrbitGreedy (headline), PipelinedOrbit (constrained variant), LP-optimal.
- `twisted_analysis/simulator/` — step-synchronous engine + instrumentation.
- `twisted_analysis/lp/` — time-indexed ILP + LP relaxation (PuLP/CBC).
- `twisted_analysis/viz/` — matplotlib plot helpers.
- `fixtures/` — persisted routing tables and schedules (`routing_table_<slice>_<router>.json`, `schedule_<slice>_<router>_<order>.json`); also legacy CSV from `scripts/dump_routing_tables.py`.
- `pallas_kernel/` — Pallas TPU kernel generator (consumes a routing table + schedule, emits `outputs/_ragged_a2a_kernel_orbit_greedy_<slice>.py`).
- `scripts/` — reproducible CLIs:
  - `generate_routing_table.py` — `(slice, router) → fixtures/routing_table_<slice>_<router>.json`
  - `generate_schedule.py` — `(routing-table, scheduler, order) → fixtures/schedule_<slice>_<router>_<order>.json`
- `experiments/` — one YAML per experiment.
- `eval/run_all.sh` — reproduces everything.
- `docs/` — algorithm, topology, schedules, LP, evaluation, results.

## Pipeline

The end-to-end TPU-kernel pipeline runs in three stages, each producing an inspectable on-disk artifact:

1. **Router** → `fixtures/routing_table_<slice>_<router>.json`. Matrix of paths (`[src][dst] → {"path": [{"node_id": int}, ...]}`). Run via `scripts/generate_routing_table.py` or call `twisted_analysis.io.save_routing_table` directly.
2. **Scheduler** → `fixtures/schedule_<slice>_<router>_<order>.json`. Flat list of `{round, src, dst, path}` entries; `path` is a list of flat-IDs from src to dst. Run via `scripts/generate_schedule.py`.
3. **Kernel generator** → `pallas_kernel/outputs/_ragged_a2a_kernel_orbit_greedy_<slice>.py`. Run via `pallas_kernel/gen_orbit_greedy_kernel.py`, which orchestrates stages 1 and 2 (or accepts an existing routing table via `--routing-table FILE`).
```

- [ ] **Step 2: Update the pallas_kernel README**

Edit `pallas_kernel/README.md` — replace the `## Files` table (lines 11-15), the `## Usage` section (lines 86-118), and the `## Generator options reference` table (lines 207-214) with the content below. Other sections (the twist explanation, integration steps, validation plan) remain unchanged.

Replace the `## Files` block with:

```markdown
## Files

| File | Purpose |
|---|---|
| [reference_kernel.py](reference_kernel.py) | Reference `ragged_all_to_all` and `_ragged_a2a_kernel_point_to_point` extracted from `google3/learning/brain/research/megablox/collectives/ragged_all_to_all.py`. The orbit-greedy kernel is a drop-in for the P2P branch only. |
| [gen_orbit_greedy_kernel.py](gen_orbit_greedy_kernel.py) | Pipeline orchestrator. Either generates a routing table (via `--router`) or loads one (via `--routing-table`); generates a schedule from it; emits a kernel `.py` file. Persists the routing table and schedule as inspectable intermediates. |
| `outputs/_ragged_a2a_kernel_orbit_greedy_<slice>.py` | Generator output. One file per (topology, router, order) combination. |
```

Replace the `## Usage` block with:

```markdown
## Usage

The kernel generator runs a 3-stage pipeline. Each stage's artifact is persisted under the project's standard directories so it can be inspected, reused, or regenerated independently.

```
[--router ilp|dor]                  [Stage 1]    fixtures/routing_table_<slice>_<router>.json
                  -- OR --
[--routing-table FILE]              (use existing)
                                          ↓
[--scheduler orbit_greedy --order …]  [Stage 2]    fixtures/schedule_<slice>_<...>_<order>.json
                                          ↓
                                    [Stage 3]    pallas_kernel/outputs/_ragged_a2a_kernel_orbit_greedy_<slice>.py
```

### Generate a kernel by also generating the routing table

Default: slice = 4,4,8, ILP routing, `lpt_tail_asc` order, no per-step barriers.

```bash
python pallas_kernel/gen_orbit_greedy_kernel.py --slice 4,4,8 --router ilp
# [1/3] wrote routing table fixtures/routing_table_4x4x8_ilp.json
# [2/3] wrote schedule     fixtures/schedule_4x4x8_ilp_lpt_tail_asc.json
# [3/3] wrote kernel       pallas_kernel/outputs/_ragged_a2a_kernel_orbit_greedy_4_4_8.py
```

### Generate a kernel from an existing routing table

```bash
python pallas_kernel/gen_orbit_greedy_kernel.py \
    --slice 4,4,8 \
    --routing-table fixtures/routing_table_8x4x4_twist.json
# [2/3] wrote schedule     fixtures/schedule_4x4x8_loaded_lpt_tail_asc.json
# [3/3] wrote kernel       pallas_kernel/outputs/_ragged_a2a_kernel_orbit_greedy_4_4_8.py
```

### Run a single stage

```bash
# Stage 1 only — emit a routing table:
python scripts/generate_routing_table.py --slice 4,4,8 --router ilp

# Stage 2 only — emit a schedule from a routing table:
python scripts/generate_schedule.py \
    --routing-table fixtures/routing_table_8x4x4_twist.json \
    --slice 4,4,8 \
    --scheduler orbit_greedy --order lpt_tail_asc
```

Common variants:

```bash
# DOR routing instead of ILP:
python pallas_kernel/gen_orbit_greedy_kernel.py --slice 4,4,8 --router dor

# Different topology in the {S, 2S}^n family:
python pallas_kernel/gen_orbit_greedy_kernel.py --slice 2,4,4 --router ilp
python pallas_kernel/gen_orbit_greedy_kernel.py --slice 4,8 --router ilp

# Per-step barriers (forces stricter ordering, less pipelining):
python pallas_kernel/gen_orbit_greedy_kernel.py --slice 4,4,8 --router ilp --per-step-barrier
```
```

Replace the `## Generator options reference` block with:

```markdown
## Generator options reference

| Flag | Default | Meaning |
|---|---|---|
| `--slice` | required | Comma-separated topology shape, e.g. `4,4,8`. Must be in `{S, 2S}^n`. |
| `--router` | (when `--routing-table` is absent) `ilp` | `ilp` (load-balanced minimal) or `dor` (dimension-order). Mutually exclusive with `--routing-table`. |
| `--routing-table` | none | Path to an existing routing-table JSON. Skips stage 1; loads paths verbatim. |
| `--order` | `lpt_tail_asc` | OrbitGreedy ordering. `lpt_tail_asc` achieves makespan = LB on every doc cell. |
| `--per-step-barrier` | off | Insert dummy-DMA barriers between OrbitGreedy steps. |
| `--function-name` | `_ragged_a2a_kernel_orbit_greedy_<slice>` | Override the generated function name. |
| `--routing-table-out` | `./fixtures/routing_table_<slice>_<router>.json` | Where to save a generated routing table. |
| `--schedule-out` | `./fixtures/schedule_<slice>_<router_or_loaded>_<order>.json` | Where to save the schedule. |
| `--out` | `./pallas_kernel/outputs/_ragged_a2a_kernel_orbit_greedy_<slice>.py` | Output kernel path. |
```

- [ ] **Step 3: Verify rendered docs read correctly**

Run: `cd /home/xutingl/collective_comm/TwistedAnalysis && grep -n "Pipeline" README.md && grep -n "3-stage" pallas_kernel/README.md`
Expected: at least one match in each file (sanity check that the new sections were applied).

- [ ] **Step 4: Commit**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
git add README.md pallas_kernel/README.md
git commit -m "docs: describe routing-table -> schedule -> kernel pipeline"
```

---

## Final Verification

- [ ] **Step 1: Full test suite**

Run: `cd /home/xutingl/collective_comm/TwistedAnalysis && uv run pytest -x -q`
Expected: all tests pass.

- [ ] **Step 2: Re-run example end-to-end**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
rm -f fixtures/schedule_4x4x8_loaded_lpt_tail_asc.json \
      pallas_kernel/outputs/_ragged_a2a_kernel_orbit_greedy_4_4_8.py
uv run python pallas_kernel/gen_orbit_greedy_kernel.py \
    --slice 4,4,8 \
    --routing-table fixtures/routing_table_8x4x4_twist.json \
    --order lpt_tail_asc
ls -la fixtures/schedule_4x4x8_loaded_lpt_tail_asc.json \
       pallas_kernel/outputs/_ragged_a2a_kernel_orbit_greedy_4_4_8.py
```
Expected: both files re-created.

- [ ] **Step 3: Re-commit if file contents drift**

```bash
cd /home/xutingl/collective_comm/TwistedAnalysis
git add fixtures/schedule_4x4x8_loaded_lpt_tail_asc.json \
        pallas_kernel/outputs/_ragged_a2a_kernel_orbit_greedy_4_4_8.py
git diff --cached --quiet || git commit -m "refresh: regenerate example schedule and kernel"
```
