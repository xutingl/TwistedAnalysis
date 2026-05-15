"""Routing-table on-disk I/O and RoutingTableRouter adapter.

On-disk format (matches fixtures/routing_table_4x4x8_twist.json shape, with
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

from twisted_analysis.io.coords import flatten
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
            assert node_ids[-1] == dst_flat, (
                f"router.path({src}, {dst}) endpoint = flat {node_ids[-1]}, "
                f"expected {dst_flat}"
            )
            matrix[src_flat][dst_flat] = {
                "path": [{"node_id": nid} for nid in node_ids]
            }
    out_path.write_text(json.dumps(matrix, indent=2))


def load_routing_table(path: Path | str) -> list[list[list[int]]]:
    """Load a routing-table JSON file. Returns `table[src][dst] = [int, ...]`.

    Tolerates a `vc` field on path nodes if present (it is dropped).
    """
    path = Path(path)
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(
            f"{path}: top-level must be a list, got {type(raw).__name__}"
        )
    n = len(raw)
    table: list[list[list[int]]] = []
    for src in range(n):
        row = raw[src]
        if not isinstance(row, list):
            raise ValueError(
                f"{path}: row {src} must be a list, got {type(row).__name__}"
            )
        if len(row) != n:
            raise ValueError(
                f"{path}: row {src} has length {len(row)}; expected {n}"
            )
        out_row = []
        for dst in range(n):
            cell = row[dst]
            if not isinstance(cell, dict):
                raise ValueError(
                    f"{path}: cell [{src}][{dst}] must be a dict, "
                    f"got {type(cell).__name__}"
                )
            if "path" not in cell:
                raise ValueError(
                    f"{path}: cell [{src}][{dst}] missing 'path'"
                )
            path_nodes = cell["path"]
            if not isinstance(path_nodes, list):
                raise ValueError(
                    f"{path}: cell [{src}][{dst}] 'path' must be a list, "
                    f"got {type(path_nodes).__name__}"
                )
            node_ids: list[int] = []
            for i, node in enumerate(path_nodes):
                if not isinstance(node, dict):
                    raise ValueError(
                        f"{path}: cell [{src}][{dst}] path entry {i} must be "
                        f"a dict, got {type(node).__name__}"
                    )
                if "node_id" not in node:
                    raise ValueError(
                        f"{path}: cell [{src}][{dst}] path entry {i} missing "
                        f"'node_id'"
                    )
                node_ids.append(node["node_id"])
            out_row.append(node_ids)
        table.append(out_row)
    return table


@dataclass
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
