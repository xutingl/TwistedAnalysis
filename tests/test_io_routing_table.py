"""Routing-table save/load + RoutingTableRouter adapter.

On-disk shape (matches fixtures/routing_table_4x4x8_twist.json minus vc):
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
    # Mimic the existing routing_table_4x4x8_twist.json shape (with vc).
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


def test_load_routing_table_rejects_non_list_top_level(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"not": "a list"}))
    with pytest.raises(ValueError, match="top-level"):
        load_routing_table(p)


def test_load_routing_table_rejects_non_dict_cell(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps([[None]]))
    with pytest.raises(ValueError):
        load_routing_table(p)


def test_load_routing_table_rejects_path_node_missing_node_id(tmp_path: Path):
    p = tmp_path / "bad.json"
    # Valid 1x1 shape, but node entry has no node_id field.
    p.write_text(json.dumps([[{"path": [{"vc": 0}]}]]))
    with pytest.raises(ValueError, match="node_id"):
        load_routing_table(p)


def test_save_routing_table_asserts_router_endpoint(tmp_path: Path):
    """Writer post-condition: detects router returning a wrong-endpoint path."""
    from twisted_analysis.topology import Topology, DORRouter
    from twisted_analysis.topology.lattice import DirectedLink
    t = Topology(slice=(2, 4))
    real = DORRouter(t)

    class BogusRouter:
        topology = t
        def path(self, src, dst):
            # Always return src->src->src (endpoint != dst when src != dst).
            if src == dst:
                return ()
            # Walk one neighbor and stop early.
            for dim in range(t.ndim):
                v = t.neighbor(src, dim, +1)
                return ((src, v, dim, +1),)  # endpoint = a neighbor, not dst
            return ()

    out = tmp_path / "rt.json"
    with pytest.raises(AssertionError, match="endpoint"):
        save_routing_table(t, BogusRouter(), out)
