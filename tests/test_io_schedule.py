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


def test_load_schedule_rejects_non_dict_entry(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps([42]))
    with pytest.raises(ValueError, match="not a dict"):
        load_schedule(p)


def test_save_schedule_rejects_non_int_round(tmp_path: Path):
    bad = [{"round": "0", "src": 0, "dst": 1, "path": [0, 1]}]
    with pytest.raises(ValueError, match="round"):
        save_schedule(bad, tmp_path / "x.json")


def test_save_schedule_rejects_non_int_path_element(tmp_path: Path):
    bad = [{"round": 0, "src": 0, "dst": 1, "path": [0, "1"]}]
    with pytest.raises(ValueError, match="path"):
        save_schedule(bad, tmp_path / "x.json")


def test_save_schedule_rejects_bool_round(tmp_path: Path):
    """Python bool is a subtype of int; explicitly reject."""
    bad = [{"round": True, "src": 0, "dst": 1, "path": [0, 1]}]
    with pytest.raises(ValueError, match="round"):
        save_schedule(bad, tmp_path / "x.json")


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


def test_schedule_from_orbit_greedy_round_structure_2x4_dor():
    """Round values must reflect _emit_orbit_greedy's hop-0 firing times,
    not be a constant. Locks in the regression where round always = 0."""
    from twisted_analysis.io.schedule import schedule_from_orbit_greedy
    from twisted_analysis.io.coords import flatten
    from twisted_analysis.topology import Topology, DORRouter

    t = Topology(slice=(2, 4))
    r = DORRouter(t)
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
    rounds = sorted({e["round"] for e in entries})
    assert rounds[0] == 0, "at least one orbit must fire at round 0"
    assert max(rounds) >= 1, "schedule must span multiple rounds"
    # 2x4 DOR has multiple orbits at round 0 — confirm by checking that at
    # least two distinct (src, dst) flow pairs share round 0.
    round0 = [(e["src"], e["dst"]) for e in entries if e["round"] == 0]
    distinct_orbits = len({(e[1] - e[0]) % t.n_nodes for e in round0})
    assert distinct_orbits >= 2, (
        f"expected >=2 distinct orbits at round 0, got {distinct_orbits}"
    )


def test_schedule_from_orbit_greedy_rejects_none_cell():
    """Malformed routing table (None cell) must surface as a debuggable ValueError."""
    from twisted_analysis.io.schedule import schedule_from_orbit_greedy
    from twisted_analysis.topology import Topology
    t = Topology(slice=(2, 4))
    n = t.n_nodes
    table = [[[i] if i == j else [i, j] for j in range(n)] for i in range(n)]
    table[0][1] = None  # poison one cell
    with pytest.raises(ValueError, match=r"\[0\]\[1\]"):
        schedule_from_orbit_greedy(t, table)


def test_schedule_from_orbit_greedy_rejects_wrong_shape():
    from twisted_analysis.io.schedule import schedule_from_orbit_greedy
    from twisted_analysis.topology import Topology
    t = Topology(slice=(2, 4))  # n=8
    table = [[[0]] * 7 for _ in range(8)]  # rows have 7 cols, expected 8
    with pytest.raises(ValueError, match="cols"):
        schedule_from_orbit_greedy(t, table)
