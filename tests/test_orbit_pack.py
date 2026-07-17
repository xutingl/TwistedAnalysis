"""orbit_pack scheduler + step-model capacity verifier tests.

orbit_pack targets the barrier-delimited execution model of the
`--per-step-barrier` (TPU v4 / pfc) kernel: each round is a synchronized
step; all DMAs of a step are in flight together (whole-path edge
occupancy), and cross-step link interactions are serialized by the
barrier. This differs from `verify_capacity`'s staggered-hop model
(hop i fires at round + i), which matches neither hardware execution
mode. See docs/orbit_greedy_optimality.md §6 for the model reconciliation.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from math import ceil
from pathlib import Path

import pytest

from twisted_analysis.io.routing_table import load_routing_table, save_routing_table
from twisted_analysis.io.schedule import schedule_from_algorithm
from twisted_analysis.schedules.orbit_pack import orbit_pack
from twisted_analysis.schedules.verify import (
    schedule_step_count,
    verify_capacity_step,
)
from twisted_analysis.topology import Topology, ILPRouter

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _loaded_8x4x4():
    topology = Topology(slice=(8, 4, 4))
    table = load_routing_table(
        FIXTURES / "routing" / "routing_table_8x4x4_twist.json"
    )
    return topology, table


# ---------------------------------------------------------------------------
# Step-model verifier
# ---------------------------------------------------------------------------

def test_verify_step_flags_whole_path_edge_overload():
    # Two same-round flows share edge (1, 2) at DIFFERENT hop indices.
    # Staggered model would allow this; the step model must flag it at cap 1.
    schedule = [
        {"round": 0, "src": 0, "dst": 2, "path": [0, 1, 2]},
        {"round": 0, "src": 1, "dst": 3, "path": [1, 2, 3]},
    ]
    violations = verify_capacity_step(schedule, max_edge_load=1)
    assert len(violations) == 1
    v = violations[0]
    assert v.kind == "edge"
    assert v.round == 0
    assert v.key == (1, 2)
    assert v.load == 2
    assert verify_capacity_step(schedule, max_edge_load=2) == []


def test_verify_step_different_rounds_never_collide():
    schedule = [
        {"round": 0, "src": 0, "dst": 2, "path": [0, 1, 2]},
        {"round": 1, "src": 1, "dst": 3, "path": [1, 2, 3]},
    ]
    assert verify_capacity_step(schedule, max_edge_load=1) == []


def test_verify_step_flags_device_send_and_recv_overload():
    # Device 0 sends twice in round 0; device 3 receives twice in round 0.
    # Edge-disjoint paths so only the device caps can fire.
    schedule = [
        {"round": 0, "src": 0, "dst": 1, "path": [0, 1]},
        {"round": 0, "src": 0, "dst": 3, "path": [0, 3]},
        {"round": 0, "src": 2, "dst": 3, "path": [2, 3]},
    ]
    violations = verify_capacity_step(
        schedule, max_edge_load=8, max_dmas_per_device=1,
    )
    kinds = {(v.kind, v.key) for v in violations}
    assert ("send", 0) in kinds
    assert ("recv", 3) in kinds
    assert all(v.round == 0 for v in violations)
    assert verify_capacity_step(
        schedule, max_edge_load=8, max_dmas_per_device=2,
    ) == []


def test_schedule_step_count_counts_distinct_rounds():
    schedule = [
        {"round": 0, "src": 0, "dst": 1, "path": [0, 1]},
        {"round": 0, "src": 1, "dst": 2, "path": [1, 2]},
        {"round": 5, "src": 2, "dst": 3, "path": [2, 3]},
    ]
    assert schedule_step_count(schedule) == 2


# ---------------------------------------------------------------------------
# orbit_pack scheduler
# ---------------------------------------------------------------------------

def test_orbit_pack_full_coverage_and_translation_symmetry_loaded_8x4x4():
    topology, table = _loaded_8x4x4()
    schedule = orbit_pack(topology, table, k=2, c=3)
    n = topology.n_nodes
    pairs = {(e["src"], e["dst"]) for e in schedule}
    assert len(schedule) == n * (n - 1)
    assert len(pairs) == n * (n - 1)
    # Translation symmetry: every source sees the same sorted round list
    # (the --per-step-barrier codegen requirement).
    by_src = defaultdict(list)
    for e in schedule:
        by_src[e["src"]].append(e["round"])
    for s in by_src:
        by_src[s].sort()
    assert all(by_src[s] == by_src[0] for s in by_src)


@pytest.mark.parametrize("k,c", [(2, 3), (3, 3), (6, 3)])
def test_orbit_pack_respects_caps_loaded_8x4x4(k, c):
    topology, table = _loaded_8x4x4()
    schedule = orbit_pack(topology, table, k=k, c=c)
    assert verify_capacity_step(
        schedule, max_edge_load=c, max_dmas_per_device=k,
    ) == []


@pytest.mark.parametrize("k,c,expected_steps", [(2, 3, 64), (3, 3, 43), (6, 3, 27)])
def test_orbit_pack_step_counts_loaded_8x4x4(k, c, expected_steps):
    # Locks the deterministic FFD packing on the loaded routing. Update
    # these numbers (and the README/cns readme rows) if the packing
    # heuristic changes intentionally.
    topology, table = _loaded_8x4x4()
    schedule = orbit_pack(topology, table, k=k, c=c)
    assert schedule_step_count(schedule) == expected_steps
    assert schedule_step_count(schedule) >= ceil((topology.n_nodes - 1) / k)


def test_orbit_pack_rounds_are_contiguous_from_zero_loaded_8x4x4():
    # The pfc kernel executes distinct rounds back-to-back; emit them as
    # 0..T-1 so round values == barrier-step indices.
    topology, table = _loaded_8x4x4()
    schedule = orbit_pack(topology, table, k=6, c=3)
    rounds = sorted({e["round"] for e in schedule})
    assert rounds == list(range(len(rounds)))


def test_orbit_pack_rejects_infeasible_c_loaded_8x4x4():
    # 26 orbits on this routing have whole-path self-load 3; c=2 is
    # unsatisfiable no matter the packing.
    topology, table = _loaded_8x4x4()
    with pytest.raises(ValueError, match="c=2"):
        orbit_pack(topology, table, k=2, c=2)


def test_orbit_pack_rejects_bad_k_and_c():
    topology, table = _loaded_8x4x4()
    with pytest.raises(ValueError):
        orbit_pack(topology, table, k=0, c=3)
    with pytest.raises(ValueError):
        orbit_pack(topology, table, k=2, c=0)


def test_orbit_pack_small_cell_2x4(tmp_path):
    topology = Topology(slice=(2, 4))
    router = ILPRouter(topology=topology)
    rt_path = tmp_path / "rt.json"
    save_routing_table(topology, router, rt_path)
    table = load_routing_table(rt_path)
    schedule = orbit_pack(topology, table, k=2, c=3)
    n = topology.n_nodes
    assert len(schedule) == n * (n - 1)
    assert verify_capacity_step(
        schedule, max_edge_load=3, max_dmas_per_device=2,
    ) == []


def test_generate_schedule_cli_orbit_pack(tmp_path):
    import json
    import subprocess
    import sys

    repo = Path(__file__).resolve().parent.parent
    out = tmp_path / "sched.json"
    res = subprocess.run(
        [sys.executable, str(repo / "scripts" / "generate_schedule.py"),
         "--routing-table",
         str(repo / "fixtures" / "routing" / "routing_table_8x4x4_twist.json"),
         "--slice", "8,4,4",
         "--scheduler", "orbit_pack", "--k", "6", "--c", "3",
         "--out", str(out)],
        cwd=str(repo), capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr
    entries = json.loads(out.read_text())
    assert len(entries) == 128 * 127
    assert len({e["round"] for e in entries}) == 27


def test_orbit_pack_via_dispatcher(tmp_path):
    topology = Topology(slice=(2, 4))
    router = ILPRouter(topology=topology)
    rt_path = tmp_path / "rt.json"
    save_routing_table(topology, router, rt_path)
    table = load_routing_table(rt_path)
    schedule = schedule_from_algorithm("orbit_pack", topology, table, k=2, c=3)
    n = topology.n_nodes
    assert len(schedule) == n * (n - 1)
