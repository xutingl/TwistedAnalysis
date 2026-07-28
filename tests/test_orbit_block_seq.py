"""orbit_block_seq scheduler + sliding-window edge-load metric tests.

orbit_block_seq targets the all-up-front (non-pfc) kernel, where no
barrier makes `round` observable and the hardware-concurrent set is a
sliding window of dest-table columns. See the module docstring of
twisted_analysis/schedules/orbit_block_seq.py for why that is the
objective and why a flat greedy on it fails.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from math import ceil
from pathlib import Path

import pytest

from twisted_analysis.io.routing_table import load_routing_table, save_routing_table
from twisted_analysis.io.schedule import schedule_from_algorithm
from twisted_analysis.schedules.orbit_block_seq import _block_sizes, orbit_block_seq
from twisted_analysis.schedules.verify import (
    max_window_edge_load,
    schedule_step_count,
)
from twisted_analysis.topology import ILPRouter, Topology

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
LB_TOTAL = 75  # max physical-edge load of the loaded 8x4x4 routing


def _loaded_8x4x4():
    topology = Topology(slice=(8, 4, 4))
    table = load_routing_table(
        FIXTURES / "routing" / "routing_table_8x4x4_twist.json"
    )
    return topology, table


# ---------------------------------------------------------------------------
# Block sizing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,w", [(127, 6), (127, 12), (127, 5), (255, 6), (31, 4)])
def test_block_sizes_all_at_least_w_and_sum_exactly(n, w):
    # Every block >= w is what caps a w-window straddle at TWO blocks.
    # A runt block would reintroduce the three-block straddle that makes
    # orbit_pack's ragged bins score 3+3+3=9 at w=6.
    sizes = _block_sizes(n, w)
    assert sum(sizes) == n
    assert all(s >= w for s in sizes)
    assert max(sizes) - min(sizes) <= 1


def test_block_sizes_rejects_w_above_item_count():
    with pytest.raises(ValueError):
        _block_sizes(5, 6)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def test_orbit_block_seq_full_coverage_loaded_8x4x4():
    topology, table = _loaded_8x4x4()
    schedule = orbit_block_seq(topology, table, w=12)
    n = topology.n_nodes
    assert len(schedule) == n * (n - 1)
    assert len({(e["src"], e["dst"]) for e in schedule}) == n * (n - 1)
    assert not any(e["src"] == e["dst"] for e in schedule)
    per_src = Counter(e["src"] for e in schedule)
    per_dst = Counter(e["dst"] for e in schedule)
    assert all(per_src[s] == n - 1 for s in range(n))
    assert all(per_dst[d] == n - 1 for d in range(n))


def test_orbit_block_seq_one_orbit_per_round_loaded_8x4x4():
    # Column k must be orbit k for EVERY source. Sharing a round across a
    # block would let the codegen's (round, dst) sort order the block
    # differently per source, scrambling the optimised sequence.
    topology, table = _loaded_8x4x4()
    schedule = orbit_block_seq(topology, table, w=12)
    n = topology.n_nodes
    assert schedule_step_count(schedule) == n - 1
    per_round = Counter(e["round"] for e in schedule)
    assert set(per_round.values()) == {n}          # one full permutation each
    assert sorted(per_round) == list(range(n - 1))  # contiguous from 0


def test_orbit_block_seq_translation_symmetry_loaded_8x4x4():
    topology, table = _loaded_8x4x4()
    schedule = orbit_block_seq(topology, table, w=12)
    by_src = defaultdict(list)
    for e in schedule:
        by_src[e["src"]].append(e["round"])
    for s in by_src:
        by_src[s].sort()
    assert all(by_src[s] == by_src[0] for s in by_src)


def test_orbit_block_seq_is_deterministic_loaded_8x4x4():
    topology, table = _loaded_8x4x4()
    assert (orbit_block_seq(topology, table, w=12)
            == orbit_block_seq(topology, table, w=12))


@pytest.mark.parametrize("w", [0, -1, "6", 6.0])
def test_orbit_block_seq_rejects_bad_w(w):
    topology, table = _loaded_8x4x4()
    with pytest.raises(ValueError):
        orbit_block_seq(topology, table, w=w)


# ---------------------------------------------------------------------------
# The point of the algorithm: beat orbit_greedy_full at every window
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("window", [6, 12, 24, 48])
def test_orbit_block_seq_beats_orbitfull_at_every_window(window):
    # Locks the headline result. A flat least-burstiness-next greedy was
    # tried and only beat orbitfull at its tuning window while regressing
    # elsewhere; the uniform-block construction must not regress anywhere.
    topology, table = _loaded_8x4x4()
    block = orbit_block_seq(topology, table, w=12)
    from twisted_analysis.io.schedule import load_schedule
    orbitfull = load_schedule(
        FIXTURES / "nonragged"
        / "schedule_8x4x4_loaded_orbit_greedy_full_lpt_tail_asc.json"
    )
    assert (max_window_edge_load(block, window)
            < max_window_edge_load(orbitfull, window))


def test_orbit_block_seq_respects_the_window_lower_bound():
    # LB(w) = ceil(w * 75 / 127); equality is forced at w = N-1 since the
    # window is then the whole collective.
    topology, table = _loaded_8x4x4()
    schedule = orbit_block_seq(topology, table, w=12)
    n = topology.n_nodes
    for w in (6, 12, 24, 48):
        assert max_window_edge_load(schedule, w) >= ceil(w * LB_TOTAL / (n - 1))
    assert max_window_edge_load(schedule, n - 1) == LB_TOTAL


def test_orbit_block_seq_small_cell_2x4(tmp_path):
    topology = Topology(slice=(2, 4))
    router = ILPRouter(topology=topology)
    rt = tmp_path / "rt.json"
    save_routing_table(topology, router, rt)
    table = load_routing_table(rt)
    schedule = orbit_block_seq(topology, table, w=2)
    n = topology.n_nodes
    assert len(schedule) == n * (n - 1)
    assert schedule_step_count(schedule) == n - 1


def test_orbit_block_seq_via_dispatcher_and_cli(tmp_path):
    import json
    import subprocess
    import sys

    topology, table = _loaded_8x4x4()
    schedule = schedule_from_algorithm("orbit_block_seq", topology, table, w=12)
    assert len(schedule) == 128 * 127

    repo = Path(__file__).resolve().parent.parent
    out = tmp_path / "sched.json"
    res = subprocess.run(
        [sys.executable, str(repo / "scripts" / "generate_schedule.py"),
         "--routing-table",
         str(repo / "fixtures" / "routing" / "routing_table_8x4x4_twist.json"),
         "--slice", "8,4,4",
         "--scheduler", "orbit_block_seq", "--w", "12",
         "--out", str(out)],
        cwd=str(repo), capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr
    assert json.loads(out.read_text()) == schedule


# ---------------------------------------------------------------------------
# The metric itself
# ---------------------------------------------------------------------------

def test_max_window_edge_load_unions_across_the_window():
    # Every column on its own peaks at 1, but edge (0,1) appears in BOTH
    # columns, so a width-2 window carries 2. This is the whole point of
    # the metric: per-round checks miss it, sliding windows do not.
    schedule = [
        {"round": 0, "src": 0, "dst": 1, "path": [0, 1]},
        {"round": 1, "src": 0, "dst": 2, "path": [0, 1, 2]},
        {"round": 0, "src": 3, "dst": 2, "path": [3, 2]},
        {"round": 1, "src": 3, "dst": 1, "path": [3, 1]},
    ]
    assert max_window_edge_load(schedule, 1) == 1
    assert max_window_edge_load(schedule, 2) == 2


def test_max_window_edge_load_sums_across_sources_within_a_column():
    # Both sources traverse (1, 2) in column 0, so that column alone is 2
    # even at window width 1.
    schedule = [
        {"round": 0, "src": 0, "dst": 2, "path": [0, 1, 2]},
        {"round": 0, "src": 3, "dst": 2, "path": [3, 1, 2]},
    ]
    assert max_window_edge_load(schedule, 1) == 2


def test_max_window_edge_load_uses_round_order_not_input_order():
    # Entries deliberately out of round order; column 0 must still be the
    # round-0 flow, matching how the codegen sorts (round, dst).
    shuffled_input = [
        {"round": 1, "src": 0, "dst": 2, "path": [0, 1, 2]},
        {"round": 0, "src": 0, "dst": 1, "path": [0, 5]},
    ]
    assert max_window_edge_load(shuffled_input, 1) == 1
    assert max_window_edge_load(shuffled_input, 2) == 1


def test_max_window_edge_load_rejects_bad_w_and_handles_empty():
    assert max_window_edge_load([], 3) == 0
    with pytest.raises(ValueError):
        max_window_edge_load([], 0)
