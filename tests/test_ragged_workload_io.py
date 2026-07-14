"""Workload JSON loader: validation, file-order preservation, fixture truth."""
import json
from pathlib import Path

import pytest

from twisted_analysis.io.routing_table import load_routing_table
from twisted_analysis.io.workload import load_workload

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
WORKLOAD_FIXTURE = FIXTURES / "ragged" / "ragged_a2a_workload_node_128_min_32_max_1024_discrete.json"
ROUTING_FIXTURE = FIXTURES / "routing" / "routing_table_8x4x4_twist.json"


def test_load_small_workload_preserves_file_order(tmp_path):
    p = tmp_path / "w.json"
    p.write_text(json.dumps([
        {"src": 2, "dst": 0, "size": 96},
        {"src": 0, "dst": 1, "size": 32},
    ]))
    w = load_workload(p)
    assert list(w.demand.items()) == [((2, 0), 96), ((0, 1), 32)]
    assert w.quantum == 32


def test_rejects_duplicate_pair(tmp_path):
    p = tmp_path / "w.json"
    p.write_text(json.dumps([
        {"src": 0, "dst": 1, "size": 32},
        {"src": 0, "dst": 1, "size": 64},
    ]))
    with pytest.raises(ValueError, match="duplicate pair"):
        load_workload(p)


def test_rejects_missing_key(tmp_path):
    p = tmp_path / "w.json"
    p.write_text(json.dumps([{"src": 0, "dst": 1}]))
    with pytest.raises(ValueError, match="missing 'size'"):
        load_workload(p)


def test_rejects_non_list_toplevel(tmp_path):
    p = tmp_path / "w.json"
    p.write_text(json.dumps({"src": 0}))
    with pytest.raises(ValueError, match="top-level must be a list"):
        load_workload(p)


def test_fixture_ground_truth():
    """Spec key quantities: 16,256 flows, quantum 32, LB 12,608 = 394 quanta."""
    w = load_workload(WORKLOAD_FIXTURE)
    table = load_routing_table(ROUTING_FIXTURE)
    assert len(w.demand) == 128 * 127
    assert w.quantum == 32
    assert w.lower_bound(table) == 12_608
    assert w.lower_bound(table) // w.quantum == 394
    assert max(
        len(table[s][d]) - 1 for (s, d) in w.demand
    ) == 6


def test_uniform_demand_reproduces_lb_75():
    """Uniform all-pairs size-1 demand through RaggedWorkload matches prior LB."""
    from twisted_analysis.model.ragged import RaggedWorkload

    table = load_routing_table(ROUTING_FIXTURE)
    n = len(table)
    w = RaggedWorkload(demand={
        (s, d): 1 for s in range(n) for d in range(n) if s != d
    })
    assert w.quantum == 1
    assert w.lower_bound(table) == 75
