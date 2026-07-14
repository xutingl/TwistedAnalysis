"""Integral earliest-feasible greedy: both variants, orders, fixture subsample."""
import json
from itertools import islice
from pathlib import Path

import pytest

from twisted_analysis.io.routing_table import load_routing_table
from twisted_analysis.io.workload import load_workload
from twisted_analysis.model.ragged import RaggedWorkload
from twisted_analysis.schedules.ragged_greedy import ragged_greedy
from twisted_analysis.schedules.verify import (
    schedule_makespan_ragged,
    verify_capacity_ragged,
    verify_workload_coverage,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"

LINE_TABLE = [
    [[0], [0, 1], [0, 1, 2]],
    [[1, 0], [1], [1, 2]],
    [[2, 1, 0], [2, 1], [2]],
]

# 4-node line 0-1-2-3 (only the cells the tests touch need to be right).
LINE4_TABLE = [
    [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3]],
    [[1, 0], [1], [1, 2], [1, 2, 3]],
    [[2, 1, 0], [2, 1], [2], [2, 3]],
    [[3, 2, 1, 0], [3, 2, 1], [3, 2], [3]],
]


def test_rejects_unknown_order():
    w = RaggedWorkload(demand={(0, 1): 32})
    with pytest.raises(ValueError, match="order"):
        ragged_greedy(LINE_TABLE, w, order="bogus")


def test_nonpreemptive_disjoint_flows_both_start_at_zero():
    """(0,2) occupies (0,1)@{0,1}, (1,2)@{1,2}; (1,2)'s single quantum fits
    at t=0 on edge (1,2) — hop-offset pipelining interleaves them."""
    w = RaggedWorkload(demand={(0, 2): 64, (1, 2): 32})
    sched = ragged_greedy(LINE_TABLE, w, order="lpt")
    by_pair = {(e["src"], e["dst"]): e for e in sched}
    assert by_pair[(0, 2)]["round"] == 0
    assert by_pair[(1, 2)]["round"] == 0
    assert all(e["rate"] == 1.0 for e in sched)
    assert verify_capacity_ragged(sched, quantum=32) == []
    assert verify_workload_coverage(sched, w) == []
    assert schedule_makespan_ragged(sched, quantum=32) == 3.0


def test_nonpreemptive_conflicting_flow_waits():
    """(0,2) size 64 takes (0,1)@{0,1}; (0,1) size 32 must wait until t=2."""
    w = RaggedWorkload(demand={(0, 2): 64, (0, 1): 32})
    sched = ragged_greedy(LINE_TABLE, w, order="lpt")
    by_pair = {(e["src"], e["dst"]): e for e in sched}
    assert by_pair[(0, 2)]["round"] == 0
    assert by_pair[(0, 1)]["round"] == 2
    assert verify_capacity_ragged(sched, quantum=32) == []


def test_preemptive_splits_around_busy_slot():
    """Natural order on LINE4: (0,3) size 32 first marks (1,2)@1. Then (1,2)
    size 96 (3 quanta) can use t=0 but not t=1 -> chunks [0] and [2,3]."""
    w = RaggedWorkload(demand={(0, 3): 32, (1, 2): 96})
    sched = ragged_greedy(LINE4_TABLE, w, order="natural", preemptive=True)
    chunks_12 = sorted(
        (e["round"], e["size"]) for e in sched
        if (e["src"], e["dst"]) == (1, 2)
    )
    assert chunks_12 == [(0, 32), (2, 64)]
    assert verify_capacity_ragged(sched, quantum=32) == []
    assert verify_workload_coverage(sched, w) == []
    # Non-preemptive on the same workload must wait for 3 contiguous slots.
    ns = ragged_greedy(LINE4_TABLE, w, order="natural", preemptive=False)
    by_pair = {(e["src"], e["dst"]): e for e in ns}
    assert by_pair[(1, 2)]["round"] == 2
    assert schedule_makespan_ragged(sched, quantum=32) <= \
        schedule_makespan_ragged(ns, quantum=32)


def _subsample_workload(n_flows):
    raw = json.loads((
        FIXTURES / "ragged_a2a_workload_node_128_min_32_max_1024_discrete.json"
    ).read_text())
    return RaggedWorkload(demand={
        (e["src"], e["dst"]): e["size"] for e in islice(raw, n_flows)
    })


@pytest.mark.parametrize("preemptive", [False, True])
def test_fixture_subsample_feasible(preemptive):
    """Spec test 4 at reduced scale (first 300 flows) to keep runtime low."""
    w = _subsample_workload(300)
    table = load_routing_table(FIXTURES / "routing_table_8x4x4_twist.json")
    sched = ragged_greedy(table, w, order="lpt", preemptive=preemptive)
    assert verify_capacity_ragged(sched, quantum=w.quantum) == []
    assert verify_workload_coverage(sched, w) == []
    if not preemptive:
        assert len(sched) == 300  # exactly one entry per flow
    else:
        assert len(sched) >= 300


def test_fixture_subsample_preemptive_no_worse():
    w = _subsample_workload(300)
    table = load_routing_table(FIXTURES / "routing_table_8x4x4_twist.json")
    ms_non = schedule_makespan_ragged(
        ragged_greedy(table, w, order="lpt", preemptive=False),
        quantum=w.quantum,
    )
    ms_pre = schedule_makespan_ragged(
        ragged_greedy(table, w, order="lpt", preemptive=True),
        quantum=w.quantum,
    )
    assert ms_pre <= ms_non
