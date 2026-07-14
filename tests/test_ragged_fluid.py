"""Closed-form water-filling schedule: feasibility and LB-tightness."""
from pathlib import Path

from twisted_analysis.io.routing_table import load_routing_table
from twisted_analysis.io.workload import load_workload
from twisted_analysis.model.ragged import RaggedWorkload
from twisted_analysis.schedules.ragged_fluid import ragged_fluid
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


def test_small_case_rates_and_makespan():
    w = RaggedWorkload(demand={(0, 2): 64, (1, 2): 32})
    sched = ragged_fluid(LINE_TABLE, w)
    assert len(sched) == 2
    by_pair = {(e["src"], e["dst"]): e for e in sched}
    assert by_pair[(0, 2)]["rate"] == 64 / 96
    assert by_pair[(1, 2)]["rate"] == 32 / 96
    assert all(e["round"] == 0 for e in sched)
    assert verify_capacity_ragged(sched, quantum=32) == []
    assert verify_workload_coverage(sched, w) == []
    # Every flow streams for LB = 3 quanta; the 2-hop flow adds 1 fill quantum.
    assert abs(schedule_makespan_ragged(sched, quantum=32) - 4.0) < 1e-9


def test_fixture_scale_lb_certificate():
    """Spec test 3: fluid schedule is feasible at makespan 399 = LB + 5."""
    w = load_workload(
        FIXTURES / "ragged_a2a_workload_node_128_min_32_max_1024_discrete.json"
    )
    table = load_routing_table(FIXTURES / "routing_table_8x4x4_twist.json")
    sched = ragged_fluid(table, w)
    assert len(sched) == len(w.demand)  # one entry per flow
    assert all(0 < e["rate"] <= 1 for e in sched)
    assert verify_capacity_ragged(sched, quantum=32) == []
    assert verify_workload_coverage(sched, w) == []
    ms = schedule_makespan_ragged(sched, quantum=32)
    assert abs(ms - 399.0) < 1e-6
