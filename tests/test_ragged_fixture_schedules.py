"""Regression guard: the committed ragged fixture schedules stay verifier-clean
and their makespans match the README's published numbers (LB = 394 quanta;
preemptive lpt achieves LB exactly)."""
from pathlib import Path

import pytest

from twisted_analysis.io.schedule import load_schedule
from twisted_analysis.io.workload import load_workload
from twisted_analysis.schedules.verify import (
    schedule_makespan_ragged,
    verify_capacity_ragged,
    verify_workload_coverage,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
WORKLOAD = FIXTURES / "ragged_a2a_workload_node_128_min_32_max_1024_discrete.json"

EXPECTED = [
    ("schedule_8x4x4_loaded_ragged_fluid.json", 399.0),
    ("schedule_8x4x4_loaded_ragged_greedy_lpt.json", 410.0),
    ("schedule_8x4x4_loaded_ragged_greedy_spt.json", 588.0),
    ("schedule_8x4x4_loaded_ragged_greedy_natural.json", 540.0),
    ("schedule_8x4x4_loaded_ragged_greedy_lpt_pre.json", 394.0),
]


@pytest.mark.parametrize("fixture_name,expected_makespan", EXPECTED)
def test_committed_fixture_is_clean_and_matches_readme(fixture_name, expected_makespan):
    sched = load_schedule(FIXTURES / fixture_name)
    workload = load_workload(WORKLOAD)
    assert verify_capacity_ragged(sched, quantum=32) == []
    assert verify_workload_coverage(sched, workload) == []
    assert abs(schedule_makespan_ragged(sched, quantum=32) - expected_makespan) < 1e-6
