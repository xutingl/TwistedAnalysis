import pytest

from twisted_analysis.schedules.verify import (
    CapacityViolation,
    schedule_makespan,
    verify_capacity,
)


def test_verify_no_violations_on_disjoint_paths():
    schedule = [
        {"round": 0, "src": 0, "dst": 1, "path": [0, 1]},
        {"round": 0, "src": 2, "dst": 3, "path": [2, 3]},
    ]
    assert verify_capacity(schedule) == []


def test_verify_detects_same_edge_same_time_violation():
    schedule = [
        # Two flows both use edge (1->2) at time t=1.
        {"round": 0, "src": 0, "dst": 2, "path": [0, 1, 2]},
        {"round": 1, "src": 1, "dst": 2, "path": [1, 2]},
    ]
    violations = verify_capacity(schedule)
    assert len(violations) == 1
    v = violations[0]
    assert v.edge == (1, 2)
    assert v.time == 1
    assert len(v.flows) == 2


def test_verify_same_edge_different_time_ok():
    schedule = [
        # Same edge but at different absolute times (0 vs 2).
        {"round": 0, "src": 0, "dst": 1, "path": [0, 1]},
        {"round": 2, "src": 0, "dst": 1, "path": [0, 1]},
    ]
    assert verify_capacity(schedule) == []


def test_schedule_makespan():
    schedule = [
        {"round": 0, "src": 0, "dst": 2, "path": [0, 1, 2]},   # uses t=0, t=1
        {"round": 3, "src": 5, "dst": 6, "path": [5, 6]},      # uses t=3
    ]
    # Latest hop fires at t=3, makespan = 4.
    assert schedule_makespan(schedule) == 4
