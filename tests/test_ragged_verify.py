"""Ragged verifier: pipelined-stream time model, sweepline capacity, coverage."""
from pathlib import Path

from twisted_analysis.io.schedule import load_schedule
from twisted_analysis.model.ragged import RaggedWorkload
from twisted_analysis.schedules.verify import (
    schedule_makespan,
    schedule_makespan_ragged,
    verify_capacity,
    verify_capacity_ragged,
    verify_workload_coverage,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _chunk(round_, src, dst, path, rate, size):
    return {"round": round_, "src": src, "dst": dst, "path": path,
            "rate": rate, "size": size}


def test_time_model_single_chunk():
    """Spec test 6: 2-hop chunk (m=4 quanta, rate=0.5) at r=0, quantum=32:
    edge 0 occupied [0, 8), edge 1 occupied [1, 9); finish = 0 + 1 + 8 = 9."""
    sched = [_chunk(0, 0, 2, [0, 1, 2], 0.5, 128)]
    assert verify_capacity_ragged(sched, quantum=32) == []
    assert schedule_makespan_ragged(sched, quantum=32) == 9.0


def test_capacity_violation_detected():
    """Chunk A holds edge (1,2) at rate 0.5 over [1, 9); chunk B holds it at
    rate 0.6 over [0, 10/3). Overlap [1, 10/3) sums to 1.1 > 1."""
    sched = [
        _chunk(0, 0, 2, [0, 1, 2], 0.5, 128),
        _chunk(0, 1, 2, [1, 2], 0.6, 64),
    ]
    violations = verify_capacity_ragged(sched, quantum=32)
    assert len(violations) == 1
    v = violations[0]
    assert v.edge == (1, 2)
    assert v.time == 1.0
    assert abs(v.total_rate - 1.1) < 1e-9
    assert v.flows == ((0, 0, 2), (0, 1, 2))


def test_rates_summing_to_one_are_feasible():
    sched = [
        _chunk(0, 0, 2, [0, 1, 2], 0.5, 128),
        _chunk(0, 1, 2, [1, 2], 0.5, 64),
    ]
    assert verify_capacity_ragged(sched, quantum=32) == []


def test_half_open_intervals_do_not_collide():
    """A ends on edge (0,1) exactly when B starts: [0,2) then [2,3) is legal
    at full rate."""
    sched = [
        _chunk(0, 0, 1, [0, 1], 1.0, 64),
        _chunk(2, 0, 1, [0, 1], 1.0, 32),
    ]
    # Both entries are the same (src, dst) pair; capacity is what's under test.
    assert verify_capacity_ragged(sched, quantum=32) == []


def test_simultaneous_overlap_reports_single_complete_violation():
    """5 chunks at rate 0.3 starting together on one edge: exactly one
    violation carrying the true total (1.5) and all five flows."""
    sched = [_chunk(0, i, 9, [0, 1], 0.3, 32) for i in range(5)]
    # src is a stand-in id; all five share edge (0, 1) over [0, 10/3).
    violations = verify_capacity_ragged(sched, quantum=32)
    assert len(violations) == 1
    v = violations[0]
    assert v.edge == (0, 1)
    assert v.time == 0.0
    assert abs(v.total_rate - 1.5) < 1e-9
    assert len(v.flows) == 5


def test_legacy_schedule_verifies_and_matches_makespan():
    """Spec test 5: defaults (rate=1, size=1, quantum=1) reproduce the
    existing uniform semantics on a real fixture."""
    sched = load_schedule(
        FIXTURES / "schedule_8x4x4_loaded_cpsat_literal_warm.json"
    )
    assert verify_capacity(sched) == []
    assert verify_capacity_ragged(sched) == []
    assert schedule_makespan_ragged(sched) == float(schedule_makespan(sched))


def test_workload_coverage_pass_and_failures():
    w = RaggedWorkload(demand={(0, 2): 96, (1, 2): 32})
    ok = [
        _chunk(0, 0, 2, [0, 1, 2], 1.0, 64),
        _chunk(4, 0, 2, [0, 1, 2], 1.0, 32),
        _chunk(0, 1, 2, [1, 2], 1.0, 32),
    ]
    assert verify_workload_coverage(ok, w) == []

    short = ok[:2]  # (1,2) missing entirely
    problems = verify_workload_coverage(short, w)
    assert problems == ["pair (1, 2): scheduled 0 != demand 32"]

    extra = ok + [_chunk(9, 2, 0, [2, 1, 0], 1.0, 32)]
    problems = verify_workload_coverage(extra, w)
    assert problems == ["pair (2, 0): scheduled 32 but not in workload"]
