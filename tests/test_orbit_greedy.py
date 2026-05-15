"""OrbitGreedy / PipelinedOrbit: full-coverage, sim-feasible, hop_schedule populated."""
from __future__ import annotations
import pytest

from twisted_analysis.model import AllToAll
from twisted_analysis.schedules import OrbitGreedySchedule, PipelinedOrbitSchedule
from twisted_analysis.simulator import Simulator
from twisted_analysis.topology import DORRouter, ILPRouter, Topology


@pytest.mark.parametrize("slice_", [(2, 4), (4, 8)])
@pytest.mark.parametrize("router_kind", ["dor", "ilp"])
@pytest.mark.parametrize("schedule_cls", [OrbitGreedySchedule, PipelinedOrbitSchedule])
def test_orbit_greedy_covers_full_alltoall_and_simulates(slice_, router_kind, schedule_cls):
    t = Topology(slice=slice_)
    r = DORRouter(t) if router_kind == "dor" else ILPRouter(t)
    w = AllToAll(t, r, msg_size=1)
    sched = schedule_cls()
    injs = sched.emit(w)
    # full coverage: N*(N-1) flows
    assert len({inj.flow for inj in injs}) == t.n_nodes * (t.n_nodes - 1)
    # every injection has a populated hop_schedule
    for inj in injs:
        path = r.path(inj.flow.src, inj.flow.dst)
        assert len(inj.hop_schedule) == len(path)
        # monotone in time
        assert all(inj.hop_schedule[i] < inj.hop_schedule[i + 1]
                   for i in range(len(inj.hop_schedule) - 1))
        assert inj.start_step == inj.hop_schedule[0]
    # simulator must accept and run to a finite makespan >= LB
    sim = Simulator(t, r, list(w.flows))
    for inj in injs:
        sim.inject(inj)
    makespan = sim.run()
    assert makespan >= w.lower_bound


def test_pipelined_orbit_hop_gaps_are_one():
    t = Topology(slice=(4, 8))
    r = ILPRouter(t)
    w = AllToAll(t, r, msg_size=1)
    injs = PipelinedOrbitSchedule().emit(w)
    for inj in injs:
        for i in range(len(inj.hop_schedule) - 1):
            assert inj.hop_schedule[i + 1] - inj.hop_schedule[i] == 1


def test_orbit_greedy_lb_match_small():
    # 2x4 has LB=4 (DOR) / LB=3 (ILP). Greedy should achieve these tight bounds.
    t = Topology(slice=(2, 4))
    r = ILPRouter(t)
    w = AllToAll(t, r, msg_size=1)
    injs = OrbitGreedySchedule().emit(w)
    sim = Simulator(t, r, list(w.flows))
    for inj in injs:
        sim.inject(inj)
    makespan = sim.run()
    assert makespan == w.lower_bound  # ratio = 1.00


# (2,4,4)-ilp and (4,8)-ilp are LB+1 under orbit_greedy_full: the (dim, dir)-keyed
# predecessor faked LB-tightness via physical-edge capacity violations
# (see plan 2026-05-15-multi-algorithm-scheduling.md Task 3). The other 6 cells
# still hit LB exactly.
@pytest.mark.parametrize("slice_,router_kind,expected_lb,expected_makespan", [
    ((2, 4), "dor", 4, 4),
    ((2, 4), "ilp", 3, 3),
    ((2, 2, 4), "dor", 7, 7),
    ((2, 2, 4), "ilp", 5, 5),
    ((2, 4, 4), "dor", 16, 16),
    ((2, 4, 4), "ilp", 11, 12),   # LB+1: see comment above
    ((4, 8), "dor", 26, 26),
    ((4, 8), "ilp", 21, 22),      # LB+1: see comment above
])
def test_orbit_greedy_default_achieves_lb(slice_, router_kind, expected_lb, expected_makespan):
    """Default `lpt_tail_asc` ordering hits makespan == LB on most cells.

    Locks the regression: plain LPT missed 2x4x4 DOR by 1 step (gave 17, LB=16)
    until we added the tail-load-ascending tiebreak.

    Two ILP-routed cells now report LB+1 because orbit_greedy_full enforces
    physical-edge capacity correctly (the prior (dim, dir) keying produced
    capacity-violating schedules that nominally matched LB).
    """
    t = Topology(slice=slice_)
    r = DORRouter(t) if router_kind == "dor" else ILPRouter(t)
    w = AllToAll(t, r, msg_size=1)
    assert w.lower_bound == expected_lb
    sim = Simulator(t, r, list(w.flows))
    for inj in OrbitGreedySchedule().emit(w):
        sim.inject(inj)
    assert sim.run() == expected_makespan


def test_orbit_greedy_lpt_misses_lb_on_2x4x4_dor():
    """Locks the empirical observation: pure LPT (no tail tiebreak) MISSES LB
    on 2x4x4 DOR. Regression-guards against silently reverting the fix.
    """
    t = Topology(slice=(2, 4, 4))
    r = DORRouter(t)
    w = AllToAll(t, r, msg_size=1)
    assert w.lower_bound == 16
    sim = Simulator(t, r, list(w.flows))
    for inj in OrbitGreedySchedule(order="lpt").emit(w):
        sim.inject(inj)
    assert sim.run() == 17  # LB+1


def test_orbit_greedy_invalid_order_raises():
    import pytest as _pt
    with _pt.raises(ValueError):
        OrbitGreedySchedule(order="bogus")
