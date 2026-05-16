"""LP-rounding scheduler: small-cell sanity tests."""
from __future__ import annotations
import pytest

pytest.importorskip("pulp")

from twisted_analysis.topology import Topology, ILPRouter
from twisted_analysis.schedules.lp_rounding import lp_rounding
from twisted_analysis.schedules.literal_greedy import literal_greedy
from twisted_analysis.schedules.verify import verify_capacity, schedule_makespan


def _table(slice_):
    """Build a flat-id routing table from ILPRouter (via save/load round-trip)."""
    import tempfile, os
    from twisted_analysis.io.routing_table import save_routing_table, load_routing_table
    t = Topology(slice=slice_)
    r = ILPRouter(t)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_path = f.name
    try:
        save_routing_table(t, r, tmp_path)
        table = load_routing_table(tmp_path)
    finally:
        os.unlink(tmp_path)
    return t, table


def test_lp_rounding_produces_feasible_2x4():
    t, table = _table((2, 4))
    sch = lp_rounding(t, table, t_upper=10, n_trials=20, seed=0)
    assert verify_capacity(sch) == []


def test_lp_rounding_beats_or_matches_literal_greedy_2x4():
    t, table = _table((2, 4))
    lg = schedule_makespan(literal_greedy(t, table, order="lpt"))
    sch = lp_rounding(t, table, t_upper=max(10, lg + 2), n_trials=50, seed=0)
    assert schedule_makespan(sch) <= lg + 1  # within 1 step of greedy in small cell
