"""Local-search scheduler: feasibility + monotone improvement tests."""
from __future__ import annotations
import pytest

from twisted_analysis.topology import Topology, ILPRouter
from twisted_analysis.schedules.local_search import local_search_repair
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


def test_local_search_preserves_feasibility_2x4():
    t, table = _table((2, 4))
    seed = literal_greedy(t, table, order="lpt")
    out = local_search_repair(t, table, seed, max_iters=50)
    assert verify_capacity(out) == []


def test_local_search_does_not_worsen_makespan_2x4():
    t, table = _table((2, 4))
    seed = literal_greedy(t, table, order="lpt")
    m0 = schedule_makespan(seed)
    out = local_search_repair(t, table, seed, max_iters=200)
    assert schedule_makespan(out) <= m0
