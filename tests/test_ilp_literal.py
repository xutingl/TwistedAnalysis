import pytest

from twisted_analysis.io.routing_table import load_routing_table, save_routing_table
from twisted_analysis.schedules.ilp_literal import ilp_literal
from twisted_analysis.schedules.verify import verify_capacity, schedule_makespan
from twisted_analysis.topology import Topology, ILPRouter


@pytest.mark.timeout(60)
def test_ilp_literal_matches_lb_on_2x4(tmp_path):
    topology = Topology(slice=(2, 4))
    router = ILPRouter(topology=topology)
    rt_path = tmp_path / "rt.json"
    save_routing_table(topology, router, rt_path)
    table = load_routing_table(rt_path)
    schedule = ilp_literal(topology, table, time_limit_s=30)
    assert verify_capacity(schedule) == []
    # For 2x4 ILP: LB = 3 per docs/results.md, diameter d = 2.
    # Literal-ILP makespan should match LB (the symmetric ILP cross-validated
    # T = LB on this cell).
    assert schedule_makespan(schedule) == 3


def test_ilp_literal_raises_without_pulp(monkeypatch):
    # Defensive: surface a clean error if pulp isn't installed.
    import sys
    monkeypatch.setitem(sys.modules, "pulp", None)
    topology = Topology(slice=(2, 4))
    # Build minimal table from DOR to avoid pulp dependency for setup.
    from twisted_analysis.topology import DORRouter
    from twisted_analysis.io.routing_table import save_routing_table, load_routing_table
    from pathlib import Path
    rt = Path("/tmp/_test_ilp_lit_no_pulp.json")
    save_routing_table(topology, DORRouter(topology=topology), rt)
    table = load_routing_table(rt)
    with pytest.raises((ImportError, RuntimeError, TypeError)):
        ilp_literal(topology, table, time_limit_s=5)
