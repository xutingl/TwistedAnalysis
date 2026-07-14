from pathlib import Path

import pytest

from twisted_analysis.io.routing_table import load_routing_table, save_routing_table
from twisted_analysis.schedules.literal_greedy import literal_greedy
from twisted_analysis.schedules.verify import verify_capacity, schedule_makespan
from twisted_analysis.topology import Topology, ILPRouter

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def test_literal_greedy_zero_violations_on_loaded_8x4x4():
    topology = Topology(slice=(8, 4, 4))
    table = load_routing_table(FIXTURES / "routing" / "routing_table_8x4x4_twist.json")
    schedule = literal_greedy(topology, table, order="lpt")
    assert verify_capacity(schedule) == []
    # Sanity: every (src, dst) pair with src != dst appears exactly once.
    pairs = {(e["src"], e["dst"]) for e in schedule}
    n = topology.n_nodes
    assert len(pairs) == n * (n - 1)


@pytest.mark.parametrize("order", ["lpt", "spt", "natural"])
def test_literal_greedy_orderings_all_feasible(tmp_path, order):
    topology = Topology(slice=(2, 4))
    router = ILPRouter(topology=topology)
    rt_path = tmp_path / "rt.json"
    save_routing_table(topology, router, rt_path)
    table = load_routing_table(rt_path)
    schedule = literal_greedy(topology, table, order=order)
    assert verify_capacity(schedule) == []
    # Loose upper bound: c + d. For 2x4 ILP, LB=3, d=2, so makespan <= some small value.
    # Use 6 * LB as a very generous bound.
    assert schedule_makespan(schedule) <= 6 * 3
