import pytest

from twisted_analysis.io.routing_table import load_routing_table, save_routing_table
from twisted_analysis.io.schedule import schedule_from_algorithm
from twisted_analysis.schedules.verify import verify_capacity
from twisted_analysis.topology import Topology, ILPRouter


@pytest.mark.parametrize("algo", ["orbit_greedy", "orbit_greedy_full", "literal_greedy"])
def test_dispatcher_runs_each_algorithm_on_2x4(tmp_path, algo):
    topology = Topology(slice=(2, 4))
    router = ILPRouter(topology=topology)
    rt_path = tmp_path / "rt.json"
    save_routing_table(topology, router, rt_path)
    table = load_routing_table(rt_path)
    schedule = schedule_from_algorithm(algo, topology, table)
    assert verify_capacity(schedule) == []
    # Exactly N*(N-1) entries.
    n = topology.n_nodes
    assert len(schedule) == n * (n - 1)


def test_dispatcher_rejects_unknown_algorithm(tmp_path):
    topology = Topology(slice=(2, 4))
    router = ILPRouter(topology=topology)
    rt_path = tmp_path / "rt.json"
    save_routing_table(topology, router, rt_path)
    table = load_routing_table(rt_path)
    with pytest.raises(ValueError, match="unknown algorithm"):
        schedule_from_algorithm("does_not_exist", topology, table)
