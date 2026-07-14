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


def test_dispatch_ragged_fluid_and_greedy():
    from pathlib import Path

    from twisted_analysis.io.routing_table import load_routing_table
    from twisted_analysis.io.schedule import schedule_from_algorithm
    from twisted_analysis.model.ragged import RaggedWorkload
    from twisted_analysis.topology import Topology

    fixtures = Path(__file__).resolve().parent.parent / "fixtures"
    topology = Topology(slice=(8, 4, 4))
    table = load_routing_table(fixtures / "routing" / "routing_table_8x4x4_twist.json")
    w = RaggedWorkload(demand={(0, 5): 64, (3, 9): 32, (100, 2): 96})

    fluid = schedule_from_algorithm(
        "ragged_fluid", topology, table, workload=w,
    )
    assert len(fluid) == 3
    assert all("rate" in e and "size" in e for e in fluid)

    greedy = schedule_from_algorithm(
        "ragged_greedy", topology, table,
        workload=w, order="lpt", preemptive=False,
    )
    assert len(greedy) == 3
    assert all(e["rate"] == 1.0 for e in greedy)
