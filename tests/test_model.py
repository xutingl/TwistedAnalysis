from twisted_analysis.topology import Topology, DORRouter
from twisted_analysis.model import AllToAll, Flow


def test_alltoall_flow_count_2x4():
    t = Topology(slice=(2, 4))
    r = DORRouter(t)
    w = AllToAll(t, r, msg_size=1)
    assert len(w.flows) == 8 * 7  # N*(N-1)


def test_alltoall_flow_size():
    t = Topology(slice=(2, 4))
    r = DORRouter(t)
    w = AllToAll(t, r, msg_size=4)
    assert all(f.size == 4 for f in w.flows)


def test_link_load_sums_to_total_hops_times_m():
    t = Topology(slice=(2, 4))
    r = DORRouter(t)
    w = AllToAll(t, r, msg_size=1)
    total_load = sum(w.link_load.values())
    expected = sum(len(w.path(f)) for f in w.flows)
    assert total_load == expected


def test_lower_bound_is_max_link_load():
    t = Topology(slice=(2, 4))
    r = DORRouter(t)
    w = AllToAll(t, r, msg_size=1)
    assert w.lower_bound == max(w.link_load.values())


def test_lower_bound_scales_with_msg_size():
    t = Topology(slice=(2, 4))
    r = DORRouter(t)
    w1 = AllToAll(t, r, msg_size=1)
    w4 = AllToAll(t, r, msg_size=4)
    assert w4.lower_bound == 4 * w1.lower_bound


def test_bottleneck_edges_attain_lb():
    t = Topology(slice=(4, 8))
    r = DORRouter(t)
    w = AllToAll(t, r, msg_size=1)
    for e in w.bottleneck_edges():
        assert w.link_load[e] == w.lower_bound
