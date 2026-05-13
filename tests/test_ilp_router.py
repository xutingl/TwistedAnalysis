from twisted_analysis.topology import Topology, DORRouter
from twisted_analysis.topology.ilp_router import ILPRouter
from twisted_analysis.model import AllToAll


def test_ilp_router_path_length_equals_bfs_2x4():
    t = Topology(slice=(2, 4))
    r = ILPRouter(t)
    dist = t.bfs_distances()
    for s in t.nodes():
        for d in t.nodes():
            assert len(r.path(s, d)) == dist[s][d]


def test_ilp_router_path_length_equals_bfs_4x8():
    t = Topology(slice=(4, 8))
    r = ILPRouter(t)
    dist = t.bfs_distances()
    for s in t.nodes():
        for d in t.nodes():
            assert len(r.path(s, d)) == dist[s][d]


def test_ilp_router_path_length_equals_bfs_4x4x8():
    t = Topology(slice=(4, 4, 8))
    r = ILPRouter(t)
    dist = t.bfs_distances()
    for s in t.nodes():
        for d in t.nodes():
            assert len(r.path(s, d)) == dist[s][d]


def test_ilp_router_lb_le_dor_lb_4x8():
    """ILP routing should produce LB <= DOR's LB (load balancing helps)."""
    t = Topology(slice=(4, 8))
    dor_w = AllToAll(t, DORRouter(t), msg_size=1)
    ilp_w = AllToAll(t, ILPRouter(t), msg_size=1)
    assert ilp_w.lower_bound <= dor_w.lower_bound


def test_ilp_router_is_deterministic():
    t = Topology(slice=(4, 8))
    r1 = ILPRouter(t)
    r2 = ILPRouter(t)
    w1 = AllToAll(t, r1, 1)
    w2 = AllToAll(t, r2, 1)
    assert w1.lower_bound == w2.lower_bound


def test_ilp_router_satisfies_protocol():
    from twisted_analysis.topology import Router
    t = Topology(slice=(2, 4))
    r = ILPRouter(t)
    assert isinstance(r, Router)
