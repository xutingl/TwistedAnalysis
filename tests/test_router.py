from twisted_analysis.topology.lattice import Topology
from twisted_analysis.topology.router import Router


def test_2x4_self_path_is_empty():
    t = Topology(slice=(2, 4))
    r = Router(t)
    assert r.path((0, 0), (0, 0)) == ()


def test_2x4_one_hop_inplane():
    t = Topology(slice=(2, 4))
    r = Router(t)
    p = r.path((0, 0), (0, 1))
    assert len(p) == 1
    u, v, dim, dir = p[0]
    assert u == (0, 0) and v == (0, 1)


def test_2x4_twist_shortcut_one_hop():
    # (0, 0) -> (1, 2) is a single backward wrap on dim 0 (since slice[0]=2 shifts dim 1 by 2).
    t = Topology(slice=(2, 4))
    r = Router(t)
    p = r.path((0, 0), (1, 2))
    assert len(p) == 1


def test_dor_path_length_equals_bfs_distance_2x4():
    t = Topology(slice=(2, 4))
    r = Router(t)
    dist = t.bfs_distances()
    for s in t.nodes():
        for d in t.nodes():
            assert len(r.path(s, d)) == dist[s][d], (
                f"DOR path {s}->{d} length {len(r.path(s, d))} != BFS {dist[s][d]}"
            )


def test_dor_path_length_equals_bfs_distance_4x8():
    t = Topology(slice=(4, 8))
    r = Router(t)
    dist = t.bfs_distances()
    for s in t.nodes():
        for d in t.nodes():
            assert len(r.path(s, d)) == dist[s][d]


def test_dor_path_length_equals_bfs_distance_4x4x8():
    t = Topology(slice=(4, 4, 8))
    r = Router(t)
    dist = t.bfs_distances()
    for s in t.nodes():
        for d in t.nodes():
            assert len(r.path(s, d)) == dist[s][d]


def test_router_is_deterministic():
    t = Topology(slice=(4, 8))
    r1 = Router(t)
    r2 = Router(t)
    for s in t.nodes():
        for d in t.nodes():
            assert r1.path(s, d) == r2.path(s, d)
