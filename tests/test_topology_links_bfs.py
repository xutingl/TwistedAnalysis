from twisted_analysis.topology.lattice import Topology


def test_2x4_node_iteration():
    t = Topology(slice=(2, 4))
    nodes = list(t.nodes())
    assert len(nodes) == 8
    assert (0, 0) in nodes and (1, 3) in nodes


def test_2x4_link_count():
    # 8 nodes, each has 2 dims * 2 dirs = 4 directed neighbors → 32 directed edges.
    t = Topology(slice=(2, 4))
    links = list(t.directed_links())
    assert len(links) == 8 * 4


def test_2x4_link_endpoints_consistent():
    t = Topology(slice=(2, 4))
    for u, v, dim, dir in t.directed_links():
        assert t.neighbor(u, dim, dir) == v


def test_2x4_bfs_distance_symmetry():
    t = Topology(slice=(2, 4))
    dist = t.bfs_distances()
    for u in t.nodes():
        for v in t.nodes():
            assert dist[u][v] == dist[v][u]


def test_2x4_bfs_zero_to_self():
    t = Topology(slice=(2, 4))
    dist = t.bfs_distances()
    for u in t.nodes():
        assert dist[u][u] == 0


def test_4x4x8_bfs_known_pair():
    # (0,0,0) → (3,0,0): one backward wrap on dim 0 → (3, 0, 4). Then need (-, 0, -4).
    # Or four forward dim-0 steps: (1,0,0) → ... → (3,0,0). 3 hops.
    # Compare against BFS truth.
    t = Topology(slice=(4, 4, 8))
    dist = t.bfs_distances()
    assert dist[(0, 0, 0)][(3, 0, 0)] == 3
