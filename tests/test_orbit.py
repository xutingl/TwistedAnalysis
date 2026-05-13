from twisted_analysis.topology import Topology
from twisted_analysis.lp.orbit import compute_orbits, OrbitId


def test_2x4_orbit_count():
    """8 nodes, AllToAll -> 56 flows -> 7 orbits each of size 8."""
    t = Topology(slice=(2, 4))
    orbits = compute_orbits(t)
    assert len(orbits) == 7  # N-1 orbits
    for members in orbits.values():
        assert len(members) == 8  # N members per orbit


def test_orbit_translation_consistency():
    """All members of an orbit have the same path length."""
    t = Topology(slice=(4, 8))
    orbits = compute_orbits(t)
    from twisted_analysis.topology import DORRouter
    r = DORRouter(t)
    for orbit_id, members in orbits.items():
        path_lens = {len(r.path(s, d)) for s, d in members}
        assert len(path_lens) == 1, f"orbit {orbit_id}: mixed path lengths {path_lens}"


def test_orbit_total_membership():
    """Every (src, dst) with src != dst is in exactly one orbit."""
    t = Topology(slice=(2, 4))
    orbits = compute_orbits(t)
    seen = set()
    for members in orbits.values():
        for m in members:
            assert m not in seen
            seen.add(m)
    nodes = list(t.nodes())
    expected = {(s, d) for s in nodes for d in nodes if s != d}
    assert seen == expected
