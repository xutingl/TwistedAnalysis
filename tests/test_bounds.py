from twisted_analysis.topology import Topology, Router
from twisted_analysis.model import AllToAll
from twisted_analysis.model.bounds import bisection_bound


def test_bisection_bound_2x4_positive():
    t = Topology(slice=(2, 4))
    r = Router(t)
    w = AllToAll(t, r, msg_size=1)
    bb = bisection_bound(w)
    assert bb >= 1
