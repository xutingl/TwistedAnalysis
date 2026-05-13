from twisted_analysis.topology import Topology, Router, DORRouter


def test_dor_router_implements_protocol():
    t = Topology(slice=(2, 4))
    r = DORRouter(t)
    # Must satisfy the Router Protocol structurally
    assert isinstance(r, Router) or hasattr(r, "path")
    assert r.path((0, 0), (0, 0)) == ()
    assert len(r.path((0, 0), (0, 1))) == 1


def test_router_is_protocol():
    # The Router export is a typing.Protocol — not a concrete class.
    import typing
    assert hasattr(Router, "_is_protocol") and Router._is_protocol is True
