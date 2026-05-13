import pytest
from twisted_analysis.topology.lattice import Topology

def test_2x4_inplane_step():
    t = Topology(slice=(2, 4))
    assert t.neighbor((0, 0), dim=1, dir=1) == (0, 1)
    assert t.neighbor((0, 1), dim=1, dir=-1) == (0, 0)

def test_2x4_smalldim_wrap_is_half_shift():
    t = Topology(slice=(2, 4))
    # (1, 0) +dim0 wraps: shift every coord by slice[0]=2
    assert t.neighbor((1, 0), dim=0, dir=1) == (0, 2)
    # Backward wrap from (0, 0)
    assert t.neighbor((0, 0), dim=0, dir=-1) == (1, 2)

def test_2x4_bigdim_wrap_has_no_effective_twist():
    t = Topology(slice=(2, 4))
    assert t.neighbor((0, 3), dim=1, dir=1) == (0, 0)
    assert t.neighbor((0, 0), dim=1, dir=-1) == (0, 3)

def test_4x8_smalldim_wrap_shifts_by_4():
    t = Topology(slice=(4, 8))
    assert t.neighbor((3, 0), dim=0, dir=1) == (0, 4)
    assert t.neighbor((0, 0), dim=0, dir=-1) == (3, 4)

def test_4x4x8_both_small_dims_twist_into_big():
    t = Topology(slice=(4, 4, 8))
    assert t.neighbor((3, 0, 0), dim=0, dir=1) == (0, 0, 4)
    assert t.neighbor((0, 3, 0), dim=1, dir=1) == (0, 0, 4)
    assert t.neighbor((0, 0, 7), dim=2, dir=1) == (0, 0, 0)

def test_assert_S_or_2S_only():
    with pytest.raises(AssertionError):
        Topology(slice=(2, 6))  # 6 is neither S=2 nor 2S=4
