"""Coordinate flatten/unflatten — must match the convention in
pallas_kernel/gen_orbit_greedy_kernel.py:78 (dim-0 most significant)."""
import pytest

from twisted_analysis.io.coords import flatten, unflatten


def test_flatten_4x4x8_node_42_is_1_1_2():
    # Convention: flat = i*32 + j*8 + k for slice=(4,4,8). Node 42 -> (1,1,2).
    assert flatten((1, 1, 2), (4, 4, 8)) == 42


def test_unflatten_4x4x8_42_is_1_1_2():
    assert unflatten(42, (4, 4, 8)) == (1, 1, 2)


def test_roundtrip_all_nodes_4x4x8():
    slice_ = (4, 4, 8)
    n = 4 * 4 * 8
    for i in range(4):
        for j in range(4):
            for k in range(8):
                f = flatten((i, j, k), slice_)
                assert 0 <= f < n
                assert unflatten(f, slice_) == (i, j, k)


def test_roundtrip_2d():
    slice_ = (2, 4)
    for i in range(2):
        for j in range(4):
            f = flatten((i, j), slice_)
            assert unflatten(f, slice_) == (i, j)


def test_flatten_validates_dim_count():
    with pytest.raises(ValueError):
        flatten((0, 0), (4, 4, 8))


def test_flatten_validates_in_range():
    with pytest.raises(ValueError):
        flatten((4, 0, 0), (4, 4, 8))  # i = slice[0] is out of range
