"""RaggedWorkload model: quantum, link_load, lower_bound, validation."""
import pytest

from twisted_analysis.model.ragged import RaggedWorkload

# 3-node line 0-1-2, same shape load_routing_table returns.
LINE_TABLE = [
    [[0], [0, 1], [0, 1, 2]],
    [[1, 0], [1], [1, 2]],
    [[2, 1, 0], [2, 1], [2]],
]


def test_quantum_is_gcd_of_sizes():
    w = RaggedWorkload(demand={(0, 2): 64, (1, 2): 32, (0, 1): 96})
    assert w.quantum == 32


def test_link_load_is_size_weighted():
    w = RaggedWorkload(demand={(0, 2): 64, (1, 2): 32})
    assert w.link_load(LINE_TABLE) == {(0, 1): 64, (1, 2): 96}


def test_lower_bound_and_bottleneck():
    w = RaggedWorkload(demand={(0, 2): 64, (1, 2): 32})
    assert w.lower_bound(LINE_TABLE) == 96
    assert w.bottleneck_edges(LINE_TABLE) == [(1, 2)]


def test_rejects_self_pair():
    with pytest.raises(ValueError, match="self-pair"):
        RaggedWorkload(demand={(1, 1): 32})


def test_rejects_nonpositive_size():
    with pytest.raises(ValueError, match="positive int"):
        RaggedWorkload(demand={(0, 1): 0})


def test_rejects_bool_size():
    with pytest.raises(ValueError, match="positive int"):
        RaggedWorkload(demand={(0, 1): True})


def test_rejects_empty():
    with pytest.raises(ValueError, match="at least one flow"):
        RaggedWorkload(demand={})


def test_demand_is_copied_and_immutable():
    src = {(0, 2): 64, (1, 2): 32}
    w = RaggedWorkload(demand=src)
    assert w.quantum == 32
    src[(0, 2)] = 5  # mutating the caller's dict must not affect the workload
    assert w.demand[(0, 2)] == 64
    with pytest.raises(TypeError):
        w.demand[(0, 2)] = 5  # mappingproxy rejects writes
