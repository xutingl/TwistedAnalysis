"""Spread-greedy AllToAll scheduler tests."""
from __future__ import annotations
from collections import Counter
from pathlib import Path

import pytest

from twisted_analysis.io.routing_table import load_routing_table, save_routing_table
from twisted_analysis.schedules.spread_greedy import spread_greedy
from twisted_analysis.schedules.literal_greedy import literal_greedy
from twisted_analysis.schedules.verify import verify_capacity, schedule_makespan
from twisted_analysis.topology import Topology, ILPRouter

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _per_device_per_round_max(schedule):
    out_counts: Counter = Counter()
    in_counts: Counter = Counter()
    for e in schedule:
        out_counts[(e["src"], e["round"])] += 1
        in_counts[(e["dst"], e["round"])] += 1
    return max(out_counts.values()), max(in_counts.values())


def test_spread_greedy_zero_violations_loaded_8x4x4_k2():
    topology = Topology(slice=(8, 4, 4))
    table = load_routing_table(FIXTURES / "routing" / "routing_table_8x4x4_twist.json")
    schedule = spread_greedy(topology, table, k=2, order="lpt")
    assert verify_capacity(schedule) == []
    pairs = {(e["src"], e["dst"]) for e in schedule}
    n = topology.n_nodes
    assert len(pairs) == n * (n - 1)


def test_spread_greedy_respects_k_cap_loaded_8x4x4():
    topology = Topology(slice=(8, 4, 4))
    table = load_routing_table(FIXTURES / "routing" / "routing_table_8x4x4_twist.json")
    for k in (1, 2, 3, 4):
        schedule = spread_greedy(topology, table, k=k, order="lpt")
        out_max, in_max = _per_device_per_round_max(schedule)
        assert out_max <= k, f"k={k}: max outgoing per device per round = {out_max} > k"
        assert in_max <= k, f"k={k}: max incoming per device per round = {in_max} > k"


def test_spread_greedy_k1_makespan_at_least_n_minus_one(tmp_path):
    topology = Topology(slice=(2, 4))
    router = ILPRouter(topology=topology)
    rt_path = tmp_path / "rt.json"
    save_routing_table(topology, router, rt_path)
    table = load_routing_table(rt_path)
    schedule = spread_greedy(topology, table, k=1, order="lpt")
    assert verify_capacity(schedule) == []
    assert schedule_makespan(schedule) >= topology.n_nodes - 1


def test_spread_greedy_large_k_matches_literal_greedy_makespan():
    """Cap is non-binding at K=N, so per-flow choices are identical to literal_greedy (same order, same tie-break)."""
    topology = Topology(slice=(8, 4, 4))
    table = load_routing_table(FIXTURES / "routing" / "routing_table_8x4x4_twist.json")
    spread = spread_greedy(topology, table, k=topology.n_nodes, order="lpt")
    lit = literal_greedy(topology, table, order="lpt")
    assert schedule_makespan(spread) == schedule_makespan(lit)


@pytest.mark.parametrize("order", ["lpt", "spt", "natural"])
def test_spread_greedy_orderings_all_feasible(order):
    topology = Topology(slice=(8, 4, 4))
    table = load_routing_table(FIXTURES / "routing" / "routing_table_8x4x4_twist.json")
    schedule = spread_greedy(topology, table, k=2, order=order)
    assert verify_capacity(schedule) == []


def test_spread_greedy_invalid_k_raises(tmp_path):
    topology = Topology(slice=(2, 4))
    router = ILPRouter(topology=topology)
    rt_path = tmp_path / "rt.json"
    save_routing_table(topology, router, rt_path)
    table = load_routing_table(rt_path)
    with pytest.raises(ValueError, match="k must be a positive integer"):
        spread_greedy(topology, table, k=0, order="lpt")
    with pytest.raises(ValueError, match="k must be a positive integer"):
        spread_greedy(topology, table, k=-1, order="lpt")


def test_spread_greedy_via_dispatch():
    """schedule_from_algorithm('spread_greedy', ...) must dispatch correctly."""
    from twisted_analysis.io.schedule import schedule_from_algorithm
    topology = Topology(slice=(8, 4, 4))
    table = load_routing_table(FIXTURES / "routing" / "routing_table_8x4x4_twist.json")
    schedule = schedule_from_algorithm("spread_greedy", topology, table, k=2)
    assert verify_capacity(schedule) == []
    n = topology.n_nodes
    assert len({(e["src"], e["dst"]) for e in schedule}) == n * (n - 1)
