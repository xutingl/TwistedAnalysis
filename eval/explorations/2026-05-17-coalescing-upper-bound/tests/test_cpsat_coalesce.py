import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest

from descriptor_counter import count_dma_descriptors


@pytest.fixture
def tiny_topology():
    """Trivial small topology for smoke testing."""
    from twisted_analysis.topology import Topology
    # 2x2 torus: 4 nodes, satisfies the {S, 2S} family constraint
    return Topology(slice=(2, 2))


def _build_tiny_routing(topology):
    """Build a simple per-flow routing for the tiny topology.

    For a 2x2 torus, just use linear flat-ID routing: src->dst always goes
    through min..max indices along the flat ordering.

    NOTE: This is a synthetic routing for smoke-test purposes only.
    The CP-SAT model treats edges as opaque identifiers, so physical
    validity of adjacency is not required for testing the solver itself.
    """
    n = topology.n_nodes
    table = [[[] for _ in range(n)] for _ in range(n)]
    for s in range(n):
        for d in range(n):
            if s == d:
                table[s][d] = [s]
            elif s < d:
                table[s][d] = list(range(s, d + 1))
            else:
                table[s][d] = list(range(s, d - 1, -1))
    return table


def test_cpsat_coalesce_returns_feasible_schedule(tiny_topology):
    """Smoke test: solver returns a feasible schedule whose coalesced count
    matches the post-hoc counter on the returned entries."""
    from cpsat_coalesce import cpsat_coalesce

    n = tiny_topology.n_nodes
    table = _build_tiny_routing(tiny_topology)
    entries, reported = cpsat_coalesce(
        tiny_topology, table, t_upper=20, time_limit_s=30, n_workers=2,
    )
    assert entries is not None, "solver returned no schedule"
    assert reported is not None
    # Validate post-hoc count matches reported objective
    _uncoalesced, coalesced = count_dma_descriptors(entries)
    assert coalesced == reported, (
        f"post-hoc coalesced count {coalesced} != reported objective {reported}"
    )
    # Validate every flow scheduled exactly once
    n_flows = n * (n - 1)
    assert len(entries) == n_flows
