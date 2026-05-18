import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from descriptor_counter import count_dma_descriptors


def test_disjoint_single_hop_flows():
    """Two flows on disjoint edges in same round: no coalescing possible."""
    entries = [
        {"round": 0, "src": 0, "dst": 1, "path": [0, 1]},
        {"round": 0, "src": 2, "dst": 3, "path": [2, 3]},
    ]
    uncoalesced, coalesced = count_dma_descriptors(entries)
    assert uncoalesced == 2
    assert coalesced == 2


def test_cross_round_same_edge_coalesces():
    """Three flows on edge (0,1) in consecutive rounds: 1 coalesced descriptor."""
    entries = [
        {"round": 0, "src": 0, "dst": 1, "path": [0, 1]},
        {"round": 1, "src": 0, "dst": 1, "path": [0, 1]},
        {"round": 2, "src": 0, "dst": 1, "path": [0, 1]},
    ]
    uncoalesced, coalesced = count_dma_descriptors(entries)
    assert uncoalesced == 3
    assert coalesced == 1


def test_multihop_flow_expands_across_rounds():
    """Flow with 3-edge path starting at round 5 uses edges at rounds 5, 6, 7."""
    entries = [
        {"round": 5, "src": 0, "dst": 3, "path": [0, 1, 2, 3]},
    ]
    uncoalesced, coalesced = count_dma_descriptors(entries)
    assert uncoalesced == 3
    assert coalesced == 3  # three distinct edges, no coalescing


def test_multihop_flows_share_edge_across_rounds():
    """
    f1: round=0, path=[0,1,2] uses (0,1)@0 and (1,2)@1
    f2: round=2, path=[1,2,3] uses (1,2)@2 and (2,3)@3
    Edge (1,2) active at rounds {1, 2} -> contiguous, 1 run.
    """
    entries = [
        {"round": 0, "src": 0, "dst": 2, "path": [0, 1, 2]},
        {"round": 2, "src": 1, "dst": 3, "path": [1, 2, 3]},
    ]
    uncoalesced, coalesced = count_dma_descriptors(entries)
    assert uncoalesced == 4
    assert coalesced == 3
