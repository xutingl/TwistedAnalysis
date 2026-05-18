"""Count per-edge per-round DMA descriptors before/after coalescing.

Schedule entries follow the on-disk schema:
  {"round": int, "src": int, "dst": int, "path": [int, ...]}
where `path` is the sequence of flat device IDs from src to dst (inclusive).

A flow with `round = s` and `path = [a, b, c, d]` uses:
  - edge (a, b) at absolute round s
  - edge (b, c) at absolute round s + 1
  - edge (c, d) at absolute round s + 2
"""
from collections import defaultdict


def count_dma_descriptors(entries):
    """Return (uncoalesced, coalesced) per-edge per-round descriptor counts.

    `uncoalesced` = total per-hop DMAs = sum over flows of (len(path) - 1).
    `coalesced` = sum over physical edges of the number of maximal contiguous
                  runs of active rounds for that edge.
    """
    edge_active = defaultdict(set)  # (u, v) -> set of absolute round ints
    uncoalesced = 0
    for e in entries:
        start = e["round"]
        path = e["path"]
        for h in range(len(path) - 1):
            edge = (path[h], path[h + 1])
            edge_active[edge].add(start + h)
            uncoalesced += 1

    coalesced = 0
    for active_rounds in edge_active.values():
        sorted_rounds = sorted(active_rounds)
        runs = 1
        for i in range(1, len(sorted_rounds)):
            if sorted_rounds[i] != sorted_rounds[i - 1] + 1:
                runs += 1
        coalesced += runs
    return uncoalesced, coalesced
