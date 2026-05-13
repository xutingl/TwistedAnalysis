from __future__ import annotations

from twisted_analysis.model.flow import AllToAll


def bisection_bound(workload: AllToAll) -> int:
    """A relaxed bisection-bandwidth lower bound.

    For each "cut" splitting nodes into halves A and B, count the number of
    flows crossing the cut. A valid bisection bound is ceil(cross / cut_edges),
    which must hold for every cut. We evaluate only the natural coordinate-
    aligned half-cuts (one per dim: split along the median plane of each axis).
    The true bisection bound (over all 2^N partitions) is >= this much.
    """
    t = workload.topology
    nodes = list(t.nodes())
    if len(nodes) < 2:
        return 0
    best = 0
    for d in range(t.ndim):
        threshold = t.slice[d] // 2
        A = {n_ for n_ in nodes if n_[d] < threshold}
        cross_flows = sum(
            workload.msg_size
            for f in workload.flows
            if (f.src in A) != (f.dst in A)
        )
        cut_edges = sum(
            1 for u, v, _, _ in t.directed_links() if (u in A) != (v in A)
        )
        if cut_edges > 0:
            best = max(best, -(-cross_flows // cut_edges))  # ceil division
    return best
