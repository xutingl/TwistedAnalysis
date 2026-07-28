"""Uniform-block orbit packing + bottleneck block sequencing.

Targets the objective that actually predicts wall-clock on the all-up-front
(non-pfc) kernel: the max whole-path directed-edge load over any sliding
window of W dest-table columns.

Why that objective. Without a per-step barrier nothing serializes steps, so
the schedule's `round` field never reaches the wire — the kernel issues a
flat `.start()` stream and the hardware's DMA queues decide what is
concurrent. With all 128 devices marching through their dest tables in
lockstep, the concurrent set is a sliding window of columns whose width W
is set by outstanding descriptors (payload / 32 KB packets) and broadened
by the engine's LRU arbitration. Makespan, barrier-step count and
per-round edge load are all invisible to this; window load is not.

Lower bound. Over the 127 cyclic W-windows each column is counted W times,
so every edge satisfies max-window-load >= W * L_e / 127, giving
`LB(W) = ceil(W * LB_total / (N-1))`.

Why blocks rather than a flat greedy. A least-burstiness-next greedy
optimises this objective directly and still loses to `orbit_greedy_full`
away from its tuning window — myopic early picks strand incompatible
orbits in the tail. The structural fix is to notice that `orbit_pack`'s
ragged bins (sizes 4-6) let a 6-column window straddle THREE bins, so
three bins at load 3 give 9. Forcing every block to size >= W caps the
straddle at two blocks. Then ordering the blocks so adjacent unions stay
light bounds what the straddle can cost.

On the loaded 8x4x4 routing this reaches 7/11/17/31/60 at
W = 6/12/24/48/96 versus `orbit_greedy_full`'s 8/13/20/36/66 — 13-15%
lower at every window, with no tuning-window overfit.

Rounds are emitted one orbit per round, so dest-table column k is exactly
the k-th orbit of the sequence for every source. Sharing a round across a
block would let the codegen's `(round, dst)` sort order the block
differently per source and scramble the sequence being optimised.
"""
from __future__ import annotations
from collections import Counter

from twisted_analysis.schedules.orbit_pack import _entries_from_bins, _orbit_data
from twisted_analysis.topology import Topology


def _block_sizes(n_items: int, w: int) -> list[int]:
    """Block sizes, all >= w, as equal as possible.

    Every block >= w is what caps a W-window straddle at two blocks.
    n_items=127, w=6 -> twenty 6s and one 7 (127 is prime, so no exact
    division exists; the remainder is spread rather than left as a runt
    block, which would reintroduce a three-block straddle).
    """
    n_blocks = n_items // w
    if n_blocks == 0:
        raise ValueError(f"w={w} exceeds the orbit count {n_items}")
    sizes = [w] * n_blocks
    for i in range(n_items - n_blocks * w):
        sizes[i % n_blocks] += 1
    return sizes


def _pack_uniform(orbit_edges: dict, sizes: list[int]):
    """Best-fit-decreasing into fixed-capacity blocks; minimise max load.

    Unlike `orbit_pack`'s first-fit, this is best-fit over fixed
    capacities: block sizes are a hard constraint (they are what bounds
    the straddle) and the union edge load is what gets minimised.
    """
    self_load = {o: max(c.values()) for o, c in orbit_edges.items()}
    order = sorted(
        orbit_edges,
        key=lambda o: (-self_load[o], -sum(orbit_edges[o].values()), o),
    )
    blocks: list[list] = [[] for _ in sizes]
    loads: list[Counter] = [Counter() for _ in sizes]
    for o in order:
        oc = orbit_edges[o]
        best, best_key = None, None
        for b, cap in enumerate(sizes):
            if len(blocks[b]) >= cap:
                continue
            cand = max((loads[b][e] + v for e, v in oc.items()), default=0)
            cand = max(cand, max(loads[b].values(), default=0))
            key = (cand, sum(loads[b][e] + v for e, v in oc.items()), b)
            if best_key is None or key < best_key:
                best_key, best = key, b
        blocks[best].append(o)
        loads[best].update(oc)
    return blocks, loads


def _sequence_blocks(loads: list[Counter]) -> list[int]:
    """Bottleneck-greedy Hamiltonian path over blocks.

    Cost between two blocks is the max edge load of their union — an
    upper bound on what any window straddling them can carry. Starting
    from the heaviest block places it where it has only one neighbour.
    """
    n = len(loads)

    def pair_cost(i: int, j: int) -> int:
        m = Counter(loads[i])
        m.update(loads[j])
        return max(m.values())

    start = max(range(n), key=lambda b: max(loads[b].values()))
    seq, unused = [start], set(range(n)) - {start}
    while unused:
        cur = seq[-1]
        nxt = min(unused, key=lambda b: (pair_cost(cur, b), b))
        seq.append(nxt)
        unused.discard(nxt)
    return seq


def orbit_block_seq(
    topology: Topology,
    table: list[list[list[int]]],
    *,
    w: int,
) -> list[dict]:
    """Sequence orbits via uniform-block packing + block ordering.

    Args:
      topology: source of `slice` / `n_nodes`.
      table: routing table; `table[src][dst]` is the flat-ID path.
      w: target block size = the window width to optimise for. Every
        block is sized >= w so a w-column window straddles at most two
        blocks. On the loaded 8x4x4 routing w=12 dominates w=6 at every
        measured window; w is a tuning knob, not a hardware cap, and the
        result is robust across windows either way.

    Returns:
      `{round, src, dst, path}` entries, one orbit per round, rounds
      contiguous from 0 in sequence order. Sorted by (round, src).

    Raises:
      ValueError: if `w` is not a positive int, or exceeds the orbit count.
    """
    if not isinstance(w, int) or w < 1:
        raise ValueError(f"w must be a positive integer; got {w!r}")

    orbit_edges, orbit_flows = _orbit_data(topology, table)
    sizes = _block_sizes(len(orbit_edges), w)
    blocks, loads = _pack_uniform(orbit_edges, sizes)
    seq = _sequence_blocks(loads)
    order = [o for b in seq for o in blocks[b]]
    # One orbit per round: column k == orbit k for every source.
    return _entries_from_bins([[o] for o in order], orbit_flows)
