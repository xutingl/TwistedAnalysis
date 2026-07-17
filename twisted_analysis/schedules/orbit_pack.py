"""Orbit packing for barrier-delimited (step-model) execution.

Packs the N-1 translation orbits into as few barrier steps as possible,
subject to:

  - at most `k` orbits per step. Orbits are permutations (each device
    appears exactly once as src and once as dst per orbit), so every
    device sends AND receives exactly `k_t <= k` DMAs in step t —
    per-device balance is structural, not incidental.
  - whole-path union edge load at most `c` per step: counting every hop
    of every member flow's path, no directed physical edge is traversed
    by more than `c` flows within one step.

Why this objective: the `--per-step-barrier` (TPU v4 / "pfc") kernel
executes one step's DMAs, drains the send semaphore, then proceeds. The
barrier serializes steps, so the staggered-hop capacity model that
`orbit_greedy_full` optimizes enforces cross-round constraints the
hardware makes irrelevant — while never checking within-step whole-path
contention, which is what actually congests the ICI during a step.
Wall-clock under the barrier model is approximately

    T * (barrier + launch)  +  sum_t max(k_t * DMA_issue, L_t * wire)

where L_t is the step's max whole-path edge load. Every schedule pays
the same total per-device DMA issue work (N-1 DMAs per device), so the
free variables are the step count T and the per-step congestion cap.
On the loaded 8x4x4 routing, (k=6, c=3) yields T=27 steps with per-step
congestion no worse than the P2P rotation baseline's worst round —
versus 80 steps for orbit_greedy_full and 127 for P2P.

The schedule is intentionally NOT feasible under `verify_capacity`'s
staggered-hop model; check it with `verify_capacity_step` instead.
Translation symmetry (identical per-source round columns) holds by
orbit atomicity, so the `--per-step-barrier` codegen accepts it.
"""
from __future__ import annotations
from collections import Counter

from twisted_analysis.io.coords import flatten
from twisted_analysis.lp.orbit import compute_orbits
from twisted_analysis.topology import Topology


def orbit_pack(
    topology: Topology,
    table: list[list[list[int]]],
    *,
    k: int,
    c: int,
) -> list[dict]:
    """Pack orbits into steps via first-fit-decreasing; return entries.

    Args:
      topology: source of `slice` / `n_nodes`.
      table: routing table; `table[src][dst]` is the flat-ID path.
      k: max orbits (= per-device DMAs) per step. Keep k * num_packets
         within the target's DMA descriptor queue (k <= 6 is proven safe
         on TPU v4 — the orbit_greedy_full pfc kernel's widest step).
      c: max whole-path edge load per step. Must be >= the hottest
         single orbit's internal load (3 on the loaded 8x4x4 routing).

    Returns:
      `{round, src, dst, path}` entries, rounds contiguous from 0; round
      values are barrier-step indices. Sorted by (round, src).

    Raises:
      ValueError: on invalid `k`/`c`, or when `c` is below some orbit's
        internal whole-path load (no packing can satisfy the cap).
    """
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be a positive integer; got {k!r}")
    if not isinstance(c, int) or c < 1:
        raise ValueError(f"c must be a positive integer; got {c!r}")

    orbits = compute_orbits(topology)
    slice_ = topology.slice

    orbit_edges: dict = {}   # orbit_id -> Counter of whole-path edge loads
    orbit_flows: dict = {}   # orbit_id -> [(src_flat, dst_flat, path), ...]
    for orbit_id, members in orbits.items():
        load: Counter = Counter()
        flows = []
        for src, dst in members:
            src_flat = flatten(src, slice_)
            dst_flat = flatten(dst, slice_)
            path = list(table[src_flat][dst_flat])
            flows.append((src_flat, dst_flat, path))
            for i in range(len(path) - 1):
                load[(path[i], path[i + 1])] += 1
        orbit_edges[orbit_id] = load
        orbit_flows[orbit_id] = flows

    self_load = {o: max(load.values()) for o, load in orbit_edges.items()}
    worst = max(self_load.values())
    if worst > c:
        offenders = sum(1 for v in self_load.values() if v > c)
        raise ValueError(
            f"c={c} is infeasible: {offenders} orbit(s) have internal "
            f"whole-path edge load up to {worst}; need c >= {worst}"
        )

    # First-fit-decreasing: hottest (then heaviest) orbits first, so they
    # claim steps while co-residents with disjoint footprints still exist.
    order = sorted(
        orbit_edges,
        key=lambda o: (-self_load[o], -sum(orbit_edges[o].values()), o),
    )
    bins: list[tuple[list, Counter]] = []   # (orbit_ids, union edge load)
    for o in order:
        oc = orbit_edges[o]
        for members_in_bin, load in bins:
            if len(members_in_bin) >= k:
                continue
            if all(load[e] + v <= c for e, v in oc.items()):
                members_in_bin.append(o)
                load.update(oc)
                break
        else:
            bins.append(([o], Counter(oc)))

    entries: list[dict] = []
    for step, (orbit_ids, _load) in enumerate(bins):
        for o in orbit_ids:
            for src_flat, dst_flat, path in orbit_flows[o]:
                entries.append({
                    "round": step,
                    "src": src_flat,
                    "dst": dst_flat,
                    "path": path,
                })
    entries.sort(key=lambda e: (e["round"], e["src"]))
    return entries
