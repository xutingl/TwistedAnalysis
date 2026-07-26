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

`orbit_pack_shuffled` is the matched negative control. Without a
barrier the step count is not a physical quantity — the schedule
reaches the hardware only as the destination order the all-up-front
kernel iterates — so a win over P2P could come either from orbit
atomicity alone (every column a permutation: no incast, no idle
devices) or from the FFD certification (bins emitted contiguously, so
a sliding window of in-flight DMAs lands on orbits proven co-resident
under `c`). The control holds the former fixed and forfeits only the
latter.
"""
from __future__ import annotations
import random
from collections import Counter

from twisted_analysis.io.coords import flatten
from twisted_analysis.lp.orbit import compute_orbits
from twisted_analysis.topology import Topology


def _orbit_data(topology: Topology, table: list[list[list[int]]]):
    """Per-orbit whole-path edge loads and flow lists.

    Returns `(orbit_edges, orbit_flows)`, both keyed by orbit id:
    a Counter of directed-edge loads over every hop of every member
    flow, and the `(src_flat, dst_flat, path)` triples themselves.
    """
    orbits = compute_orbits(topology)
    slice_ = topology.slice
    orbit_edges: dict = {}
    orbit_flows: dict = {}
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
    return orbit_edges, orbit_flows


def _ffd_bins(orbit_edges, *, k: int, c: int) -> list[list]:
    """First-fit-decreasing packing of orbits into steps; orbit ids per step.

    Raises ValueError on invalid `k`/`c`, or when some orbit's own
    whole-path load already exceeds `c` (no packing can satisfy the cap).
    """
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be a positive integer; got {k!r}")
    if not isinstance(c, int) or c < 1:
        raise ValueError(f"c must be a positive integer; got {c!r}")

    self_load = {o: max(load.values()) for o, load in orbit_edges.items()}
    worst = max(self_load.values())
    if worst > c:
        offenders = sum(1 for v in self_load.values() if v > c)
        raise ValueError(
            f"c={c} is infeasible: {offenders} orbit(s) have internal "
            f"whole-path edge load up to {worst}; need c >= {worst}"
        )

    # Hottest (then heaviest) orbits first, so they claim steps while
    # co-residents with disjoint footprints still exist.
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
    return [orbit_ids for orbit_ids, _load in bins]


def _entries_from_bins(bins, orbit_flows) -> list[dict]:
    """Flatten `[[orbit_id, ...], ...]` (step-major) into schedule entries."""
    entries: list[dict] = []
    for step, orbit_ids in enumerate(bins):
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
    orbit_edges, orbit_flows = _orbit_data(topology, table)
    return _entries_from_bins(_ffd_bins(orbit_edges, k=k, c=c), orbit_flows)


def orbit_pack_shuffled(
    topology: Topology,
    table: list[list[list[int]]],
    *,
    k: int,
    c: int,
    seed: int = 0,
) -> list[dict]:
    """Negative control for `orbit_pack`: same steps, uncertified packing.

    Assigns orbits to steps uniformly at random while reproducing the
    step-size profile that `orbit_pack(k, c)` found. Everything the
    hardware win could be credited to is therefore held fixed:

      - the same 127 orbits, still atomic, so every step is a set of
        permutations and per-device send/recv counts stay balanced;
      - the same step count T;
      - the same per-step orbit counts, hence the same per-device DMA
        depth in every step.

    The single forfeited property is the FFD certification itself: the
    whole-path union edge load per step is no longer bounded by `c`, and
    co-resident orbits are no longer adjacent in the emitted destination
    order that the all-up-front (non-pfc) kernel iterates. Measuring this
    against `orbit_pack` isolates congestion control as the treatment.
    Read the achieved cap with `verify.max_step_edge_load` — it exceeds
    `c` by construction, and the kernel generator needs it for
    `--step-edge-cap`.

    Args:
      topology: source of `slice` / `n_nodes`.
      table: routing table; `table[src][dst]` is the flat-ID path.
      k: max orbits per step for the reference `orbit_pack` packing.
      c: whole-path edge cap for the reference packing. Bounds the step
         profile that gets reproduced; it does NOT bound this schedule.
      seed: RNG seed. Vary it to measure the spread over assignments —
        a single draw could be lucky.

    Returns:
      `{round, src, dst, path}` entries, rounds contiguous from 0, sorted
      by (round, src) — same shape as `orbit_pack`.

    Raises:
      ValueError: propagated from the reference `orbit_pack(k, c)`.
    """
    orbit_edges, orbit_flows = _orbit_data(topology, table)
    profile = [len(step) for step in _ffd_bins(orbit_edges, k=k, c=c)]

    order = sorted(orbit_flows)          # deterministic base order
    random.Random(seed).shuffle(order)

    bins: list[list] = []
    cut = 0
    for size in profile:
        bins.append(order[cut:cut + size])
        cut += size
    assert cut == len(order), f"profile covers {cut} of {len(order)} orbits"

    return _entries_from_bins(bins, orbit_flows)
