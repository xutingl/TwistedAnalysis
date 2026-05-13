"""Adapter: convert symmetric-LP orbit assignment to per-unit Injections."""
from __future__ import annotations
from collections import defaultdict

from twisted_analysis.lp.orbit import compute_orbits
from twisted_analysis.model.flow import Flow
from twisted_analysis.schedules.base import Injection
from twisted_analysis.topology import Topology, Router


def symmetric_assignment_to_injections(
    topology: Topology,
    router: Router,
    flows: list[Flow],
    assignment: dict,  # (orbit_id, hop_i, t) -> 0/1
) -> list[Injection]:
    """Expand the orbit-level schedule into per-unit Injections.

    Each orbit's hop_i fire-time at canonical orbit time t becomes the fire-time
    for every src in the orbit (translation-equivariant).
    """
    orbits = compute_orbits(topology)
    pair_to_orbit: dict[tuple, "OrbitId"] = {}
    for orbit_id, members in orbits.items():
        for m in members:
            pair_to_orbit[m] = orbit_id
    by_orbit_hop: dict[tuple, dict[int, int]] = defaultdict(dict)
    for (orbit_id, hop_i, t), val in assignment.items():
        if val is not None and val > 0.5:
            by_orbit_hop[orbit_id][hop_i] = t
    injections: list[Injection] = []
    for f in flows:
        orbit_id = pair_to_orbit.get((f.src, f.dst))
        if orbit_id is None:
            continue
        hop_times = by_orbit_hop[orbit_id]
        hop_schedule = tuple(hop_times[i] for i in sorted(hop_times.keys()))
        start = hop_schedule[0] if hop_schedule else 0
        injections.append(Injection(
            flow=f, start_step=start, priority=0, hop_schedule=hop_schedule,
        ))
    return injections
