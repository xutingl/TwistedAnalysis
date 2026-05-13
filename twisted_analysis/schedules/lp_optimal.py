from __future__ import annotations
from collections import defaultdict

from twisted_analysis.model.flow import Flow
from twisted_analysis.schedules.base import Injection
from twisted_analysis.topology import Router


def lp_assignment_to_injections(
    flows: list[Flow],
    router: Router,
    assignment: dict[tuple[int, int, int], float],
) -> list[Injection]:
    """Translate the LP `x[unit, i, t]` assignment into Injection records.

    The LP encodes per-unit per-hop firing times.  Each unit gets:
    - ``start_step``  : the LP's fire step at hop 0.
    - ``hop_schedule``: LP fire step for every hop in order (hop 0, 1, …).

    Storing the full per-hop schedule lets the simulator use the LP-intended
    ordering at every intermediate link, not just at injection time.  Without
    this, units that the LP schedules with a gap between consecutive hops
    (non-pipelined) would arrive at the next link too early and displace units
    that the LP intended to fire there first.
    """
    # Build unit_id -> Flow (same traversal order as _unroll_units in ilp.py)
    unit_to_flow: dict[int, Flow] = {}
    uid = 0
    for f in flows:
        for _ in range(f.size):
            unit_to_flow[uid] = f
            uid += 1

    # Extract per-unit per-hop firing step from the LP assignment
    unit_hop_steps: dict[int, dict[int, int]] = defaultdict(dict)
    for (uid, hop_idx, t), val in assignment.items():
        if val > 0.5:
            unit_hop_steps[uid][hop_idx] = t

    injections: list[Injection] = []
    for unit_id, f in unit_to_flow.items():
        hops = unit_hop_steps.get(unit_id, {})
        # first-hop firing time = start_step for the simulator
        start = hops.get(0, 0)
        # Build the ordered tuple of LP fire steps for all hops
        if hops:
            max_hop = max(hops.keys())
            hop_schedule = tuple(hops.get(i, 0) for i in range(max_hop + 1))
        else:
            hop_schedule = ()
        injections.append(
            Injection(
                flow=f,
                start_step=start,
                priority=unit_id,  # stable tie-break when hop_schedule is absent
                hop_schedule=hop_schedule,
            )
        )
    return injections
