from pathlib import Path

import pytest

from twisted_analysis.io.routing_table import load_routing_table, save_routing_table
from twisted_analysis.schedules.orbit_greedy_full import (
    compute_hop0_firing_times_full,
)
from twisted_analysis.schedules.verify import verify_capacity, schedule_makespan
from twisted_analysis.topology import Topology, ILPRouter

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _schedule_from_full(topology, table, order="lpt_tail_asc"):
    # Inline adapter; the io/schedule.py dispatcher in Task 7 will subsume this.
    from twisted_analysis.io.coords import flatten
    from twisted_analysis.lp.orbit import compute_orbits
    t0 = compute_hop0_firing_times_full(topology, table, order=order)
    orbits = compute_orbits(topology)
    entries = []
    for orbit_id, members in orbits.items():
        r = int(t0[orbit_id])
        for src, dst in members:
            src_flat = flatten(src, topology.slice)
            dst_flat = flatten(dst, topology.slice)
            entries.append({"round": r, "src": src_flat, "dst": dst_flat,
                            "path": list(table[src_flat][dst_flat])})
    entries.sort(key=lambda e: (e["round"], e["src"]))
    return entries


def test_orbit_greedy_full_zero_violations_on_loaded_8x4x4():
    topology = Topology(slice=(8, 4, 4))
    table = load_routing_table(FIXTURES / "routing_table_8x4x4_twist.json")
    schedule = _schedule_from_full(topology, table)
    violations = verify_capacity(schedule)
    assert violations == [], (
        f"orbit_greedy_full produced {len(violations)} physical-edge "
        f"violations on loaded 8x4x4 routing. First: {violations[0]}"
    )


def test_orbit_greedy_full_zero_violations_on_ilp_4x8(tmp_path):
    topology = Topology(slice=(4, 8))
    router = ILPRouter(topology=topology)
    rt_path = tmp_path / "rt.json"
    save_routing_table(topology, router, rt_path)
    table = load_routing_table(rt_path)
    schedule = _schedule_from_full(topology, table)
    assert verify_capacity(schedule) == []
    # On ILP-routed cells the LB-tight makespan should match docs/results.md.
    # For 4x8 ILP, LB = 21; makespan_full <= LB + diameter = 21 + 4 = 25.
    assert schedule_makespan(schedule) <= 25
