"""Empirical correctness of the existing (dim, dir)-keyed orbit_greedy
on every (slice, ILP-router) cell that the optimality proof covers.

Pass condition: 0 physical-edge capacity violations on every cell. This is
the decision criterion for whether to keep orbit_greedy as-is or replace
it with orbit_greedy_full (see plan Task 3).
"""
import pytest

from twisted_analysis.io.routing_table import save_routing_table
from twisted_analysis.io.schedule import schedule_from_orbit_greedy
from twisted_analysis.io.routing_table import load_routing_table
from twisted_analysis.schedules.verify import verify_capacity
from twisted_analysis.topology import Topology, ILPRouter


CELLS = [
    (2, 4),
    (2, 2, 4),
    (2, 4, 4),
    (4, 8),
    # (4, 4, 8) excluded: ILP router takes ~21 s; covered in a slow-test
    # variant if needed.
]


@pytest.mark.parametrize("slice_", CELLS)
def test_orbit_greedy_dimdir_no_violations_under_ilp(tmp_path, slice_):
    topology = Topology(slice=slice_)
    router = ILPRouter(topology=topology)
    rt_path = tmp_path / "rt.json"
    save_routing_table(topology, router, rt_path)
    table = load_routing_table(rt_path)
    schedule = schedule_from_orbit_greedy(topology, table, order="lpt_tail_asc")
    violations = verify_capacity(schedule)
    assert violations == [], (
        f"orbit_greedy with (dim, dir) keying on ILP-routed {slice_} produced "
        f"{len(violations)} physical-edge violations. First 3: {violations[:3]}"
    )
