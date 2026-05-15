"""Regression: the orbit_greedy schedule baked against the LOADED
8x4x4 routing table has physical-edge capacity violations.

This documents that orbit_greedy's (dim, dir) keying is unsound for
routings that are not strictly translation-equivariant under that key
(see docs/superpowers/plans/2026-05-15-multi-algorithm-scheduling.md
Task 3 for the explanation).
"""
from pathlib import Path

from twisted_analysis.io.schedule import load_schedule
from twisted_analysis.schedules.verify import verify_capacity

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def test_loaded_8x4x4_orbit_greedy_has_violations():
    schedule = load_schedule(FIXTURES / "schedule_8x4x4_loaded_lpt_tail_asc.json")
    violations = verify_capacity(schedule)
    # The exact count was 640 at the time this regression was added; we assert
    # "many" rather than the exact number to avoid brittleness if greedy
    # tie-breaking changes harmlessly.
    assert len(violations) >= 100, (
        f"Expected loaded-routing orbit_greedy to have >=100 physical-edge "
        f"violations under (dim, dir) keying; got {len(violations)}. "
        f"If this test now passes with 0, document why and update the plan."
    )
