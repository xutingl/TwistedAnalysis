"""End-to-end test for gen_orbit_greedy_kernel.py orchestration.

Verifies:
  * Generating from --slice + --router produces both intermediate files
    AND the kernel file.
  * Generating from an existing --routing-table reuses the table and
    produces a schedule + kernel.
  * The generated kernel parses as Python.
"""
from __future__ import annotations
import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "pallas_kernel" / "gen_orbit_greedy_kernel.py"


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(REPO), capture_output=True, text=True,
    )


def test_pipeline_from_router_writes_all_three_artifacts(tmp_path: Path):
    rt_out = tmp_path / "rt.json"
    sched_out = tmp_path / "sched.json"
    kernel_out = tmp_path / "kernel.py"
    res = _run([
        "--slice", "2,4",
        "--router", "dor",
        "--routing-table-out", str(rt_out),
        "--schedule-out", str(sched_out),
        "--out", str(kernel_out),
    ])
    assert res.returncode == 0, res.stderr
    assert rt_out.exists()
    assert sched_out.exists()
    assert kernel_out.exists()
    ast.parse(kernel_out.read_text())


def test_pipeline_from_existing_routing_table_does_not_overwrite(tmp_path: Path):
    rt_in = tmp_path / "rt_in.json"
    sched_out = tmp_path / "sched.json"
    kernel_out = tmp_path / "kernel.py"

    res0 = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "generate_routing_table.py"),
         "--slice", "2,4", "--router", "dor", "--out", str(rt_in)],
        cwd=str(REPO), capture_output=True, text=True,
    )
    assert res0.returncode == 0, res0.stderr
    rt_mtime_before = rt_in.stat().st_mtime_ns

    res = _run([
        "--slice", "2,4",
        "--routing-table", str(rt_in),
        "--schedule-out", str(sched_out),
        "--out", str(kernel_out),
    ])
    assert res.returncode == 0, res.stderr
    assert rt_in.stat().st_mtime_ns == rt_mtime_before
    assert sched_out.exists()
    assert kernel_out.exists()
    ast.parse(kernel_out.read_text())


def test_dest_table_from_minimal_schedule():
    """Hand-crafted schedule probe: lock down the helper's contract.

    Without round-tripping through subprocess, asserts that the helper:
      - Sorts each src's entries by (round, dst).
      - Produces a _DEST_TABLE_NP whose every row is a permutation of {0..n-1}\\{src}.
      - Produces _ORBIT_STEPS that bucket columns by hop-0 round (per src=0).
    """
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "pallas_kernel"))
    from gen_orbit_greedy_kernel import _dest_table_and_orbit_steps_from_schedule

    # n=3 mini schedule. src=0 -> {1, 2}, src=1 -> {0, 2}, src=2 -> {0, 1}.
    sched = [
        {"round": 0, "src": 0, "dst": 1, "path": [0, 1]},
        {"round": 0, "src": 1, "dst": 0, "path": [1, 0]},
        {"round": 0, "src": 2, "dst": 1, "path": [2, 1]},
        {"round": 1, "src": 0, "dst": 2, "path": [0, 2]},
        {"round": 1, "src": 1, "dst": 2, "path": [1, 2]},
        {"round": 1, "src": 2, "dst": 0, "path": [2, 0]},
    ]
    dt, steps = _dest_table_and_orbit_steps_from_schedule(sched, n=3)
    # Each row is a permutation of {0..n-1}\\{src}.
    for src in range(3):
        assert set(int(x) for x in dt[src]) == set(range(3)) - {src}
    # _ORBIT_STEPS buckets columns by round; src=0 has rounds [0, 1] so steps = [[0], [1]].
    assert steps == [[0], [1]]


def test_dest_table_raises_on_incomplete_schedule():
    """Schedule missing entries for some src should fail with a descriptive RuntimeError."""
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "pallas_kernel"))
    from gen_orbit_greedy_kernel import _dest_table_and_orbit_steps_from_schedule

    sched = [
        {"round": 0, "src": 0, "dst": 1, "path": [0, 1]},
    ]  # missing entries for src=1, src=2; src=0 only has 1 entry, expected 2
    with pytest.raises(RuntimeError, match="full AllToAll"):
        _dest_table_and_orbit_steps_from_schedule(sched, n=3)


def test_pipeline_from_routing_table_8x4x4_twist_fixture(tmp_path: Path):
    """Exercise the example from the spec: load the existing 8x4x4 fixture
    and produce a schedule + kernel."""
    rt = REPO / "fixtures" / "routing_table_8x4x4_twist.json"
    assert rt.exists()
    sched_out = tmp_path / "sched_8x4x4.json"
    kernel_out = tmp_path / "kernel_8x4x4.py"
    res = _run([
        "--slice", "8,4,4",
        "--routing-table", str(rt),
        "--schedule-out", str(sched_out),
        "--out", str(kernel_out),
    ])
    assert res.returncode == 0, res.stderr
    assert sched_out.exists()
    assert kernel_out.exists()
    ast.parse(kernel_out.read_text())
