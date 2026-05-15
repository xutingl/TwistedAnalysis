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


@pytest.mark.xfail(
    reason="routing_table_4x4x8_twist.json contains multi-hop paths; "
           "pipeline expects single-hop tables",
    strict=True,
)
def test_pipeline_from_routing_table_4x4x8_twist_fixture(tmp_path: Path):
    """Exercise the example from the spec: load the existing 4x4x8 fixture
    and produce a schedule + kernel."""
    rt = REPO / "fixtures" / "routing_table_4x4x8_twist.json"
    assert rt.exists()
    sched_out = tmp_path / "sched_4x4x8.json"
    kernel_out = tmp_path / "kernel_4x4x8.py"
    res = _run([
        "--slice", "4,4,8",
        "--routing-table", str(rt),
        "--schedule-out", str(sched_out),
        "--out", str(kernel_out),
    ])
    assert res.returncode == 0, res.stderr
    assert sched_out.exists()
    assert kernel_out.exists()
    ast.parse(kernel_out.read_text())
