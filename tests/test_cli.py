import json
import subprocess
import sys
from pathlib import Path


def test_cli_runs_2x4_rr(tmp_path):
    # Run CLI as a module; expect a summary file in tmp_path
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        f"name: smoke\nslice: [2, 4]\nmsg_size: 1\nschedule: round_robin\n"
        f"output_dir: {tmp_path}/out\n"
    )
    res = subprocess.run(
        [sys.executable, "-m", "twisted_analysis.cli", "run", str(cfg)],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
    out_dir = tmp_path / "out"
    assert (out_dir / "summary.json").exists()


def test_cli_uses_ilp_router_by_default(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "name: smoke_ilp\nslice: [2, 4]\nmsg_size: 1\nschedule: round_robin\n"
        f"output_dir: {tmp_path}/out\n"
    )
    res = subprocess.run(
        [sys.executable, "-m", "twisted_analysis.cli", "run", str(cfg)],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["router"] == "ilp"


def test_cli_uses_dor_router_when_requested(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "name: smoke_dor\nslice: [2, 4]\nmsg_size: 1\nschedule: round_robin\n"
        f"router: dor\noutput_dir: {tmp_path}/out\n"
    )
    res = subprocess.run(
        [sys.executable, "-m", "twisted_analysis.cli", "run", str(cfg)],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["router"] == "dor"
