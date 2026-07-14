"""CLI smoke test on the real fixture (fluid only — greedy is covered by
unit tests and is too slow at full fixture scale for the test suite)."""
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def test_cli_fluid_end_to_end(tmp_path, capsys):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import generate_ragged_schedule

    out = tmp_path / "sched.json"
    csv = tmp_path / "metrics.csv"
    rc = generate_ragged_schedule.main([
        "--routing-table", str(FIXTURES / "routing_table_8x4x4_twist.json"),
        "--slice", "8,4,4",
        "--workload", str(
            FIXTURES / "ragged_a2a_workload_node_128_min_32_max_1024_discrete.json"
        ),
        "--scheduler", "ragged_fluid",
        "--out", str(out),
        "--csv-append", str(csv),
    ])
    assert rc == 0
    assert out.exists()

    captured = capsys.readouterr().out
    assert "lb_quanta=394" in captured
    assert "makespan_quanta=399.00" in captured
    assert "entries=16256" in captured

    lines = csv.read_text().strip().splitlines()
    assert lines[0].startswith("scheduler,order,preemptive,lb_quanta")
    assert lines[1].startswith("ragged_fluid,,False,394,399.00")
