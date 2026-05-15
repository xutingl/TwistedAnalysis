"""Schedule JSON save/load.

Format: list of dicts {"round": int, "src": int, "dst": int, "path": [int, ...]}.
"""
from __future__ import annotations
import json
from pathlib import Path

import pytest

from twisted_analysis.io.schedule import save_schedule, load_schedule


def test_save_schedule_writes_list_of_dicts(tmp_path: Path):
    entries = [
        {"round": 0, "src": 0, "dst": 1, "path": [0, 1]},
        {"round": 0, "src": 1, "dst": 0, "path": [1, 0]},
        {"round": 1, "src": 0, "dst": 2, "path": [0, 1, 2]},
    ]
    p = tmp_path / "sched.json"
    save_schedule(entries, p)
    raw = json.loads(p.read_text())
    assert raw == entries


def test_load_schedule_roundtrip(tmp_path: Path):
    entries = [
        {"round": 0, "src": 0, "dst": 42,
         "path": [0, 16, 32, 36, 40, 41, 42]},
    ]
    p = tmp_path / "sched.json"
    save_schedule(entries, p)
    out = load_schedule(p)
    assert out == entries


def test_save_schedule_validates_required_keys(tmp_path: Path):
    bad = [{"round": 0, "src": 0, "dst": 1}]  # missing 'path'
    with pytest.raises(ValueError, match="path"):
        save_schedule(bad, tmp_path / "x.json")


def test_save_schedule_validates_path_endpoints(tmp_path: Path):
    bad = [{"round": 0, "src": 0, "dst": 5, "path": [0, 1, 2]}]  # last != dst
    with pytest.raises(ValueError, match="dst"):
        save_schedule(bad, tmp_path / "x.json")


def test_save_schedule_creates_parent_dirs(tmp_path: Path):
    p = tmp_path / "nested" / "deep" / "sched.json"
    save_schedule(
        [{"round": 0, "src": 0, "dst": 1, "path": [0, 1]}],
        p,
    )
    assert p.exists()


def test_load_schedule_rejects_non_dict_entry(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps([42]))
    with pytest.raises(ValueError, match="not a dict"):
        load_schedule(p)


def test_save_schedule_rejects_non_int_round(tmp_path: Path):
    bad = [{"round": "0", "src": 0, "dst": 1, "path": [0, 1]}]
    with pytest.raises(ValueError, match="round"):
        save_schedule(bad, tmp_path / "x.json")


def test_save_schedule_rejects_non_int_path_element(tmp_path: Path):
    bad = [{"round": 0, "src": 0, "dst": 1, "path": [0, "1"]}]
    with pytest.raises(ValueError, match="path"):
        save_schedule(bad, tmp_path / "x.json")


def test_save_schedule_rejects_bool_round(tmp_path: Path):
    """Python bool is a subtype of int; explicitly reject."""
    bad = [{"round": True, "src": 0, "dst": 1, "path": [0, 1]}]
    with pytest.raises(ValueError, match="round"):
        save_schedule(bad, tmp_path / "x.json")
