"""Schedule on-disk I/O.

Format: list of dicts, one per (round, src, dst) triple. Each dict has at
least the keys: {"round": int, "src": int, "dst": int, "path": [int, ...]}.

`src` and `dst` are flat-IDs; `path` is the sequence of flat-IDs traversed
from src to dst (inclusive of both endpoints).
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable, Mapping

ScheduleEntry = Mapping[str, object]
_REQUIRED = ("round", "src", "dst", "path")


def _validate(entries: Iterable[ScheduleEntry]) -> list[dict]:
    out = []
    for i, e in enumerate(entries):
        for k in _REQUIRED:
            if k not in e:
                raise ValueError(f"entry {i} missing required key {k!r}: {dict(e)}")
        path = e["path"]
        if not isinstance(path, list) or not path:
            raise ValueError(f"entry {i}: path must be non-empty list")
        if path[0] != e["src"]:
            raise ValueError(
                f"entry {i}: path[0]={path[0]} != src={e['src']}"
            )
        if path[-1] != e["dst"]:
            raise ValueError(
                f"entry {i}: path[-1]={path[-1]} != dst={e['dst']}"
            )
        out.append(dict(e))
    return out


def save_schedule(entries: Iterable[ScheduleEntry], out_path: Path | str) -> None:
    out_path = Path(out_path)
    validated = _validate(entries)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(validated, indent=2))


def load_schedule(path: Path | str) -> list[dict]:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list):
        raise ValueError(f"{path}: top-level must be a list")
    return _validate(raw)
