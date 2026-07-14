"""Ragged-workload on-disk I/O.

Format: top-level list of `{"src": int, "dst": int, "size": int}` dicts
(matches fixtures/ragged/ragged_a2a_workload_node_128_min_32_max_1024_discrete.json).
Pair order in the file is preserved in the returned demand dict; the
`natural` scheduler order iterates it.
"""
from __future__ import annotations
import json
from pathlib import Path

from twisted_analysis.model.ragged import RaggedWorkload


def load_workload(path: Path | str) -> RaggedWorkload:
    path = Path(path)
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(
            f"{path}: top-level must be a list, got {type(raw).__name__}"
        )
    demand: dict[tuple[int, int], int] = {}
    for i, e in enumerate(raw):
        if not isinstance(e, dict):
            raise ValueError(
                f"{path}: entry {i} must be a dict, got {type(e).__name__}"
            )
        for k in ("src", "dst", "size"):
            if k not in e:
                raise ValueError(f"{path}: entry {i} missing {k!r}")
            if not isinstance(e[k], int) or isinstance(e[k], bool):
                raise ValueError(
                    f"{path}: entry {i}: {k}={e[k]!r} must be int, "
                    f"got {type(e[k]).__name__}"
                )
        key = (e["src"], e["dst"])
        if key in demand:
            raise ValueError(f"{path}: duplicate pair {key} at entry {i}")
        demand[key] = e["size"]
    return RaggedWorkload(demand=demand)
