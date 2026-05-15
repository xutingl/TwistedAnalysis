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

from twisted_analysis.topology import Topology

ScheduleEntry = Mapping[str, object]
_REQUIRED = ("round", "src", "dst", "path")


def _validate(entries: Iterable[ScheduleEntry]) -> list[dict]:
    out = []
    for i, e in enumerate(entries):
        if not isinstance(e, Mapping):
            raise ValueError(f"entry {i}: not a dict (got {type(e).__name__})")
        for k in _REQUIRED:
            if k not in e:
                raise ValueError(f"entry {i} missing required key {k!r}: {dict(e)}")
        for k in ("round", "src", "dst"):
            if not isinstance(e[k], int) or isinstance(e[k], bool):
                raise ValueError(
                    f"entry {i}: {k}={e[k]!r} must be int, got {type(e[k]).__name__}"
                )
        path = e["path"]
        if not isinstance(path, list) or not path:
            raise ValueError(f"entry {i}: path must be non-empty list")
        for j, x in enumerate(path):
            if not isinstance(x, int) or isinstance(x, bool):
                raise ValueError(
                    f"entry {i}: path[{j}]={x!r} must be int, got {type(x).__name__}"
                )
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


def schedule_from_orbit_greedy(
    topology: Topology,
    table: list[list[list[int]]],
    *,
    order: str = "lpt_tail_asc",
) -> list[dict]:
    """Run the OrbitGreedy scheduler against a routing table; return entries.

    For each orbit O firing hop-0 at OrbitGreedy step `t_0^O`, emit one entry
    per source `s`:
        {"round": t_0^O, "src": flat(s), "dst": flat(s + δ_O),
         "path": [flat-IDs along the canonical path translated to s]}

    Total entries: `N * (N - 1)` for a full AllToAll. Entries are sorted by
    (round, src) for determinism.

    Raises ValueError if `table` is not an N×N matrix of non-empty int paths.
    """
    from twisted_analysis.io.coords import flatten
    from twisted_analysis.io.routing_table import (
        RoutingTableRouter, validate_routing_table_shape,
    )
    from twisted_analysis.lp.orbit import compute_orbits
    from twisted_analysis.schedules.orbit_greedy import (
        compute_hop0_firing_times,
    )

    validate_routing_table_shape(table, topology.n_nodes)

    rt_router = RoutingTableRouter(topology=topology, table=table)
    t0 = compute_hop0_firing_times(topology, rt_router, order)

    orbits = compute_orbits(topology)
    slice_ = topology.slice

    entries: list[dict] = []
    for orbit_id, members in orbits.items():
        round_t = int(t0[orbit_id])
        for (src, dst) in members:
            src_flat = flatten(src, slice_)
            dst_flat = flatten(dst, slice_)
            entries.append({
                "round": round_t,
                "src": src_flat,
                "dst": dst_flat,
                "path": list(table[src_flat][dst_flat]),
            })
    entries.sort(key=lambda e: (e["round"], e["src"]))
    return entries
