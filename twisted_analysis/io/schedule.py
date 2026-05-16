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


def schedule_from_orbit_greedy_full(
    topology: Topology,
    table: list[list[list[int]]],
    *,
    order: str = "lpt_tail_asc",
) -> list[dict]:
    """Adapter: orbit_greedy_full -> schedule entries."""
    from twisted_analysis.io.coords import flatten
    from twisted_analysis.io.routing_table import validate_routing_table_shape
    from twisted_analysis.lp.orbit import compute_orbits
    from twisted_analysis.schedules.orbit_greedy_full import (
        compute_hop0_firing_times_full,
    )

    validate_routing_table_shape(table, topology.n_nodes)
    t0 = compute_hop0_firing_times_full(topology, table, order=order)
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


def schedule_from_literal_greedy(
    topology: Topology,
    table: list[list[list[int]]],
    *,
    order: str = "lpt",
) -> list[dict]:
    """Adapter: literal_greedy -> schedule entries (already in correct format)."""
    from twisted_analysis.io.routing_table import validate_routing_table_shape
    from twisted_analysis.schedules.literal_greedy import literal_greedy

    validate_routing_table_shape(table, topology.n_nodes)
    return literal_greedy(topology, table, order=order)


def schedule_from_ilp_literal(
    topology: Topology,
    table: list[list[list[int]]],
    *,
    t_upper: int | None = None,
    time_limit_s: int = 600,
) -> list[dict]:
    """Adapter: ilp_literal -> schedule entries."""
    from twisted_analysis.io.routing_table import validate_routing_table_shape
    from twisted_analysis.schedules.ilp_literal import ilp_literal

    validate_routing_table_shape(table, topology.n_nodes)
    return ilp_literal(
        topology, table, t_upper=t_upper, time_limit_s=time_limit_s,
    )


def schedule_from_cpsat_literal(
    topology: Topology,
    table: list[list[list[int]]],
    *,
    t_upper: int,
    time_limit_s: int = 600,
    n_workers: int = 8,
    warm_start_schedule: list[dict] | None = None,
) -> list[dict]:
    """Adapter: cpsat_literal -> schedule entries.

    `warm_start_schedule`: when provided, prior schedule entries are fed
    to CP-SAT as variable hints (see cpsat_literal docstring).
    """
    from twisted_analysis.io.routing_table import validate_routing_table_shape
    from twisted_analysis.schedules.cpsat_literal import cpsat_literal

    validate_routing_table_shape(table, topology.n_nodes)
    return cpsat_literal(
        topology, table, t_upper=t_upper,
        time_limit_s=time_limit_s, n_workers=n_workers,
        warm_start_schedule=warm_start_schedule,
    )


def schedule_from_lp_rounding(
    topology: Topology,
    table: list[list[list[int]]],
    *,
    t_upper: int,
    n_trials: int = 100,
    seed: int = 0,
) -> list[dict]:
    """Adapter: lp_rounding -> schedule entries."""
    from twisted_analysis.io.routing_table import validate_routing_table_shape
    from twisted_analysis.schedules.lp_rounding import lp_rounding

    validate_routing_table_shape(table, topology.n_nodes)
    return lp_rounding(topology, table, t_upper=t_upper,
                       n_trials=n_trials, seed=seed)


def schedule_from_local_search(
    topology: Topology,
    table: list[list[list[int]]],
    *,
    seed_schedule: list[dict],
    max_iters: int = 1000,
) -> list[dict]:
    """Adapter: local_search_repair on a seed schedule."""
    from twisted_analysis.io.routing_table import validate_routing_table_shape
    from twisted_analysis.schedules.local_search import local_search_repair

    validate_routing_table_shape(table, topology.n_nodes)
    return local_search_repair(topology, table, seed_schedule, max_iters=max_iters)


_SCHEDULER_DISPATCH = {
    "orbit_greedy": schedule_from_orbit_greedy,
    "orbit_greedy_full": schedule_from_orbit_greedy_full,
    "literal_greedy": schedule_from_literal_greedy,
    "ilp_literal": schedule_from_ilp_literal,
    "cpsat_literal": schedule_from_cpsat_literal,
    "lp_rounding": schedule_from_lp_rounding,
    "local_search": schedule_from_local_search,
}


def schedule_from_algorithm(
    algorithm: str,
    topology: Topology,
    table: list[list[list[int]]],
    **kwargs,
) -> list[dict]:
    """Dispatch to the named scheduler.

    Available algorithms:
      - "orbit_greedy":      original, (dim, dir)-keyed orbit greedy.
        Provably correct on translation-equivariant routings only.
      - "orbit_greedy_full": orbit greedy with full physical-edge accounting.
        Correct under any translation-symmetric workload (including loaded TPU routings).
      - "literal_greedy":    LMR-style per-flow earliest-feasible greedy.
      - "ilp_literal":       exact ILP on literal flows (CBC). Small cells only.
      - "cpsat_literal":     exact CP-SAT on literal flows (OR-Tools). Faster
        than ilp_literal on this structure due to native at-most-one constraints
        and parallel search workers. Requires `t_upper` kwarg.
      - "lp_rounding":       LP relaxation of the literal ILP + randomized rounding.
        Polynomial-time alternative to CP-SAT for large cells where CP-SAT may time
        out. Requires `t_upper` kwarg; optional `n_trials` (default 100) and `seed`.
      - "local_search":      Hill-climbing post-processor on a feasible seed schedule.
        Shifts makespan-defining flows earlier when feasible. Polynomial per iteration.
        No LB guarantee but cheap to chain after any greedy or LP-rounding output.
        Requires `seed_schedule` kwarg; optional `max_iters` (default 1000).

    Per-algorithm kwargs (e.g., `order`, `time_limit_s`, `t_upper`) are passed through.
    """
    if algorithm not in _SCHEDULER_DISPATCH:
        raise ValueError(
            f"unknown algorithm: {algorithm!r}; "
            f"choices: {sorted(_SCHEDULER_DISPATCH)}"
        )
    return _SCHEDULER_DISPATCH[algorithm](topology, table, **kwargs)
