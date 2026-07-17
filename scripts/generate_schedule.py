"""Generate a schedule JSON from a routing table.

Currently supports only `--scheduler orbit_greedy`. Extend the dispatch dict
in `_run` to add more schedulers.

Usage:
    python scripts/generate_schedule.py \\
        --routing-table fixtures/routing/routing_table_8x4x4_twist.json \\
        --slice 8,4,4 \\
        --scheduler orbit_greedy \\
        --order lpt_tail_asc
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Make `python scripts/generate_schedule.py` work without install.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from twisted_analysis.io.routing_table import load_routing_table
from twisted_analysis.io.schedule import (
    save_schedule,
    schedule_from_orbit_greedy,
    schedule_from_orbit_pack,
    schedule_from_spread_greedy,
)
from twisted_analysis.topology import Topology


def _parse_slice(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def _run(
    scheduler: str,
    topology: Topology,
    table: list,
    order: str,
    *,
    k: int | None = None,
    c: int | None = None,
) -> list[dict]:
    if scheduler == "orbit_greedy":
        return schedule_from_orbit_greedy(topology, table, order=order)
    if scheduler == "spread_greedy":
        # k is guaranteed non-None here (validated before _run is called)
        return schedule_from_spread_greedy(topology, table, k=k, order=order)
    if scheduler == "orbit_pack":
        # k and c are guaranteed non-None here (validated before _run).
        # order is ignored: orbit_pack uses a fixed FFD ordering.
        return schedule_from_orbit_pack(topology, table, k=k, c=c)
    raise ValueError(f"unknown scheduler: {scheduler!r}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Generate a schedule JSON from a routing-table JSON.",
    )
    p.add_argument("--routing-table", required=True, type=Path,
                   help="Path to routing-table JSON (matrix-of-paths shape)")
    p.add_argument("--slice", required=True,
                   help="Comma-separated slice, e.g. 4,4,8 — must match the table size")
    p.add_argument("--scheduler", default="orbit_greedy",
                   choices=["orbit_greedy", "spread_greedy", "orbit_pack"])
    p.add_argument("--order", default="lpt_tail_asc",
                   choices=["lpt_tail_asc", "lpt", "spt", "tail_asc"])
    p.add_argument(
        "--k",
        type=int,
        default=None,
        help="Per-device-per-round DMA cap for spread_greedy (positive int), "
             "or max orbits per step for orbit_pack. "
             "Required when --scheduler spread_greedy or orbit_pack.",
    )
    p.add_argument(
        "--c",
        type=int,
        default=None,
        help="Max whole-path edge load per barrier step for orbit_pack "
             "(positive int; >= hottest orbit's internal load — 3 on the "
             "loaded 8x4x4 routing). Required when --scheduler orbit_pack. "
             "orbit_pack schedules are step-model: verify with "
             "verify_capacity_step, not verify_capacity.",
    )
    p.add_argument("--out", default=None,
                   help="Output path (default: ./fixtures/nonragged/schedule_<slice>_<scheduler>_<order>.json)")
    args = p.parse_args(argv)

    # Validate scheduler-specific required args.
    if args.scheduler == "spread_greedy" and args.k is None:
        p.error("--k is required when --scheduler spread_greedy")
    if args.scheduler == "orbit_pack":
        if args.k is None or args.c is None:
            p.error("--k and --c are required when --scheduler orbit_pack")

    slice_ = _parse_slice(args.slice)
    topology = Topology(slice=slice_)
    table = load_routing_table(args.routing_table)
    if len(table) != topology.n_nodes:
        raise SystemExit(
            f"routing table has {len(table)} sources; slice {slice_} expects {topology.n_nodes}"
        )

    try:
        entries = _run(args.scheduler, topology, table, args.order,
                       k=args.k, c=args.c)
    except ValueError as e:
        raise SystemExit(
            f"scheduler failed: {e}\n"
            f"hint: this CLI requires routing tables whose paths are sequences of "
            f"single-hop topology neighbors. If your routing table contains multi-hop "
            f"steps (e.g. paths from external tools), regenerate it via "
            f"scripts/generate_routing_table.py."
        )

    if args.out is None:
        slice_str = "x".join(str(s) for s in slice_)
        out_path = _HERE.parent / "fixtures" / "nonragged" / (
            f"schedule_{slice_str}_{args.scheduler}_{args.order}.json"
        )
    else:
        out_path = Path(args.out)

    save_schedule(entries, out_path)
    print(f"wrote {out_path} ({out_path.stat().st_size:,} bytes, "
          f"{len(entries):,} entries)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
