"""Generate a schedule JSON from a routing table.

Currently supports only `--scheduler orbit_greedy`. Extend the dispatch dict
in `_run` to add more schedulers.

Usage:
    python scripts/generate_schedule.py \\
        --routing-table fixtures/routing_table_4x4x8_twist.json \\
        --slice 4,4,8 \\
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
from twisted_analysis.io.schedule import save_schedule, schedule_from_orbit_greedy
from twisted_analysis.topology import Topology


def _parse_slice(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def _run(scheduler: str, topology: Topology, table: list, order: str) -> list[dict]:
    if scheduler == "orbit_greedy":
        return schedule_from_orbit_greedy(topology, table, order=order)
    raise ValueError(f"unknown scheduler: {scheduler!r}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Generate a schedule JSON from a routing-table JSON.",
    )
    p.add_argument("--routing-table", required=True, type=Path,
                   help="Path to routing-table JSON (matrix-of-paths shape)")
    p.add_argument("--slice", required=True,
                   help="Comma-separated slice, e.g. 4,4,8 — must match the table size")
    p.add_argument("--scheduler", default="orbit_greedy", choices=["orbit_greedy"])
    p.add_argument("--order", default="lpt_tail_asc",
                   choices=["lpt_tail_asc", "lpt", "spt", "tail_asc"])
    p.add_argument("--out", default=None,
                   help="Output path (default: ./fixtures/schedule_<slice>_<scheduler>_<order>.json)")
    args = p.parse_args(argv)

    slice_ = _parse_slice(args.slice)
    topology = Topology(slice=slice_)
    table = load_routing_table(args.routing_table)
    if len(table) != topology.n_nodes:
        raise SystemExit(
            f"routing table has {len(table)} sources; slice {slice_} expects {topology.n_nodes}"
        )

    entries = _run(args.scheduler, topology, table, args.order)

    if args.out is None:
        slice_str = "x".join(str(s) for s in slice_)
        out_path = _HERE.parent / "fixtures" / (
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
