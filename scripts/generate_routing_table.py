"""Generate a routing-table JSON file for a {S, 2S}^n twisted-torus topology.

Output: matrix-of-paths JSON in the shape of fixtures/routing/routing_table_8x4x4_twist.json
(with `vc` omitted). Default destination: fixtures/routing/routing_table_<slice>_<router>.json.

Usage:
    python scripts/generate_routing_table.py --slice 4,4,8 --router ilp
    python scripts/generate_routing_table.py --slice 2,4 --router dor --out /tmp/rt.json
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Make `python scripts/generate_routing_table.py` work without install.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from twisted_analysis.io.routing_table import save_routing_table
from twisted_analysis.topology import Topology, DORRouter, ILPRouter


def _parse_slice(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def _build_router(name: str, topology: Topology):
    name = name.lower()
    if name == "ilp":
        return ILPRouter(topology=topology)
    if name == "dor":
        return DORRouter(topology=topology)
    raise ValueError(f"unknown router: {name!r} (choose ilp|dor)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Generate a routing-table JSON for a twisted-torus topology.",
    )
    p.add_argument("--slice", required=True,
                   help="Comma-separated slice, e.g. 4,4,8")
    p.add_argument("--router", default="ilp", choices=["ilp", "dor"])
    p.add_argument("--out", default=None,
                   help="Output path (default: ./fixtures/routing/routing_table_<slice>_<router>.json)")
    args = p.parse_args(argv)

    slice_ = _parse_slice(args.slice)
    topology = Topology(slice=slice_)
    router = _build_router(args.router, topology)

    if args.out is None:
        slice_str = "x".join(str(s) for s in slice_)
        out_path = _HERE.parent / "fixtures" / "routing" / f"routing_table_{slice_str}_{args.router}.json"
    else:
        out_path = Path(args.out)

    save_routing_table(topology, router, out_path)
    print(f"wrote {out_path} ({out_path.stat().st_size:,} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
