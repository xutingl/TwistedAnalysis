"""Dump routing tables to CSV for inspection and as test fixtures."""
import csv
import sys
from pathlib import Path

from twisted_analysis.topology import Topology, DORRouter, ILPRouter

OUT = Path(__file__).parent.parent / "fixtures" / "routing"


def dump(slice_: tuple[int, ...], name: str, router_kind: str) -> None:
    t = Topology(slice=slice_)
    if router_kind == "dor":
        r = DORRouter(t)
        prefix = "routing"
    elif router_kind == "ilp":
        r = ILPRouter(t)
        prefix = "routing_ilp"
    else:
        raise ValueError(router_kind)
    out_path = OUT / f"{prefix}_{name}.csv"
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src", "dst", "hops", "path"])
        for s in t.nodes():
            for d in t.nodes():
                path = r.path(s, d)
                path_str = "|".join(f"{u}->{v}({dim},{dir})"
                                    for u, v, dim, dir in path)
                w.writerow([str(s), str(d), len(path), path_str])
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    dump((2, 4), "2x4", "dor")
    dump((4, 8), "4x8", "dor")
    dump((2, 4), "2x4", "ilp")
    dump((4, 8), "4x8", "ilp")
    if "--include-3d" in sys.argv:
        dump((4, 4, 8), "4x4x8", "dor")
        dump((4, 4, 8), "4x4x8", "ilp")
