"""Dump routing tables to CSV for inspection and as test fixtures."""
import csv
import sys
from pathlib import Path

from twisted_analysis.topology import Topology, Router

OUT = Path(__file__).parent.parent / "fixtures"


def dump(slice_: tuple[int, ...], name: str) -> None:
    t = Topology(slice=slice_)
    r = Router(t)
    out_path = OUT / f"routing_{name}.csv"
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src", "dst", "hops", "path"])
        for s in t.nodes():
            for d in t.nodes():
                path = r.path(s, d)
                path_str = "|".join(f"{u}->{v}({dim},{dir})" for u, v, dim, dir in path)
                w.writerow([str(s), str(d), len(path), path_str])
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    dump((2, 4), "2x4")
    dump((4, 8), "4x8")
    # 4x4x8 is large; gate behind explicit flag.
    if "--include-3d" in sys.argv:
        dump((4, 4, 8), "4x4x8")
