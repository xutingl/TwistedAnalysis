import csv
from pathlib import Path

from twisted_analysis.topology import Topology, DORRouter, ILPRouter

FIXT = Path(__file__).parent.parent / "fixtures"


def _load(name: str) -> list[tuple[str, str, int, str]]:
    rows = []
    with (FIXT / f"routing_{name}.csv").open() as f:
        r = csv.reader(f)
        next(r)  # header
        for row in r:
            rows.append((row[0], row[1], int(row[2]), row[3]))
    return rows


def test_2x4_fixture_matches_router():
    t = Topology(slice=(2, 4))
    r = DORRouter(t)
    for src_s, dst_s, hops, path_str in _load("2x4"):
        src = eval(src_s)
        dst = eval(dst_s)
        path = r.path(src, dst)
        assert len(path) == hops


def test_4x8_fixture_matches_router():
    t = Topology(slice=(4, 8))
    r = DORRouter(t)
    for src_s, dst_s, hops, _ in _load("4x8"):
        assert len(r.path(eval(src_s), eval(dst_s))) == hops


def test_2x4_ilp_fixture_matches_router():
    t = Topology(slice=(2, 4))
    r = ILPRouter(t)
    for src_s, dst_s, hops, _ in _load("ilp_2x4"):
        assert len(r.path(eval(src_s), eval(dst_s))) == hops


def test_4x8_ilp_fixture_matches_router():
    t = Topology(slice=(4, 8))
    r = ILPRouter(t)
    for src_s, dst_s, hops, _ in _load("ilp_4x8"):
        assert len(r.path(eval(src_s), eval(dst_s))) == hops
