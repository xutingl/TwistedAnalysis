"""Inspect the symmetric ILP solution to look for closed-form structure.

For 2x4 and 4x8 (with ILP routing), prints:
  - orbit_id (canonical dst-from-origin)
  - path: list of (dim, dir) hops
  - hop_schedule: t_0, t_1, ..., t_{L-1}
  - offset: t_0 (when does orbit fire first hop)
  - hop_gaps: t_{i+1} - t_i

Hypothesis 1: t_i = offset(orbit) + i   (pipelined; gap = 1)
Hypothesis 2: offset(orbit) is a simple function of canonical dst (e.g., bit-reversal, lex rank, manhattan-modulo).
"""
from __future__ import annotations

from twisted_analysis.lp.orbit import compute_orbits
from twisted_analysis.lp.symmetric import solve_symmetric_makespan
from twisted_analysis.model.flow import AllToAll
from twisted_analysis.topology import ILPRouter, Topology


def inspect(slice_: tuple[int, ...]) -> None:
    t = Topology(slice=slice_)
    r = ILPRouter(t)
    w = AllToAll(t, r, msg_size=1)
    lb = w.lower_bound

    print("=" * 78)
    print(f"Topology: {slice_}    N={t.n_nodes}    LB={lb}")
    print("=" * 78)

    # T_upper: 4*LB is generous
    makespan, assignment = solve_symmetric_makespan(t, r, list(w.flows), T_upper=4 * lb)
    print(f"Symmetric ILP makespan: {makespan}   gap from LB: {makespan - lb}")
    print()

    orbits = compute_orbits(t)
    origin = tuple([0] * t.ndim)

    # Collect orbit hop schedules
    by_orbit_hop: dict = {}
    for (orbit_id, hop_i, step), val in assignment.items():
        if val is not None and val > 0.5:
            by_orbit_hop.setdefault(orbit_id, {})[hop_i] = step

    # Pull canonical paths
    canon_path: dict = {}
    for orbit_id, members in orbits.items():
        canon = next(((s, d) for (s, d) in members if s == origin), None)
        assert canon is not None
        canon_path[orbit_id] = r.path(canon[0], canon[1])

    # Sort orbits by total path length, then by orbit_id for stability.
    def sort_key(oid):
        p = canon_path[oid]
        return (len(p), oid)

    print(f"{'orbit_id':<14} {'len':>3} {'path':<28} {'offset':>6} {'schedule':<22} {'gaps':<12}")
    print("-" * 90)
    rows = []
    for orbit_id in sorted(orbits.keys(), key=sort_key):
        path = canon_path[orbit_id]
        hops = by_orbit_hop.get(orbit_id, {})
        schedule = [hops[i] for i in sorted(hops.keys())]
        offset = schedule[0] if schedule else None
        gaps = [schedule[i + 1] - schedule[i] for i in range(len(schedule) - 1)]
        path_str = "".join(f"({dim},{'+' if dr > 0 else '-'})" for _, _, dim, dr in path)
        sched_str = "[" + ",".join(str(s) for s in schedule) + "]"
        gap_str = "[" + ",".join(str(g) for g in gaps) + "]"
        rows.append((orbit_id, len(path), path_str, offset, schedule, gaps))
        print(f"{str(orbit_id):<14} {len(path):>3} {path_str:<28} {offset!s:>6} {sched_str:<22} {gap_str:<12}")
    print()

    # Aggregate stats.
    pipelined = sum(1 for _, _, _, _, sched, gaps in rows if all(g == 1 for g in gaps))
    total = len(rows)
    print(f"Pipelined (all gaps==1): {pipelined}/{total}")
    one_hop = [r for r in rows if r[1] == 1]
    print(f"One-hop orbits ({len(one_hop)}): offsets = "
          f"{sorted(set(r[3] for r in one_hop))}")
    print()

    # Try hypothesis: offset(orbit) depends only on first-hop's (dim,dir).
    print("offset distribution by *first hop* (dim,dir):")
    by_first = {}
    for orbit_id, _, _, offset, _, _ in rows:
        path = canon_path[orbit_id]
        if not path:
            continue
        first = (path[0][2], path[0][3])
        by_first.setdefault(first, []).append(offset)
    for k, v in sorted(by_first.items()):
        print(f"  first-hop {k}: offsets={sorted(set(v))}  counts="
              f"{ {x: v.count(x) for x in sorted(set(v))} }")
    print()

    # Try hypothesis: offset depends on canonical dst hash function.
    print("offset distribution by orbit_id (canonical dst):")
    by_dst = {}
    for orbit_id, _, _, offset, _, _ in rows:
        by_dst.setdefault(orbit_id, offset)
    print(f"  unique offsets across {len(by_dst)} orbits: "
          f"{sorted(set(by_dst.values()))}")
    print()


if __name__ == "__main__":
    inspect((2, 4))
    inspect((4, 8))
