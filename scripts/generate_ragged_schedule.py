"""Generate a ragged-workload schedule JSON from a routing table + workload.

Usage:
    python scripts/generate_ragged_schedule.py \\
        --routing-table fixtures/routing/routing_table_8x4x4_twist.json \\
        --slice 8,4,4 \\
        --workload fixtures/ragged/ragged_a2a_workload_node_128_min_32_max_1024_discrete.json \\
        --scheduler ragged_greedy --order lpt [--preemptive]

Verifies capacity and workload coverage before writing; exits non-zero on
any violation. Optionally appends a metrics row via --csv-append (header:
scheduler,order,preemptive,lb_quanta,makespan_quanta,gap_pct,entries,max_chunks_per_flow).
"""
from __future__ import annotations
import argparse
import sys
from collections import Counter
from pathlib import Path

# Make `python scripts/generate_ragged_schedule.py` work without install.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from twisted_analysis.io.routing_table import load_routing_table
from twisted_analysis.io.schedule import save_schedule, schedule_from_algorithm
from twisted_analysis.io.workload import load_workload
from twisted_analysis.schedules.verify import (
    schedule_makespan_ragged,
    verify_capacity_ragged,
    verify_workload_coverage,
)
from twisted_analysis.topology import Topology

CSV_HEADER = (
    "scheduler,order,preemptive,lb_quanta,makespan_quanta,"
    "gap_pct,entries,max_chunks_per_flow"
)


def _parse_slice(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Generate a ragged-A2A schedule JSON.",
    )
    p.add_argument("--routing-table", required=True, type=Path)
    p.add_argument("--slice", required=True,
                   help="Comma-separated slice, e.g. 8,4,4")
    p.add_argument("--workload", required=True, type=Path,
                   help="Ragged workload JSON ([{src, dst, size}, ...])")
    p.add_argument("--scheduler", required=True,
                   choices=["ragged_fluid", "ragged_greedy"])
    p.add_argument("--order", default="lpt",
                   choices=["lpt", "spt", "natural"],
                   help="Flow order for ragged_greedy (ignored by ragged_fluid)")
    p.add_argument("--preemptive", action="store_true",
                   help="ragged_greedy only: allow chunk splitting")
    p.add_argument("--out", default=None,
                   help="Output path (default: fixtures/ragged/schedule_<slice>_loaded_"
                        "ragged_<fluid|greedy_<order>[_pre]>.json)")
    p.add_argument("--csv-append", default=None, type=Path,
                   help="Append a metrics row to this CSV (header written if new)")
    args = p.parse_args(argv)

    slice_ = _parse_slice(args.slice)
    topology = Topology(slice=slice_)
    table = load_routing_table(args.routing_table)
    if len(table) != topology.n_nodes:
        raise SystemExit(
            f"routing table has {len(table)} sources; "
            f"slice {slice_} expects {topology.n_nodes}"
        )
    workload = load_workload(args.workload)
    quantum = workload.quantum
    lb_quanta = workload.lower_bound(table) // quantum

    kwargs = {"workload": workload}
    if args.scheduler == "ragged_greedy":
        kwargs.update(order=args.order, preemptive=args.preemptive)
    entries = schedule_from_algorithm(args.scheduler, topology, table, **kwargs)

    violations = verify_capacity_ragged(entries, quantum=quantum)
    if violations:
        raise SystemExit(
            f"capacity check FAILED: {len(violations)} violations; "
            f"first: {violations[0]}"
        )
    problems = verify_workload_coverage(entries, workload)
    if problems:
        raise SystemExit(
            f"coverage check FAILED: {len(problems)} problems; "
            f"first: {problems[0]}"
        )

    makespan = schedule_makespan_ragged(entries, quantum=quantum)
    gap_pct = 100.0 * (makespan - lb_quanta) / lb_quanta
    chunks_per_flow = Counter((e["src"], e["dst"]) for e in entries)
    max_chunks = max(chunks_per_flow.values())

    if args.out is None:
        slice_str = "x".join(str(s) for s in slice_)
        suffix = ("fluid" if args.scheduler == "ragged_fluid"
                  else f"greedy_{args.order}" + ("_pre" if args.preemptive else ""))
        out_path = _HERE.parent / "fixtures" / "ragged" / (
            f"schedule_{slice_str}_loaded_ragged_{suffix}.json"
        )
    else:
        out_path = Path(args.out)
    save_schedule(entries, out_path)

    order_str = args.order if args.scheduler == "ragged_greedy" else ""
    print(
        f"{args.scheduler} order={order_str or '-'} preemptive={args.preemptive} "
        f"lb_quanta={lb_quanta} makespan_quanta={makespan:.2f} "
        f"gap_pct={gap_pct:.2f} entries={len(entries)} "
        f"max_chunks_per_flow={max_chunks}"
    )
    print(f"wrote {out_path}", file=sys.stderr)

    if args.csv_append is not None:
        is_new = not args.csv_append.exists()
        with args.csv_append.open("a") as f:
            if is_new:
                f.write(CSV_HEADER + "\n")
            f.write(
                f"{args.scheduler},{order_str},{args.preemptive},"
                f"{lb_quanta},{makespan:.2f},{gap_pct:.2f},"
                f"{len(entries)},{max_chunks}\n"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
