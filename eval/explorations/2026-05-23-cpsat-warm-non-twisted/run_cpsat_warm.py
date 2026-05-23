"""Per-cell driver: orbit_greedy_full seed -> CP-SAT warm-started sweep.

Pipeline (single cell):
  1. Load routing table.
  2. Compute LB = max physical-edge load over the routing.
  3. Seed = orbit_greedy_full(table, order="lpt_tail_asc").
  4. CP-SAT at t_upper=LB, warm-started from seed. If feasible, that's optimal.
  5. On INFEASIBLE at LB, sweep t_upper in {LB+1, ..., M_seed-1} warm-started
     from running best. First feasible wins.
  6. Save best schedule to --out-schedule. Append a row to --results-json.

Usage:
    .venv/bin/python -u eval/explorations/2026-05-23-cpsat-warm-non-twisted/run_cpsat_warm.py \\
        --routing-table fixtures/routing_table_torus_2x2x4.json \\
        --slice 2,2,4 \\
        --out-schedule fixtures/schedule_torus_2x2x4_cpsat_literal_warm.json \\
        --time-limit-s 300 \\
        --results-json eval/explorations/2026-05-23-cpsat-warm-non-twisted/results.json
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

# Make `python eval/.../run_cpsat_warm.py` work without install.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_ROOT))

from twisted_analysis.topology import Topology
from twisted_analysis.io.routing_table import load_routing_table
from twisted_analysis.io.schedule import save_schedule, schedule_from_orbit_greedy_full
from twisted_analysis.schedules.cpsat_literal import cpsat_literal
from twisted_analysis.schedules.verify import verify_capacity, schedule_makespan


def physical_edge_lb(table, n: int) -> int:
    c: Counter = Counter()
    for s in range(n):
        for d in range(n):
            if s == d:
                continue
            p = table[s][d]
            for h in range(len(p) - 1):
                c[(p[h], p[h + 1])] += 1
    return max(c.values())


def parse_slice(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def append_result_row(results_json: Path, row: dict) -> None:
    if results_json.exists():
        rows = json.loads(results_json.read_text())
        if not isinstance(rows, list):
            raise SystemExit(f"{results_json}: expected a JSON list, got {type(rows).__name__}")
    else:
        rows = []
    rows.append(row)
    results_json.write_text(json.dumps(rows, indent=2))


def run_cell(
    *,
    routing_table: Path,
    slice_: tuple[int, ...],
    out_schedule: Path,
    time_limit_s: int,
    n_workers: int,
    results_json: Path,
) -> int:
    topology = Topology(slice=slice_)
    table = load_routing_table(routing_table)
    if len(table) != topology.n_nodes:
        raise SystemExit(
            f"routing table {routing_table} has {len(table)} sources; "
            f"slice {slice_} expects {topology.n_nodes}"
        )
    lb = physical_edge_lb(table, topology.n_nodes)
    print(f"[stage 1] slice={slice_} N={topology.n_nodes} LB={lb}", flush=True)

    # Stage 2: orbit_greedy_full seed (via the public adapter).
    t0 = time.time()
    seed = schedule_from_orbit_greedy_full(topology, table, order="lpt_tail_asc")
    seed_makespan = schedule_makespan(seed)
    seed_violations = len(verify_capacity(seed))
    dt_seed = time.time() - t0
    print(f"[stage 2] orbit_greedy_full: makespan={seed_makespan} "
          f"viol={seed_violations} t={dt_seed:.2f}s", flush=True)
    if seed_violations:
        raise SystemExit(
            f"orbit_greedy_full produced {seed_violations} capacity violations on "
            f"{routing_table} — refusing to proceed."
        )

    # Stage 3+4: cold CP-SAT at t_upper=LB; on INFEASIBLE sweep upward.
    best = seed
    best_makespan = seed_makespan
    tried: list[dict] = []
    t_uppers = list(range(lb, seed_makespan))  # LB, LB+1, ..., M_seed-1
    if not t_uppers:
        # Seed already at LB; no room to search. Use seed.
        print(f"[stage 3] seed_makespan ({seed_makespan}) == LB ({lb}); "
              f"seed is already optimal.", flush=True)
        chosen_status = "SEED_AT_LB"
    else:
        chosen_status = None
        current_seed = seed
        for t_upper in t_uppers:
            print(f"\n--- t_upper={t_upper}, budget={time_limit_s}s, "
                  f"warm-start makespan={schedule_makespan(current_seed)} ---",
                  flush=True)
            t0 = time.time()
            try:
                sch = cpsat_literal(
                    topology, table,
                    t_upper=t_upper,
                    time_limit_s=time_limit_s,
                    n_workers=n_workers,
                    solver_msg=True,
                    warm_start_schedule=current_seed,
                )
                dt = time.time() - t0
                m = schedule_makespan(sch)
                v = len(verify_capacity(sch))
                tried.append({"t_upper": t_upper, "status": "FEASIBLE",
                              "makespan": m, "violations": v,
                              "runtime_s": round(dt, 2)})
                print(f"FEASIBLE: makespan={m} viol={v} t={dt:.2f}s", flush=True)
                if v != 0:
                    raise SystemExit(
                        f"cpsat_literal returned a schedule with {v} capacity "
                        f"violations at t_upper={t_upper} — bug in solver wrapper."
                    )
                if m < best_makespan:
                    best = sch
                    best_makespan = m
                    current_seed = sch  # chain warm-start at the next t_upper
                chosen_status = "FEASIBLE"
                break  # first feasible wins; further t_upper can only equal or exceed.
            except RuntimeError as e:
                dt = time.time() - t0
                msg = str(e)
                status = "INFEASIBLE" if "infeasible" in msg.lower() else "TIMEOUT_OR_ERROR"
                tried.append({"t_upper": t_upper, "status": status,
                              "makespan": None, "violations": None,
                              "runtime_s": round(dt, 2), "error": msg})
                print(f"{status}: {msg} (t={dt:.2f}s)", flush=True)
                if status != "INFEASIBLE":
                    # Hard error (e.g. ortools missing). Give up.
                    raise
                # else: INFEASIBLE at this t_upper -> continue sweep.
        if chosen_status is None:
            chosen_status = "ALL_INFEASIBLE_KEEPING_SEED"

    # Stage 5: save best.
    out_schedule.parent.mkdir(parents=True, exist_ok=True)
    save_schedule(best, out_schedule)
    print(f"\n[stage 5] saved best schedule to {out_schedule} "
          f"(makespan={best_makespan})", flush=True)

    # Append row to results.json.
    row = {
        "slice": list(slice_),
        "routing_table": str(routing_table),
        "lb": lb,
        "seed_makespan": seed_makespan,
        "cpsat_makespan": best_makespan,
        "cpsat_t_uppers_tried": tried,
        "chosen_status": chosen_status,
        "out_schedule": str(out_schedule),
    }
    results_json.parent.mkdir(parents=True, exist_ok=True)
    append_result_row(results_json, row)
    print(f"appended row to {results_json}", flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--routing-table", required=True, type=Path)
    p.add_argument("--slice", required=True, help="Comma-separated, e.g. 2,2,4")
    p.add_argument("--out-schedule", required=True, type=Path)
    p.add_argument("--time-limit-s", type=int, default=300)
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--results-json", required=True, type=Path)
    args = p.parse_args(argv)
    return run_cell(
        routing_table=args.routing_table,
        slice_=parse_slice(args.slice),
        out_schedule=args.out_schedule,
        time_limit_s=args.time_limit_s,
        n_workers=args.n_workers,
        results_json=args.results_json,
    )


if __name__ == "__main__":
    sys.exit(main())
