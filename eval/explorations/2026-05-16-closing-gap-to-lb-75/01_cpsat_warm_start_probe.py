"""Probe 1: warm-start CP-SAT from the makespan-80 incumbent.

For each t_upper in {79, 78, 77, 76}, run cpsat_literal with a 4-hour
(14400s) budget, warm-started from the previous best incumbent (initially
fixtures/schedule_8x4x4_loaded_cpsat_literal.json; updated to whatever
FEASIBLE incumbent the prior probe produced).

Saves per-t_upper rows to 01_cpsat_warm_start_results.json. The best
schedule (overall lowest makespan) is saved to 01_best_warm_start_schedule.json.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

from twisted_analysis.topology import Topology
from twisted_analysis.io.routing_table import load_routing_table
from twisted_analysis.io.schedule import save_schedule, load_schedule
from twisted_analysis.schedules.cpsat_literal import cpsat_literal
from twisted_analysis.schedules.verify import verify_capacity, schedule_makespan

ROUTING = "fixtures/routing_table_8x4x4_twist.json"
SLICE = (8, 4, 4)
T_UPPER_SCHEDULE = [79, 78, 77, 76]
TIME_LIMIT_S = 14400  # 4 hours
WORKERS = 8

SEED_SCHEDULE = "fixtures/schedule_8x4x4_loaded_cpsat_literal.json"
OUT = Path(__file__).parent / "01_cpsat_warm_start_results.json"
BEST = Path(__file__).parent / "01_best_warm_start_schedule.json"


def run():
    topology = Topology(slice=SLICE)
    table = load_routing_table(ROUTING)
    seed = load_schedule(SEED_SCHEDULE)
    seed_makespan = schedule_makespan(seed)
    print(f"Seed: {SEED_SCHEDULE}, makespan={seed_makespan}, entries={len(seed)}",
          flush=True)

    rows = []
    best_makespan = seed_makespan
    current_seed = seed
    for t_upper in T_UPPER_SCHEDULE:
        print(f"\n--- t_upper={t_upper}, budget={TIME_LIMIT_S}s, "
              f"warm-start from makespan={schedule_makespan(current_seed)} ---",
              flush=True)
        t0 = time.time()
        try:
            sch = cpsat_literal(
                topology, table,
                t_upper=t_upper, time_limit_s=TIME_LIMIT_S,
                n_workers=WORKERS, solver_msg=True,
                warm_start_schedule=current_seed,
            )
            dt = time.time() - t0
            v = verify_capacity(sch)
            m = schedule_makespan(sch)
            row = {"t_upper": t_upper, "status": "FEASIBLE",
                   "makespan": m, "violations": len(v),
                   "runtime_s": round(dt, 1)}
            print(f"FEASIBLE: makespan={m}, viol={len(v)}, t={dt:.1f}s",
                  flush=True)
            if m < best_makespan:
                best_makespan = m
                save_schedule(sch, BEST)
                print(f"  saved new best to {BEST}", flush=True)
                current_seed = sch  # chain warm-start
        except RuntimeError as e:
            dt = time.time() - t0
            msg = str(e)
            status = "INFEASIBLE" if "infeasible" in msg.lower() else "TIMEOUT"
            row = {"t_upper": t_upper, "status": status,
                   "makespan": None, "violations": None,
                   "runtime_s": round(dt, 1), "error": msg}
            print(f"{status}: {msg} (t={dt:.1f}s)", flush=True)
            if status == "INFEASIBLE":
                rows.append(row)
                break
        rows.append(row)

    result = {
        "schedule": T_UPPER_SCHEDULE,
        "seed_schedule_file": SEED_SCHEDULE,
        "seed_makespan": seed_makespan,
        "best_makespan": best_makespan,
        "best_schedule_file": str(BEST) if best_makespan < seed_makespan else None,
        "rows": rows,
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    run()
