"""Phase 2a: CP-SAT at decreasing t_upper on loaded 8x4x4.

For each t_upper in {84, 83, 82, 81, 80, 78, 76}, run the CP-SAT solver
with a 1800s (30 min) wall-clock budget. Record:
  - status (FEASIBLE / INFEASIBLE / TIMEOUT)
  - actual makespan of the returned schedule (when FEASIBLE)
  - violation count (sanity, should be 0)
  - runtime

If a probe returns FEASIBLE with makespan < current best, save the schedule
to best_schedule.json."""
from __future__ import annotations
import json
import time
from pathlib import Path

from twisted_analysis.topology import Topology
from twisted_analysis.io.routing_table import load_routing_table
from twisted_analysis.io.schedule import save_schedule
from twisted_analysis.schedules.cpsat_literal import cpsat_literal
from twisted_analysis.schedules.verify import verify_capacity, schedule_makespan

ROUTING = "fixtures/routing/routing_table_8x4x4_twist.json"
SLICE = (8, 4, 4)
T_UPPER_SCHEDULE = [84, 83, 82, 81, 80, 78, 76]
TIME_LIMIT_S = 1800
WORKERS = 8

OUT = Path(__file__).parent / "03_cpsat_results.json"
BEST_FROM_PHASE = Path(__file__).parent / "03_best_cpsat_schedule.json"
GLOBAL_BEST = Path(__file__).parent / "best_schedule.json"


def run():
    topology = Topology(slice=SLICE)
    table = load_routing_table(ROUTING)
    rows = []
    best_makespan = None
    for t_upper in T_UPPER_SCHEDULE:
        print(f"\n--- t_upper={t_upper}, budget={TIME_LIMIT_S}s ---", flush=True)
        t0 = time.time()
        try:
            sch = cpsat_literal(
                topology, table,
                t_upper=t_upper, time_limit_s=TIME_LIMIT_S,
                n_workers=WORKERS, solver_msg=True,
            )
            dt = time.time() - t0
            v = verify_capacity(sch)
            m = schedule_makespan(sch)
            row = {"t_upper": t_upper, "status": "FEASIBLE",
                   "makespan": m, "violations": len(v),
                   "runtime_s": round(dt, 1)}
            print(f"FEASIBLE: makespan={m}, viol={len(v)}, t={dt:.1f}s", flush=True)
            if best_makespan is None or m < best_makespan:
                best_makespan = m
                save_schedule(sch, BEST_FROM_PHASE)
                print(f"  saved schedule to {BEST_FROM_PHASE}", flush=True)
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
        "best_makespan": best_makespan,
        "best_schedule_file": str(BEST_FROM_PHASE) if best_makespan is not None else None,
        "rows": rows,
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    run()
