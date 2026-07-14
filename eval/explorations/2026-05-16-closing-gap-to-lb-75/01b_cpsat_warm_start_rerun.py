"""Probe 1 re-run: recover the makespan-76 incumbent lost to the
UNKNOWN-status bug in the first run of 01_cpsat_warm_start_probe.py.

The first run reached FEASIBLE makespan 78 at t_upper=79 and t_upper=78,
then lost CP-SAT incumbents at t_upper=77 (objective=77) and t_upper=76
(objective=76) because the wrapper treated status=UNKNOWN as "no
incumbent". The fix in commit 2860800 extracts the incumbent on UNKNOWN
status when one exists. This re-run targets the lost ground:

  - t_upper=76: warm-start from the makespan-78 schedule produced in
    the first run. Expected to recover at least makespan 76.
  - t_upper=75: warm-start from the t_upper=76 result (LB-tight if
    found; otherwise the incumbent at t_upper=75 if any).

We do not re-run t_upper=79 or t_upper=78 — both produced FEASIBLE
makespan 78 in the first run (commit dc4f778 results JSON).
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

ROUTING = "fixtures/routing/routing_table_8x4x4_twist.json"
SLICE = (8, 4, 4)
T_UPPER_SCHEDULE = [76, 75]
TIME_LIMIT_S = 14400  # 4 hours
WORKERS = 8

SEED_SCHEDULE = "eval/explorations/2026-05-16-closing-gap-to-lb-75/01_best_warm_start_schedule.json"
OUT = Path(__file__).parent / "01b_cpsat_warm_start_rerun_results.json"
BEST = Path(__file__).parent / "01b_best_warm_start_schedule.json"


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
                current_seed = sch
        except RuntimeError as e:
            dt = time.time() - t0
            msg = str(e)
            if "infeasible" in msg.lower():
                status = "INFEASIBLE"
            elif "no incumbent" in msg.lower():
                status = "TIMEOUT_NO_INCUMBENT"
            else:
                status = "ERROR"
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
