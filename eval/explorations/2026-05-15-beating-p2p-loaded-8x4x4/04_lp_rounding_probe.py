"""Phase 2b: LP-relaxation + randomized rounding on loaded 8x4x4."""
from __future__ import annotations
import json
import time
from pathlib import Path

from twisted_analysis.topology import Topology
from twisted_analysis.io.routing_table import load_routing_table
from twisted_analysis.io.schedule import save_schedule
from twisted_analysis.schedules.lp_rounding import lp_rounding
from twisted_analysis.schedules.verify import verify_capacity, schedule_makespan

ROUTING = "fixtures/routing_table_8x4x4_twist.json"
SLICE = (8, 4, 4)
T_UPPER = 95
N_TRIALS = 200
SEED = 0
OUT = Path(__file__).parent / "04_lp_rounding_results.json"
BEST_PHASE = Path(__file__).parent / "04_best_lp_rounding_schedule.json"


def run():
    topology = Topology(slice=SLICE)
    table = load_routing_table(ROUTING)
    t0 = time.time()
    sch = lp_rounding(topology, table,
                      t_upper=T_UPPER, n_trials=N_TRIALS, seed=SEED)
    dt = time.time() - t0
    m = schedule_makespan(sch)
    v = verify_capacity(sch)
    save_schedule(sch, BEST_PHASE)
    result = {
        "t_upper": T_UPPER,
        "n_trials": N_TRIALS,
        "best_makespan": m,
        "violations": len(v),
        "runtime_s": round(dt, 1),
        "schedule_file": str(BEST_PHASE),
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(f"LP-rounding best of {N_TRIALS} trials: makespan={m}, "
          f"viol={len(v)}, t={dt:.1f}s")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    run()
