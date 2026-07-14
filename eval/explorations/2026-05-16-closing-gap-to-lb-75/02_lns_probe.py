"""Probe 2: LNS with CP-SAT subsolver, seeded from the makespan-80 incumbent.

Runs lns_cpsat_repair for 100 iterations with a 300s (5 min) per-subproblem
budget. Strategies rotate time_window / random_subset / makespan_flows.
Logs every iteration (strategy, destroy size, current makespan, whether
accepted) to 02_lns_results.json. The final schedule (which is the best
incumbent across all iterations) is saved to 02_best_lns_schedule.json
if its makespan is strictly below the seed's.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

from twisted_analysis.topology import Topology
from twisted_analysis.io.routing_table import load_routing_table
from twisted_analysis.io.schedule import save_schedule, load_schedule
from twisted_analysis.schedules.lns_cpsat import lns_cpsat_repair
from twisted_analysis.schedules.verify import verify_capacity, schedule_makespan

ROUTING = "fixtures/routing/routing_table_8x4x4_twist.json"
SLICE = (8, 4, 4)
SEED_SCHEDULE = "eval/explorations/2026-05-16-closing-gap-to-lb-75/01_best_warm_start_schedule.json"

N_ITERS = 100
PER_SUBPROBLEM_S = 300
DESTROY_FRAC = 0.05
RNG_SEED = 20260516
WORKERS = 8
STRATEGIES = ("time_window", "random_subset", "makespan_flows")

OUT = Path(__file__).parent / "02_lns_results.json"
BEST = Path(__file__).parent / "02_best_lns_schedule.json"


def run():
    topology = Topology(slice=SLICE)
    table = load_routing_table(ROUTING)
    seed = load_schedule(SEED_SCHEDULE)
    seed_makespan = schedule_makespan(seed)
    print(f"Seed: {SEED_SCHEDULE}, makespan={seed_makespan}", flush=True)

    iter_log: list[dict] = []
    t_start = time.time()

    def log(it: int, info: dict):
        t = round(time.time() - t_start, 1)
        info["iter"] = it
        info["elapsed_s"] = t
        iter_log.append(dict(info))
        print(f"  iter {it:3d}: strat={info['strategy']:<16s} "
              f"|D|={info['destroy_size']:>5d} "
              f"cur_M={info['current_makespan']:>3d} "
              f"target={info['target_t_upper']:>3d} "
              f"t={t:>7.1f}s",
              flush=True)

    print(f"--- LNS: n_iters={N_ITERS}, per_subproblem={PER_SUBPROBLEM_S}s, "
          f"destroy_frac={DESTROY_FRAC}, strategies={STRATEGIES} ---",
          flush=True)
    sch = lns_cpsat_repair(
        topology, table, seed,
        n_iters=N_ITERS,
        per_subproblem_budget_s=PER_SUBPROBLEM_S,
        destroy_strategies=STRATEGIES,
        destroy_size_frac=DESTROY_FRAC,
        rng_seed=RNG_SEED,
        n_workers=WORKERS,
        log_fn=log,
    )
    elapsed = round(time.time() - t_start, 1)
    v = verify_capacity(sch)
    m = schedule_makespan(sch)
    print(f"\nFinal: makespan={m}, viol={len(v)}, total={elapsed}s", flush=True)

    if m < seed_makespan:
        save_schedule(sch, BEST)
        print(f"  saved new best to {BEST}", flush=True)

    result = {
        "seed_schedule_file": SEED_SCHEDULE,
        "seed_makespan": seed_makespan,
        "final_makespan": m,
        "violations": len(v),
        "total_runtime_s": elapsed,
        "best_schedule_file": str(BEST) if m < seed_makespan else None,
        "params": {
            "n_iters": N_ITERS, "per_subproblem_budget_s": PER_SUBPROBLEM_S,
            "destroy_size_frac": DESTROY_FRAC, "rng_seed": RNG_SEED,
            "strategies": list(STRATEGIES), "n_workers": WORKERS,
        },
        "iter_log": iter_log,
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    run()
