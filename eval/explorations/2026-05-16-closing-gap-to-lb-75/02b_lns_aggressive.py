"""Probe 2b: LNS with aggressive destroy + long subproblem budget.

The default LNS probe (02_lns_probe.py, destroy_size_frac=0.05,
per_subproblem_budget_s=300) ran 100 iterations in 6 minutes — every
subproblem returned INFEASIBLE in ~3s. Hypothesis: with only 5% of flows
destroyed, the pinned 95% saturate the makespan-77 budget and CP-SAT
immediately proves infeasibility. This probe tests whether larger
destroys + longer subproblem budgets escape that local optimum.

Parameters:
  destroy_size_frac = 0.30  (4877 flows destroyed of 16256)
  per_subproblem_budget_s = 600  (10 min)
  n_iters = 24  (=> ~4 h wall-clock)
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

ROUTING = "fixtures/routing_table_8x4x4_twist.json"
SLICE = (8, 4, 4)
SEED_SCHEDULE = "eval/explorations/2026-05-16-closing-gap-to-lb-75/01_best_warm_start_schedule.json"

N_ITERS = 24
PER_SUBPROBLEM_S = 600
DESTROY_FRAC = 0.30
RNG_SEED = 20260516
WORKERS = 8
STRATEGIES = ("time_window", "random_subset", "makespan_flows")

OUT = Path(__file__).parent / "02b_lns_aggressive_results.json"
BEST = Path(__file__).parent / "02b_best_lns_aggressive_schedule.json"


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
        result = info.get("result", "?")
        nm = info.get("new_makespan")
        acc = info.get("accepted")
        suffix = ""
        if result == "feasible":
            suffix = f" -> new_M={nm} accepted={acc}"
        elif "error_msg" in info:
            em = info["error_msg"]
            if len(em) > 60:
                em = em[:60] + "..."
            suffix = f" ({em})"
        print(f"  iter {it:3d}: strat={info['strategy']:<16s} "
              f"|D|={info['destroy_size']:>5d} "
              f"cur_M={info['current_makespan']:>3d} "
              f"target={info['target_t_upper']:>3d} "
              f"result={result:<22s}"
              f" t={t:>7.1f}s{suffix}",
              flush=True)

    print(f"--- LNS aggressive: n_iters={N_ITERS}, per_subproblem={PER_SUBPROBLEM_S}s, "
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

    # Summarize outcomes
    from collections import Counter
    outcome_counts = Counter(r["result"] for r in iter_log)
    print(f"\noutcome breakdown: {dict(outcome_counts)}", flush=True)

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
        "outcome_counts": dict(outcome_counts),
        "iter_log": iter_log,
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    run()
