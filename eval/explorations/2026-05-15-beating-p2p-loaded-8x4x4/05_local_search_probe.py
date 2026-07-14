"""Phase 3: run local_search_repair starting from each schedule produced in
earlier phases and from each baseline. Record per-seed improvement."""
from __future__ import annotations
import json
import time
from pathlib import Path

from twisted_analysis.topology import Topology
from twisted_analysis.io.routing_table import load_routing_table
from twisted_analysis.io.schedule import load_schedule, save_schedule, schedule_from_orbit_greedy_full
from twisted_analysis.schedules.local_search import local_search_repair
from twisted_analysis.schedules.verify import schedule_makespan, verify_capacity

ROUTING = "fixtures/routing/routing_table_8x4x4_twist.json"
SLICE = (8, 4, 4)
FOLDER = Path(__file__).parent
OUT = FOLDER / "05_local_search_results.json"

# Seed paths from earlier phases.
SEED_PATHS = [
    Path("fixtures/nonragged/schedule_8x4x4_loaded_orbit_greedy_full_lpt_tail_asc.json"),
    Path("fixtures/nonragged/schedule_8x4x4_loaded_literal_greedy_lpt.json"),
    FOLDER / "02_best_random_shuffle_schedule.json",
    FOLDER / "03_best_cpsat_schedule.json",
    FOLDER / "04_best_lp_rounding_schedule.json",
    FOLDER / "_seed_orbit_greedy_full_lpt.json",  # generated below if missing
]


def _maybe_generate_lpt_seed(topology, table):
    path = FOLDER / "_seed_orbit_greedy_full_lpt.json"
    if path.exists():
        return path
    print("  generating orbit_greedy_full[lpt] seed (Phase 1 best at makespan 84)...",
          flush=True)
    sch = schedule_from_orbit_greedy_full(topology, table, order="lpt")
    save_schedule(sch, path)
    print(f"  -> saved {path} (makespan={schedule_makespan(sch)})", flush=True)
    return path


def run():
    topology = Topology(slice=SLICE)
    table = load_routing_table(ROUTING)
    _maybe_generate_lpt_seed(topology, table)
    rows = []
    best_makespan = None
    best_seed_path = None
    for seed_path in SEED_PATHS:
        if not seed_path.exists():
            print(f"  SKIP missing: {seed_path}", flush=True)
            rows.append({"seed": str(seed_path), "status": "MISSING"})
            continue
        seed = load_schedule(seed_path)
        m_in = schedule_makespan(seed)
        t0 = time.time()
        out = local_search_repair(topology, table, seed, max_iters=2000)
        dt = time.time() - t0
        m_out = schedule_makespan(out)
        v = verify_capacity(out)
        row = {
            "seed": str(seed_path),
            "makespan_in": m_in,
            "makespan_out": m_out,
            "delta": m_in - m_out,
            "violations": len(v),
            "runtime_s": round(dt, 1),
        }
        rows.append(row)
        out_path = FOLDER / f"05_localsearch_from_{seed_path.stem}.json"
        save_schedule(out, out_path)
        print(f"  {seed_path.name}: {m_in} -> {m_out} (Δ={m_in - m_out}), "
              f"viol={len(v)}, t={dt:.1f}s", flush=True)
        if best_makespan is None or m_out < best_makespan:
            best_makespan = m_out
            best_seed_path = str(out_path)
    OUT.write_text(json.dumps({"rows": rows,
                               "best_makespan": best_makespan,
                               "best_schedule_file": best_seed_path},
                              indent=2))
    print(f"\nbest after local-search: makespan={best_makespan}", flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    run()
