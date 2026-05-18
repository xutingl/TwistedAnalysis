"""Phase 2: run cpsat_coalesce on the real loaded 8x4x4 problem with t_upper=78.

Outputs `02_results.json` with the best-found coalescing factor.
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from twisted_analysis.io.routing_table import load_routing_table
from twisted_analysis.topology import Topology
from cpsat_coalesce import cpsat_coalesce
from descriptor_counter import count_dma_descriptors


ROUTING_PATH = REPO / "fixtures" / "routing_table_8x4x4_twist.json"
OUT_PATH = Path(__file__).resolve().parent / "02_results.json"
SCHEDULE_OUT = Path(__file__).resolve().parent / "02_coalesce_schedule.json"


def main(time_limit_s: int = 3600, n_workers: int = 8) -> None:
    topology = Topology(slice=(8, 4, 4))
    table = load_routing_table(ROUTING_PATH)

    t0 = time.time()
    entries, reported = cpsat_coalesce(
        topology, table,
        t_upper=78,
        time_limit_s=time_limit_s,
        n_workers=n_workers,
        solver_msg=True,
    )
    runtime = time.time() - t0

    if entries is None:
        result = {
            "status": "no_incumbent",
            "t_upper": 78,
            "time_limit_s": time_limit_s,
            "runtime_s": round(runtime, 1),
        }
        OUT_PATH.write_text(json.dumps(result, indent=2))
        print("No incumbent found within time budget.")
        return

    uncoalesced, coalesced_post_hoc = count_dma_descriptors(entries)
    assert coalesced_post_hoc == reported, (
        f"sanity check failed: post-hoc={coalesced_post_hoc} vs reported={reported}"
    )
    factor = uncoalesced / coalesced_post_hoc
    makespan = max(e["round"] + (len(e["path"]) - 1) for e in entries)

    SCHEDULE_OUT.write_text(json.dumps(entries, indent=2))

    result = {
        "status": "ok",
        "t_upper": 78,
        "achieved_makespan": makespan,
        "time_limit_s": time_limit_s,
        "runtime_s": round(runtime, 1),
        "uncoalesced_descriptors": uncoalesced,
        "coalesced_descriptors": coalesced_post_hoc,
        "coalescing_factor_upper_bound": round(factor, 3),
        "break_even_factor": round(uncoalesced / 16256, 3),
        "headroom_above_break_even": round(factor - (uncoalesced / 16256), 3),
    }
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
