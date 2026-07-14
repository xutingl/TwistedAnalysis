"""Phase 1: measure coalescing factor on existing shipped schedules.

For each shipped fixture, compute uncoalesced and coalesced per-(edge, round)
descriptor counts and the coalescing factor (uncoalesced / coalesced). Also
report avg path length and active-edge-round occupancy.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from twisted_analysis.io.schedule import load_schedule
from descriptor_counter import count_dma_descriptors


FIXTURES = REPO / "fixtures" / "nonragged"
OUT_PATH = Path(__file__).resolve().parent / "01_results.json"

SCHEDULES = [
    ("cpsat_literal_warm", "schedule_8x4x4_loaded_cpsat_literal_warm.json"),
    ("spread_greedy_k2",   "schedule_8x4x4_loaded_spread_greedy_k2.json"),
    ("literal_greedy_lpt", "schedule_8x4x4_loaded_literal_greedy_lpt.json"),
    ("orbit_greedy_full",  "schedule_8x4x4_loaded_orbit_greedy_full_lpt_tail_asc.json"),
]


def summarize(name: str, entries: list[dict]) -> dict:
    uncoalesced, coalesced = count_dma_descriptors(entries)
    n_flows = len(entries)
    path_lengths = [len(e["path"]) - 1 for e in entries]
    avg_hops = sum(path_lengths) / n_flows
    makespan = max(e["round"] + (len(e["path"]) - 1) for e in entries)
    factor = uncoalesced / coalesced if coalesced else float("inf")
    return {
        "name": name,
        "n_flows": n_flows,
        "makespan": makespan,
        "avg_hop_length": round(avg_hops, 3),
        "uncoalesced_descriptors": uncoalesced,
        "coalesced_descriptors": coalesced,
        "coalescing_factor": round(factor, 3),
        "break_even_factor_for_kernel_switch": round(avg_hops, 3),
        "headroom_above_break_even": round(factor - avg_hops, 3),
    }


def main() -> None:
    results = []
    for name, fname in SCHEDULES:
        path = FIXTURES / fname
        if not path.exists():
            print(f"SKIP {name}: {path} not found")
            continue
        entries = load_schedule(path)
        summary = summarize(name, entries)
        results.append(summary)
        print(
            f"{name:24s} | makespan={summary['makespan']:3d} | "
            f"uncoalesced={summary['uncoalesced_descriptors']:6d} | "
            f"coalesced={summary['coalesced_descriptors']:6d} | "
            f"factor={summary['coalescing_factor']:.3f} | "
            f"break_even={summary['break_even_factor_for_kernel_switch']:.3f} | "
            f"headroom={summary['headroom_above_break_even']:+.3f}"
        )
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
