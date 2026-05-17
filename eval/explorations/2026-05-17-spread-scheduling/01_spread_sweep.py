"""Probe 1: K-sweep over spread_greedy(k) on loaded 8x4x4.

Runs K in {1, 2, 3, 4} with order='lpt'. For each K, computes:
  - makespan (max round + L over flows)
  - capacity violations (must be 0)
  - max outgoing DMAs per device per round (must be <= K)
  - max incoming DMAs per device per round (must be <= K)
  - average DMAs per device per active round (pipeline density signal)
  - number of distinct rounds containing at least one DMA (spread signal)

Saves four schedule JSONs (schedule_k1.json ... schedule_k4.json) and a
single 01_spread_sweep_results.json comparison table.
"""
from __future__ import annotations
import json
import time
from collections import Counter
from pathlib import Path

from twisted_analysis.topology import Topology
from twisted_analysis.io.routing_table import load_routing_table
from twisted_analysis.io.schedule import save_schedule
from twisted_analysis.schedules.spread_greedy import spread_greedy
from twisted_analysis.schedules.verify import verify_capacity, schedule_makespan

ROUTING = "fixtures/routing_table_8x4x4_twist.json"
SLICE = (8, 4, 4)
K_VALUES = [1, 2, 3, 4]
ORDER = "lpt"

OUT = Path(__file__).parent / "01_spread_sweep_results.json"


def _metrics(schedule, n):
    out_counts: Counter = Counter()
    in_counts: Counter = Counter()
    rounds_with_dma: set[int] = set()
    for e in schedule:
        out_counts[(e["src"], e["round"])] += 1
        in_counts[(e["dst"], e["round"])] += 1
        rounds_with_dma.add(e["round"])
    max_out = max(out_counts.values())
    max_in = max(in_counts.values())
    # Average DMAs per (device, active-round) pair across both sides.
    total_out_dmas = sum(out_counts.values())
    avg_out_per_device_round = total_out_dmas / len(out_counts)
    return {
        "max_out_per_device_round": max_out,
        "max_in_per_device_round": max_in,
        "avg_out_per_device_round": round(avg_out_per_device_round, 3),
        "n_rounds_with_dma": len(rounds_with_dma),
    }


def run():
    topology = Topology(slice=SLICE)
    table = load_routing_table(ROUTING)
    n = topology.n_nodes
    print(f"Loaded {ROUTING}, n={n}", flush=True)

    rows = []
    for k in K_VALUES:
        t0 = time.time()
        sch = spread_greedy(topology, table, k=k, order=ORDER)
        dt = time.time() - t0
        viol = verify_capacity(sch)
        mks = schedule_makespan(sch)
        m = _metrics(sch, n)
        row = {
            "k": k,
            "makespan": mks,
            "violations": len(viol),
            "runtime_s": round(dt, 2),
            **m,
        }
        rows.append(row)
        print(f"  k={k}: makespan={mks} viol={len(viol)} "
              f"max_out={m['max_out_per_device_round']} "
              f"max_in={m['max_in_per_device_round']} "
              f"avg={m['avg_out_per_device_round']} "
              f"rounds_used={m['n_rounds_with_dma']} "
              f"t={dt:.2f}s",
              flush=True)

        out_sched = Path(__file__).parent / f"schedule_k{k}.json"
        save_schedule(sch, out_sched)
        print(f"    saved {out_sched}", flush=True)

    result = {
        "routing": ROUTING, "slice": list(SLICE), "n": n,
        "order": ORDER, "k_values": K_VALUES,
        "rows": rows,
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    run()
