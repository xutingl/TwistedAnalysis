from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import yaml

from twisted_analysis.topology import Topology, DORRouter, ILPRouter
from twisted_analysis.model import AllToAll
from twisted_analysis.schedules.round_robin import RoundRobinSchedule
from twisted_analysis.schedules.dim_phased import DimPhasedSchedule
from twisted_analysis.schedules.xla import XLASchedule
from twisted_analysis.schedules.orbit_greedy import (
    OrbitGreedySchedule, PipelinedOrbitSchedule,
)
from twisted_analysis.schedules.lp_optimal import lp_assignment_to_injections
from twisted_analysis.simulator import Simulator
from twisted_analysis.simulator.instrumentation import collect_idle_trace, write_gantt_csv


SCHEDULES = {
    "round_robin": RoundRobinSchedule(),
    "dim_phased": DimPhasedSchedule(),
    "xla": XLASchedule(),
    "orbit_greedy": OrbitGreedySchedule(order="lpt_tail_asc"),
    "orbit_greedy_lpt": OrbitGreedySchedule(order="lpt"),
    "orbit_greedy_spt": OrbitGreedySchedule(order="spt"),
    "pipelined_orbit": PipelinedOrbitSchedule(order="lpt_tail_asc"),
    "pipelined_orbit_lpt": PipelinedOrbitSchedule(order="lpt"),
    "pipelined_orbit_spt": PipelinedOrbitSchedule(order="spt"),
}


def run_experiment(cfg: dict) -> dict:
    slice_ = tuple(cfg["slice"])
    msg_size = cfg.get("msg_size", 1)
    sched_name = cfg["schedule"]
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    t = Topology(slice=slice_)
    router_name = cfg.get("router", "ilp")  # default: ILP
    if router_name == "ilp":
        r = ILPRouter(t)
    elif router_name == "dor":
        r = DORRouter(t)
    else:
        raise ValueError(f"unknown router: {router_name}")
    w = AllToAll(t, r, msg_size=msg_size)

    # Allow YAML to override the binary-search upper bound. Default is 4*LB.
    # Set to `lb` when you have an independent witness that LB is achievable
    # (e.g. OrbitGreedy) — skips the expensive feasibility checks above LB.
    t_upper_mult = cfg.get("ilp_T_upper_multiplier", 4)
    t_upper = int(w.lower_bound * t_upper_mult)

    if sched_name == "ilp_optimal":
        from twisted_analysis.lp.ilp import solve_makespan
        m_opt, assignment = solve_makespan(
            t, r, list(w.flows), T_upper=t_upper,
        )
        injs = lp_assignment_to_injections(list(w.flows), r, assignment)
    elif sched_name == "ilp_optimal_symmetric":
        from twisted_analysis.lp.symmetric import solve_symmetric_makespan
        from twisted_analysis.schedules.lp_symmetric import symmetric_assignment_to_injections
        m_opt, assignment = solve_symmetric_makespan(
            t, r, list(w.flows), T_upper=t_upper,
            time_limit_seconds=cfg.get("ilp_time_limit_seconds"),
        )
        injs = symmetric_assignment_to_injections(t, r, list(w.flows), assignment)
    else:
        sched = SCHEDULES[sched_name]
        injs = sched.emit(w)

    # Build the simulator from the actual flows the schedule will inject — this
    # supports partial-coverage schedules (e.g. DimPhased) without hanging.
    sim_flows = list({inj.flow for inj in injs})
    sim = Simulator(t, r, sim_flows, record_history=True)
    for inj in injs:
        sim.inject(inj)
    makespan = sim.run()

    # LB for THIS schedule's actual workload (= full-AllToAll LB when the
    # schedule covers every (src,dst); subset LB when partial-coverage like
    # DimPhased). This makes `ratio` ≥ 1 for any honest schedule and lets us
    # also report the full-AllToAll LB separately as `full_alltoall_lb` for
    # the partial-coverage caveat.
    from collections import Counter
    subset_load: Counter = Counter()
    for f in sim_flows:
        for e in r.path(f.src, f.dst):
            subset_load[e] += f.size
    schedule_lb = max(subset_load.values()) if subset_load else 0

    idle = collect_idle_trace(sim, w.bottleneck_edges())
    write_gantt_csv(sim, out_dir / "gantt.csv")

    n_flows_covered = len(sim_flows)
    n_flows_full = len(w.flows)
    summary = {
        "name": cfg["name"],
        "slice": list(slice_),
        "msg_size": msg_size,
        "schedule": sched_name,
        "router": router_name,
        "lower_bound": schedule_lb,
        "full_alltoall_lb": w.lower_bound,
        "n_flows_covered": n_flows_covered,
        "n_flows_full": n_flows_full,
        "coverage": n_flows_covered / n_flows_full if n_flows_full else 0.0,
        "makespan": makespan,
        "ratio": makespan / schedule_lb if schedule_lb else 0.0,
        "bottleneck_edges": [list(map(list, [e[0], e[1]])) + [e[2], e[3]]
                              for e in w.bottleneck_edges()],
        "idle_steps_on_bottleneck": {
            str(e): v for e, v in idle.items()
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="twisted_analysis")
    sub = p.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run")
    p_run.add_argument("config", type=Path)
    args = p.parse_args(argv)
    if args.cmd == "run":
        cfg = yaml.safe_load(args.config.read_text())
        summary = run_experiment(cfg)
        print(json.dumps(summary, indent=2))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
