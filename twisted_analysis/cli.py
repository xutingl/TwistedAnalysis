from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import yaml

from twisted_analysis.topology import Topology, Router
from twisted_analysis.model import AllToAll
from twisted_analysis.schedules.round_robin import RoundRobinSchedule
from twisted_analysis.schedules.dim_phased import DimPhasedSchedule
from twisted_analysis.schedules.lp_optimal import lp_assignment_to_injections
from twisted_analysis.simulator import Simulator
from twisted_analysis.simulator.instrumentation import collect_idle_trace, write_gantt_csv


SCHEDULES = {
    "round_robin": RoundRobinSchedule(),
    "dim_phased": DimPhasedSchedule(),
}


def run_experiment(cfg: dict) -> dict:
    slice_ = tuple(cfg["slice"])
    msg_size = cfg.get("msg_size", 1)
    sched_name = cfg["schedule"]
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    t = Topology(slice=slice_)
    r = Router(t)
    w = AllToAll(t, r, msg_size=msg_size)

    if sched_name == "ilp_optimal":
        from twisted_analysis.lp.ilp import solve_makespan
        m_opt, assignment = solve_makespan(
            t, r, list(w.flows), T_upper=w.lower_bound * 4
        )
        injs = lp_assignment_to_injections(list(w.flows), r, assignment)
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

    idle = collect_idle_trace(sim, w.bottleneck_edges())
    write_gantt_csv(sim, out_dir / "gantt.csv")

    summary = {
        "name": cfg["name"],
        "slice": list(slice_),
        "msg_size": msg_size,
        "schedule": sched_name,
        "lower_bound": w.lower_bound,
        "makespan": makespan,
        "ratio": makespan / w.lower_bound if w.lower_bound else 0.0,
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
