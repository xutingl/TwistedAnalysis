"""Report step-model statistics for a schedule JSON.

Prints the barrier-step count, the achieved whole-path edge cap, and the
per-device DMA depth per step. The edge cap is what `orbit_pack_shuffled`
schedules need for the kernel generator's `--step-edge-cap`, since they
do not respect the `c` their reference packing was built against.

Usage:
    python scripts/schedule_stats.py fixtures/nonragged/schedule_....json
    python scripts/schedule_stats.py --field edge_cap sched.json   # bare int
"""
from __future__ import annotations
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Make `python scripts/schedule_stats.py` work without install.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from twisted_analysis.io.schedule import load_schedule
from twisted_analysis.schedules.verify import (
    max_step_edge_load,
    schedule_step_count,
)


def _device_depth(schedule: list[dict]) -> tuple[int, int]:
    """(max sends, max recvs) by any one device within any one step."""
    sends: dict[int, Counter] = defaultdict(Counter)
    recvs: dict[int, Counter] = defaultdict(Counter)
    for e in schedule:
        sends[e["round"]][e["src"]] += 1
        recvs[e["round"]][e["dst"]] += 1
    max_send = max((max(c.values()) for c in sends.values()), default=0)
    max_recv = max((max(c.values()) for c in recvs.values()), default=0)
    return max_send, max_recv


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("schedule", type=Path, help="Path to a schedule JSON")
    p.add_argument(
        "--field",
        choices=["steps", "edge_cap", "max_send", "max_recv"],
        default=None,
        help="Print just this value as a bare integer (for shell capture). "
             "Default: print a human-readable summary.",
    )
    args = p.parse_args(argv)

    schedule = load_schedule(args.schedule)
    steps = schedule_step_count(schedule)
    edge_cap = max_step_edge_load(schedule)
    max_send, max_recv = _device_depth(schedule)

    if args.field is not None:
        print({
            "steps": steps,
            "edge_cap": edge_cap,
            "max_send": max_send,
            "max_recv": max_recv,
        }[args.field])
        return 0

    print(f"{args.schedule}")
    print(f"  entries                    {len(schedule):,}")
    print(f"  barrier steps              {steps}")
    print(f"  max whole-path edge load   {edge_cap}   (per step)")
    print(f"  max DMAs/device/step       send {max_send}, recv {max_recv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
