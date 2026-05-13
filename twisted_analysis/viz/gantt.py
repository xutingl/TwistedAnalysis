"""Minimal Gantt plotter: one row per link, time on x-axis, colored bars per flow."""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt

from twisted_analysis.simulator.engine import Simulator


def plot_gantt(sim: Simulator, out_path: Path, max_links: int = 32) -> None:
    if not sim.record_history:
        raise ValueError("Simulator needs record_history=True")
    links = list(sim.link_busy.keys())[:max_links]
    fig, ax = plt.subplots(figsize=(10, max(2, len(links) * 0.25)))
    for i, e in enumerate(links):
        busy = sim.link_busy[e]
        for t, b in enumerate(busy):
            if b:
                ax.barh(i, 1, left=t, height=0.8, color="C0")
    ax.set_yticks(range(len(links)))
    ax.set_yticklabels([str(e) for e in links], fontsize=6)
    ax.set_xlabel("Step")
    ax.set_title("Link usage Gantt")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
