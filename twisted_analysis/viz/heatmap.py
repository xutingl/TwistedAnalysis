from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from twisted_analysis.simulator.engine import Simulator


def plot_link_utilization_heatmap(sim: Simulator, out_path: Path) -> None:
    if not sim.record_history:
        raise ValueError("Simulator needs record_history=True")
    links = list(sim.link_busy.keys())
    T = len(sim.busy_per_step)
    mat = np.zeros((len(links), T), dtype=int)
    for i, e in enumerate(links):
        mat[i, :] = sim.link_busy[e][:T]
    fig, ax = plt.subplots(figsize=(10, max(3, len(links) * 0.1)))
    ax.imshow(mat, aspect="auto", cmap="Greys", interpolation="nearest")
    ax.set_xlabel("Step")
    ax.set_ylabel("Directed link")
    ax.set_title("Per-step link utilization")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
