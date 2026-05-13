from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt

from twisted_analysis.model.flow import AllToAll


def plot_load_histogram(workload: AllToAll, out_path: Path) -> None:
    loads = list(workload.link_load.values())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(loads, bins=range(min(loads), max(loads) + 2))
    ax.set_xlabel("Link load (flow-units)")
    ax.set_ylabel("# directed links")
    ax.set_title(f"Link-load distribution: slice={workload.topology.slice}, "
                  f"m={workload.msg_size}, LB={workload.lower_bound}")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
