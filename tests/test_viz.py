import matplotlib
matplotlib.use("Agg")

from pathlib import Path

from twisted_analysis.topology import Topology, Router
from twisted_analysis.model import AllToAll
from twisted_analysis.viz.load_histogram import plot_load_histogram


def test_load_histogram_writes_png(tmp_path: Path):
    t = Topology(slice=(2, 4))
    r = Router(t)
    w = AllToAll(t, r, msg_size=1)
    out = tmp_path / "hist.png"
    plot_load_histogram(w, out)
    assert out.exists() and out.stat().st_size > 0
