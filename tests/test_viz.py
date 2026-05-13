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


def test_plot_gantt_writes_png(tmp_path: Path):
    t = Topology(slice=(2, 4))
    r = Router(t)
    w = AllToAll(t, r, msg_size=1)
    from twisted_analysis.schedules.round_robin import RoundRobinSchedule
    from twisted_analysis.simulator import Simulator
    from twisted_analysis.viz.gantt import plot_gantt
    sim = Simulator(t, r, list(w.flows), record_history=True)
    sched = RoundRobinSchedule()
    for inj in sched.emit(w):
        sim.inject(inj)
    sim.run()
    out = tmp_path / "gantt.png"
    plot_gantt(sim, out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_link_utilization_heatmap_writes_png(tmp_path: Path):
    t = Topology(slice=(2, 4))
    r = Router(t)
    w = AllToAll(t, r, msg_size=1)
    from twisted_analysis.schedules.round_robin import RoundRobinSchedule
    from twisted_analysis.simulator import Simulator
    from twisted_analysis.viz.heatmap import plot_link_utilization_heatmap
    sim = Simulator(t, r, list(w.flows), record_history=True)
    sched = RoundRobinSchedule()
    for inj in sched.emit(w):
        sim.inject(inj)
    sim.run()
    out = tmp_path / "heatmap.png"
    plot_link_utilization_heatmap(sim, out)
    assert out.exists() and out.stat().st_size > 0
