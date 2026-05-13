from twisted_analysis.viz.load_histogram import plot_load_histogram

__all__ = ["plot_load_histogram", "plot_gantt", "plot_link_utilization_heatmap"]

# Import these only when needed to avoid circular dependencies
def __getattr__(name: str):
    if name == "plot_gantt":
        from twisted_analysis.viz.gantt import plot_gantt
        return plot_gantt
    elif name == "plot_link_utilization_heatmap":
        from twisted_analysis.viz.heatmap import plot_link_utilization_heatmap
        return plot_link_utilization_heatmap
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
