"""Visualization module for Person E: spatial congestion maps, corridor analytics, and benchmark plots."""
from __future__ import annotations

from src.visualization.spatial_mapping import (
    load_sensor_geo_metadata,
    assign_freeway_corridors,
    compute_corridor_statistics,
    plot_spatial_congestion_map,
    plot_corridor_error_distribution,
)
from src.visualization.benchmark_plots import (
    plot_horizon_comparison,
    plot_peak_vs_offpeak,
    plot_feature_importance_comparison,
    plot_rf_ablation_progression,
    plot_graph_gap_diagram,
)

__all__ = [
    "load_sensor_geo_metadata",
    "assign_freeway_corridors",
    "compute_corridor_statistics",
    "plot_spatial_congestion_map",
    "plot_corridor_error_distribution",
    "plot_horizon_comparison",
    "plot_peak_vs_offpeak",
    "plot_feature_importance_comparison",
    "plot_rf_ablation_progression",
    "plot_graph_gap_diagram",
]
