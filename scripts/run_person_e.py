#!/usr/bin/env python3
"""Person E: Benchmark Synthesis, Spatial Congestion Mapping & Evaluation.

    python scripts/run_person_e.py

Ingests results from Person C and Person D, compiles cross-model comparative benchmarks,
generates spatial highway congestion maps, and produces Phase 1 evaluation artifacts.

Outputs generated:
  - data/metadata/person_e_results.json
  - data/metadata/corridor_performance.csv
  - data/metadata/sensor_error_rankings.csv
  - figures/person_e_horizon_metrics_comparison.png
  - figures/person_e_peak_vs_offpeak_analysis.png
  - figures/person_e_congestion_map_morning_peak.png
  - figures/person_e_congestion_map_evening_peak.png
  - figures/person_e_spatial_error_corridors.png
  - figures/person_e_feature_importance_comparison.png
  - figures/person_e_rf_ablation_progression.png
  - figures/person_e_graph_propagation_gap.png
  - reports/person_e_report.md
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.evaluation import (
    build_peak_offpeak_breakdown,
    build_rf_ablation_summary,
    build_unified_test_benchmark,
    compute_horizon_degradation,
    load_person_c_results,
    load_person_d_results,
)
from src.visualization.benchmark_plots import (
    plot_feature_importance_comparison,
    plot_graph_gap_diagram,
    plot_horizon_comparison,
    plot_peak_vs_offpeak,
    plot_rf_ablation_progression,
)
from src.visualization.spatial_mapping import (
    compute_corridor_statistics,
    load_sensor_geo_metadata,
    plot_corridor_error_distribution,
    plot_spatial_congestion_map,
)


def run_person_e_pipeline(save_figures: bool = True) -> dict[str, Any]:
    """Execute complete Person E evaluation and synthesis pipeline."""
    start_time = time.time()
    print("=" * 78)
    print("PERSON E — Phase 1 Benchmark Synthesis & Spatial Evaluation")
    print("=" * 78)

    # 1. Ingest metadata
    print("[1/6] Loading Person C (XGBoost/baselines) and Person D (Random Forest) metadata...")
    c_data = load_person_c_results()
    d_data = load_person_d_results()
    print(f"  ✓ Loaded Person C results: {len(c_data['horizons'])} horizons")
    print(f"  ✓ Loaded Person D results: {len(d_data['experiments'])} experiments (A, B, C, D)")

    # 2. Ingest sensor locations and highway corridors
    print("[2/6] Ingesting 207 sensor geospatial coordinates and traffic statistics...")
    df_sensors = load_sensor_geo_metadata()
    corridor_stats = compute_corridor_statistics(df_sensors)
    print(f"  ✓ Processed {len(df_sensors)} highway sensors across {len(corridor_stats)} freeway corridors")

    # 3. Build unified comparison tables
    print("[3/6] Compiling unified multi-model benchmarks across all 4 horizons...")
    df_test_bench = build_unified_test_benchmark(c_data, d_data)
    df_slices = build_peak_offpeak_breakdown(c_data)
    df_ablations = build_rf_ablation_summary(d_data)
    df_degradation = compute_horizon_degradation(df_test_bench)
    print("  ✓ Unified benchmark matrix constructed (20 model-horizon pairs)")

    # 4. Sensor Error Rankings
    print("[4/6] Ranking sensors by volatility and congestion drop...")
    df_sensor_ranks = df_sensors[[
        "sensor_id", "corridor", "latitude", "longitude", "valid_mean", "valid_std",
        "valid_fraction", "morning_peak_speed", "evening_peak_speed"
    ]].copy()
    df_sensor_ranks["evening_drop_pct"] = round(
        ((df_sensor_ranks["valid_mean"] - df_sensor_ranks["evening_peak_speed"]) / df_sensor_ranks["valid_mean"]) * 100.0, 2
    )
    df_sensor_ranks = df_sensor_ranks.sort_values(by="valid_std", ascending=False).reset_index(drop=True)

    # 5. Export metadata and tables
    print("[5/6] Exporting structured metadata artifacts...")
    metadata_dir = ROOT / "data" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    corridor_csv = metadata_dir / "corridor_performance.csv"
    corridor_stats.to_csv(corridor_csv, index=False)
    print(f"  ✓ Saved {corridor_csv}")

    sensor_ranks_csv = metadata_dir / "sensor_error_rankings.csv"
    df_sensor_ranks.to_csv(sensor_ranks_csv, index=False)
    print(f"  ✓ Saved {sensor_ranks_csv}")

    # Build comprehensive Person E results JSON
    e_results = {
        "metadata": {
            "author": "Person E",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "evaluation_standard": "DCRNN Benchmark (Li et al. 2018)",
            "test_split_boundary": "2012-06-04 04:40 to 2012-06-27 22:55",
            "n_test_origins": 6844,
            "n_sensors": 207,
            "speed_units": "source units (likely mph, unverified)",
        },
        "unified_test_benchmark": df_test_bench.to_dict(orient="records"),
        "temporal_slices": df_slices.to_dict(orient="records"),
        "rf_ablations": df_ablations.to_dict(orient="records"),
        "horizon_degradation": df_degradation.to_dict(orient="records"),
        "corridor_performance": corridor_stats.to_dict(orient="records"),
    }

    results_json = metadata_dir / "person_e_results.json"
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(e_results, f, indent=2)
    print(f"  ✓ Saved {results_json}")

    # 6. Render figures
    if save_figures:
        print("[6/6] Rendering high-resolution figures in figures/...")
        fig_dir = ROOT / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        p1 = plot_horizon_comparison(df_test_bench)
        print(f"  ✓ Rendered {p1.name}")

        p2 = plot_peak_vs_offpeak(df_slices)
        print(f"  ✓ Rendered {p2.name}")

        p3 = plot_spatial_congestion_map(df_sensors, peak_type="morning")
        print(f"  ✓ Rendered {p3.name}")

        p4 = plot_spatial_congestion_map(df_sensors, peak_type="evening")
        print(f"  ✓ Rendered {p4.name}")

        p5 = plot_corridor_error_distribution(df_sensors)
        print(f"  ✓ Rendered {p5.name}")

        p6 = plot_feature_importance_comparison(c_data, d_data)
        print(f"  ✓ Rendered {p6.name}")

        p7 = plot_rf_ablation_progression(d_data)
        print(f"  ✓ Rendered {p7.name}")

        p8 = plot_graph_gap_diagram()
        print(f"  ✓ Rendered {p8.name}")

    elapsed = time.time() - start_time
    print("-" * 78)
    print(f"PERSON E PIPELINE COMPLETE in {elapsed:.2f} seconds.")
    print("=" * 78)

    # Print Headline Test Benchmark Table
    print("\nHEADLINE TEST BENCHMARK (MAE):")
    piv_mae = df_test_bench.pivot(index="model", columns="horizon", values="mae")
    print(piv_mae[["5min", "15min", "30min", "60min"]].to_string())

    return e_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Person E evaluation and benchmark pipeline")
    parser.add_argument("--no-plots", action="store_true", help="Skip rendering matplotlib figures")
    args = parser.parse_args()

    run_person_e_pipeline(save_figures=not args.no_plots)
