"""Visualization suite for Phase 2 forecast evaluation and spatial error analysis (Member 4).

Contract: reports/phase2_plan.md section 4.8, 8.4.
All figures conform to 300 DPI, rigorous scientific labeling:
- 'source speed units'
- 'stored clock'
- 'coordinate regions'
- 'proxy congestion'
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.prediction_evaluation import geographic_regions
from src.phase2.common import HORIZON_NAMES, ROOT

FIGURES_DIR = ROOT / "figures" / "phase2"
METADATA_DIR = ROOT / "data" / "metadata"


def ensure_figures_dir(path: Path = FIGURES_DIR) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_horizon_mae(
    metrics_summary_df: pd.DataFrame,
    output_path: Optional[Path | str] = None,
) -> Path:
    """Plot MAE vs forecast horizon (5, 15, 30, 60 min) across models."""
    out_dir = ensure_figures_dir()
    output_path = Path(output_path) if output_path else out_dir / "fig_horizon_mae.png"

    df = metrics_summary_df[
        (metrics_summary_df["split"] == "test") & (metrics_summary_df["period"] == "all")
    ]
    if df.empty:
        df = metrics_summary_df[metrics_summary_df["period"] == "all"]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    horizons = ["5min", "15min", "30min", "60min"]
    x = np.arange(len(horizons))

    models = df["model"].unique()
    styles = {
        "persistence": ("--", "grey", "Persistence (Baseline)"),
        "historical_average": (":", "orange", "Historical Average"),
        "random_forest_32": ("-.", "forestgreen", "Random Forest 32"),
        "xgboost_32": ("-.", "dodgerblue", "XGBoost 32"),
        "xgboost_ha": ("-", "navy", "XGBoost + HA 36"),
        "gru_local": ("-.", "crimson", "GRU Local (Control)"),
        "gat": ("-", "purple", "GAT (Graph)"),
        "dcrnn": ("-", "darkgreen", "DCRNN (Graph)"),
    }

    for m in models:
        sub = df[df["model"] == m].set_index("horizon").reindex(horizons)
        if sub["mae_mean"].isna().all():
            continue
        ls, color, label = styles.get(m, ("-", None, m))
        ax.plot(
            x, sub["mae_mean"], marker="o", linestyle=ls, color=color, label=label, linewidth=1.8
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["5 min", "15 min", "30 min", "60 min"])
    ax.set_ylabel("Mean Absolute Error (source speed units)")
    ax.set_xlabel("Forecast Horizon")
    ax.set_title("Multi-Horizon Forecasting Accuracy on Held-Out Test Split")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_peak_vs_offpeak(
    metrics_summary_df: pd.DataFrame,
    output_path: Optional[Path | str] = None,
) -> Path:
    """Bar chart comparing peak vs off-peak MAE at 15 and 60 min."""
    out_dir = ensure_figures_dir()
    output_path = Path(output_path) if output_path else out_dir / "fig_peak_offpeak.png"

    sub = metrics_summary_df[
        (metrics_summary_df["split"] == "test")
        & (metrics_summary_df["period"].isin(["peak", "off_peak"]))
        & (metrics_summary_df["horizon"] == "60min")
    ]
    if sub.empty:
        sub = metrics_summary_df[
            (metrics_summary_df["period"].isin(["peak", "off_peak"]))
            & (metrics_summary_df["horizon"] == "60min")
        ]

    models = [m for m in sub["model"].unique() if m != "last_value"]
    if not models:
        models = sub["model"].unique()

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    x = np.arange(len(models))
    width = 0.35

    peak_vals = [
        sub[(sub["model"] == m) & (sub["period"] == "peak")]["mae_mean"].values[0]
        if not sub[(sub["model"] == m) & (sub["period"] == "peak")].empty else np.nan
        for m in models
    ]
    off_vals = [
        sub[(sub["model"] == m) & (sub["period"] == "off_peak")]["mae_mean"].values[0]
        if not sub[(sub["model"] == m) & (sub["period"] == "off_peak")].empty else np.nan
        for m in models
    ]

    ax.bar(x - width / 2, peak_vals, width, label="Peak (07-10h, 16-19h stored clock)", color="#d95f02")
    ax.bar(x + width / 2, off_vals, width, label="Off-Peak", color="#7570b3")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylabel("60-Minute Forecast MAE (source speed units)")
    ax.set_title("Peak vs. Off-Peak Forecast Degradation (60-minute horizon)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_graph_ablation(
    metrics_summary_df: pd.DataFrame,
    output_path: Optional[Path | str] = None,
) -> Path:
    """Compare GAT vs gat_identity vs gat_shuffled."""
    out_dir = ensure_figures_dir()
    output_path = Path(output_path) if output_path else out_dir / "fig_graph_ablation.png"

    ablation_models = ["gat", "gat_identity", "gat_shuffled"]
    sub = metrics_summary_df[
        (metrics_summary_df["model"].isin(ablation_models))
        & (metrics_summary_df["period"] == "all")
        & (metrics_summary_df["split"] == "test")
    ]
    if sub.empty:
        sub = metrics_summary_df[
            (metrics_summary_df["model"].isin(ablation_models))
            & (metrics_summary_df["period"] == "all")
        ]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    horizons = ["5min", "15min", "30min", "60min"]
    x = np.arange(len(horizons))

    colors = {"gat": "purple", "gat_identity": "grey", "gat_shuffled": "crimson"}
    labels = {
        "gat": "GAT (Road Network Adjacency)",
        "gat_identity": "GAT Identity (No Graph Control)",
        "gat_shuffled": "GAT Shuffled (Permuted Graph Control)",
    }

    plotted_any = False
    for m in ablation_models:
        m_sub = sub[sub["model"] == m].set_index("horizon").reindex(horizons)
        if m_sub["mae_mean"].isna().all():
            continue
        ax.plot(
            x, m_sub["mae_mean"], marker="s", color=colors.get(m, "blue"),
            label=labels.get(m, m), linewidth=1.8
        )
        plotted_any = True

    if not plotted_any:
        ax.text(0.5, 0.5, "Graph ablation models pending training runs",
                ha="center", va="center", transform=ax.transAxes, color="grey")

    ax.set_xticks(x)
    ax.set_xticklabels(["5 min", "15 min", "30 min", "60 min"])
    ax.set_ylabel("MAE (source speed units)")
    ax.set_title("Graph Architecture & Topology Ablation on Test Partition")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_seed_spread(
    metrics_df: pd.DataFrame,
    output_path: Optional[Path | str] = None,
) -> Path:
    """Plot seed-to-seed variance across models at 60 min."""
    out_dir = ensure_figures_dir()
    output_path = Path(output_path) if output_path else out_dir / "fig_seed_spread.png"

    sub = metrics_df[
        (metrics_df["horizon"] == "60min")
        & (metrics_df["period"] == "all")
        & (metrics_df["split"] == "test")
    ]
    if sub.empty:
        sub = metrics_df[(metrics_df["horizon"] == "60min") & (metrics_df["period"] == "all")]

    multi_seed_models = [
        m for m, g in sub.groupby("model") if g["seed"].nunique() > 1
    ]

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    if multi_seed_models:
        for idx, m in enumerate(multi_seed_models):
            m_sub = sub[sub["model"] == m]
            seeds = m_sub["seed"].values
            maes = m_sub["mae"].values
            ax.scatter([idx] * len(maes), maes, label=m, s=50)
            ax.plot([idx - 0.1, idx + 0.1], [np.mean(maes), np.mean(maes)], color="black", lw=2)
        ax.set_xticks(range(len(multi_seed_models)))
        ax.set_xticklabels(multi_seed_models)
        ax.set_ylabel("60-min MAE (source speed units)")
        ax.set_title("Seed-to-Seed Stability across Seeds 42, 43, 44")
    else:
        ax.text(0.5, 0.5, "Single seed evaluated (multi-seed runs pending)",
                ha="center", va="center", transform=ax.transAxes, color="grey")
        ax.set_title("Seed-to-Seed Stability")

    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_error_by_degree(
    sensor_scores_df: pd.DataFrame,
    output_path: Optional[Path | str] = None,
) -> Path:
    """Scatter/binned error vs graph degree."""
    out_dir = ensure_figures_dir()
    output_path = Path(output_path) if output_path else out_dir / "fig_error_by_degree.png"

    sub = sensor_scores_df[
        (sensor_scores_df["horizon"] == "60min")
        & (sensor_scores_df["period"] == "all")
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    if not sub.empty and "graph_degree" in sub.columns and not sub["graph_degree"].isna().all():
        for m in sub["model"].unique():
            if m in ["xgboost_ha", "gat", "dcrnn", "persistence"]:
                m_sub = sub[sub["model"] == m]
                ax.scatter(m_sub["graph_degree"], m_sub["mae"], alpha=0.5, label=m, s=25)
        ax.set_xlabel("Sensor Graph Degree (Number of Directed Neighbours)")
        ax.set_ylabel("60-min MAE (source speed units)")
        ax.set_title("Forecast Error vs. Highway Sensor Connectivity")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Sensor degree data not attached or pending",
                ha="center", va="center", transform=ax.transAxes, color="grey")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_spatial_comparison_map(
    sensor_scores_df: pd.DataFrame,
    metadata_df: Optional[pd.DataFrame] = None,
    baseline_model: str = "xgboost_ha",
    candidate_model: str = "gat",
    horizon: str = "15min",
    period: str = "evening_peak",
    output_path: Optional[Path | str] = None,
) -> Path:
    """4-panel spatial error comparison map with shared colorbar (plan 4.8).

    Panels: Observed Mean, Baseline MAE, Candidate MAE, and Difference (Baseline - Candidate).
    """
    out_dir = ensure_figures_dir()
    if output_path is None:
        period_short = "evening" if "evening" in period else "morning"
        output_path = out_dir / f"fig_map_{period_short}_{horizon}.png"
    output_path = Path(output_path)

    if metadata_df is None:
        from src.visualization.spatial_mapping import load_sensor_geo_metadata

        try:
            metadata_df = load_sensor_geo_metadata()
        except Exception:
            metadata_df = pd.read_csv(
                METADATA_DIR / "sensors.csv", dtype={"sensor_id": str}
            )

    meta = metadata_df[["sensor_id", "latitude", "longitude"]].copy()
    meta["sensor_id"] = meta["sensor_id"].astype(str)

    b_scores = sensor_scores_df[
        (sensor_scores_df["model"] == baseline_model)
        & (sensor_scores_df["horizon"] == horizon)
        & (sensor_scores_df["period"] == period)
    ]
    c_scores = sensor_scores_df[
        (sensor_scores_df["model"] == candidate_model)
        & (sensor_scores_df["horizon"] == horizon)
        & (sensor_scores_df["period"] == period)
    ]

    fig, axes = plt.subplots(
        1, 4, figsize=(16, 4), dpi=300, constrained_layout=True
    )

    if not b_scores.empty and not c_scores.empty:
        merged = (
            meta.merge(
                b_scores[["sensor_id", "actual_mean", "mae"]], on="sensor_id"
            )
            .rename(columns={"mae": "mae_base"})
            .merge(c_scores[["sensor_id", "mae"]], on="sensor_id")
            .rename(columns={"mae": "mae_cand"})
        )
        merged["mae_diff"] = (
            merged["mae_base"] - merged["mae_cand"]
        )  # positive means candidate is better

        # Panel 1: Observed Mean
        sc0 = axes[0].scatter(
            merged.longitude,
            merged.latitude,
            c=merged.actual_mean,
            cmap="RdYlGn",
            s=15,
            vmin=15,
            vmax=70,
        )
        axes[0].set(
            title="Observed Mean Speed", xlabel="Longitude", ylabel="Latitude"
        )
        fig.colorbar(
            sc0, ax=axes[0], shrink=0.8, label="source units (likely mph)"
        )

        # Panel 2: Baseline MAE
        sc1 = axes[1].scatter(
            merged.longitude,
            merged.latitude,
            c=merged.mae_base,
            cmap="magma_r",
            s=15,
            vmin=0,
            vmax=12,
        )
        axes[1].set(title=f"{baseline_model} MAE", xlabel="Longitude")
        fig.colorbar(sc1, ax=axes[1], shrink=0.8, label="MAE")

        # Panel 3: Candidate MAE
        sc2 = axes[2].scatter(
            merged.longitude,
            merged.latitude,
            c=merged.mae_cand,
            cmap="magma_r",
            s=15,
            vmin=0,
            vmax=12,
        )
        axes[2].set(title=f"{candidate_model} MAE", xlabel="Longitude")
        fig.colorbar(sc2, ax=axes[2], shrink=0.8, label="MAE")

        # Panel 4: Difference Map (Positive = Candidate beats Baseline)
        vlimit = max(0.5, float(np.abs(merged.mae_diff).quantile(0.95)))
        sc3 = axes[3].scatter(
            merged.longitude,
            merged.latitude,
            c=merged.mae_diff,
            cmap="RdBu",
            s=15,
            vmin=-vlimit,
            vmax=vlimit,
        )
        axes[3].set(
            title="Gain (Base MAE - Cand MAE)",
            xlabel="Longitude",
        )
        fig.colorbar(sc3, ax=axes[3], shrink=0.8, label="MAE Gain")
    else:
        for ax in axes:
            ax.text(
                0.5,
                0.5,
                "Spatial map pending prediction data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="grey",
            )
            ax.set(xlabel="Longitude")
        axes[0].set(ylabel="Latitude")

    fig.suptitle(
        f"Spatial Forecast Comparison: {baseline_model} vs. {candidate_model} ({horizon}, {period.replace('_', ' ')})",
        fontsize=12,
    )
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_congestion_spatial_map(
    sensor_scores_df: pd.DataFrame,
    metadata_df: Optional[pd.DataFrame] = None,
    model: str = "xgboost_ha",
    horizon: str = "15min",
    period: str = "evening_peak",
    output_path: Optional[Path | str] = None,
) -> Path:
    """Observed vs predicted proxy congestion frequency on highway network map."""
    out_dir = ensure_figures_dir()
    if output_path is None:
        period_short = "evening" if "evening" in period else "morning"
        output_path = out_dir / f"fig_congestion_map_{period_short}_{horizon}.png"
    output_path = Path(output_path)

    if metadata_df is None:
        try:
            from src.visualization.spatial_mapping import (
                load_sensor_geo_metadata,
            )

            metadata_df = load_sensor_geo_metadata()
        except Exception:
            metadata_df = pd.read_csv(
                METADATA_DIR / "sensors.csv", dtype={"sensor_id": str}
            )

    meta = metadata_df[["sensor_id", "latitude", "longitude"]].copy()
    meta["sensor_id"] = meta["sensor_id"].astype(str)

    scores = sensor_scores_df[
        (sensor_scores_df["model"] == model)
        & (sensor_scores_df["horizon"] == horizon)
        & (sensor_scores_df["period"] == period)
    ]

    fig, axes = plt.subplots(
        1, 2, figsize=(9, 4), dpi=300, constrained_layout=True
    )
    if (
        not scores.empty
        and "actual_congestion_rate" in scores.columns
        and "predicted_congestion_rate" in scores.columns
    ):
        merged = meta.merge(scores, on="sensor_id")
        sc0 = axes[0].scatter(
            merged.longitude,
            merged.latitude,
            c=merged.actual_congestion_rate,
            cmap="YlOrRd",
            s=15,
            vmin=0,
            vmax=1,
        )
        axes[0].set(
            title="Observed Congestion Rate",
            xlabel="Longitude",
            ylabel="Latitude",
        )
        fig.colorbar(sc0, ax=axes[0], shrink=0.8, label="Frequency")

        sc1 = axes[1].scatter(
            merged.longitude,
            merged.latitude,
            c=merged.predicted_congestion_rate,
            cmap="YlOrRd",
            s=15,
            vmin=0,
            vmax=1,
        )
        axes[1].set(title=f"Predicted ({model})", xlabel="Longitude")
        fig.colorbar(sc1, ax=axes[1], shrink=0.8, label="Frequency")
    else:
        for ax in axes:
            ax.text(
                0.5,
                0.5,
                "Congestion map pending prediction data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="grey",
            )
            ax.set(xlabel="Longitude")
        axes[0].set(ylabel="Latitude")

    fig.suptitle(
        f"Highway Congestion Proxy Map ({horizon}, {period.replace('_', ' ')})\n"
        "Congestion proxy: speed < 0.6 * training 85th percentile",
        fontsize=10,
    )
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_attention_example(
    attention_path: Optional[Path | str] = None,
    output_path: Optional[Path | str] = None,
) -> Optional[Path]:
    """Plot multi-head attention weight distribution from GAT attention export."""
    out_dir = ensure_figures_dir()
    output_path = (
        Path(output_path) if output_path else out_dir / "fig_attention_example.png"
    )

    if attention_path is None or not Path(attention_path).is_file():
        # Skip gracefully if attention file absent
        return None

    try:
        with np.load(attention_path) as data:
            attn = data["attn"]  # [origins, layers, heads, 207, 207]
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        # Plot mean attention across heads for first origin and layer
        mean_attn = attn[0, -1].mean(axis=0)  # [207, 207]
        im = ax.imshow(
            mean_attn, cmap="viridis", interpolation="nearest", rasterized=True
        )
        ax.set_title(
            "GAT Multi-Head Attention Matrix (Sample Peak Origin)\n"
            "Weights reflect learned feature weighting, not physical shockwave speed",
            fontsize=9,
        )
        ax.set_xlabel("Key Sensor Index")
        ax.set_ylabel("Query Sensor Index")
        fig.colorbar(im, ax=ax, label="Attention Weight")
        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return output_path
    except Exception:
        return None


def render_all_figures(
    metrics_df: pd.DataFrame,
    metrics_summary_df: pd.DataFrame,
    sensor_scores_df: pd.DataFrame,
    attention_path: Optional[Path | str] = None,
) -> Dict[str, Path]:
    """Render all Phase 2 figures specified in plan 4.8."""
    out_dir = ensure_figures_dir()
    figures = {}

    figures["fig_horizon_mae"] = plot_horizon_mae(
        metrics_summary_df, out_dir / "fig_horizon_mae.png"
    )
    figures["fig_peak_offpeak"] = plot_peak_vs_offpeak(
        metrics_summary_df, out_dir / "fig_peak_offpeak.png"
    )
    figures["fig_graph_ablation"] = plot_graph_ablation(
        metrics_summary_df, out_dir / "fig_graph_ablation.png"
    )
    figures["fig_seed_spread"] = plot_seed_spread(
        metrics_df, out_dir / "fig_seed_spread.png"
    )
    figures["fig_error_by_degree"] = plot_error_by_degree(
        sensor_scores_df, out_dir / "fig_error_by_degree.png"
    )

    # Spatial difference maps for morning & evening at 15m & 60m
    for period in ["morning_peak", "evening_peak"]:
        for h in ["15min", "60min"]:
            p_short = "morning" if "morning" in period else "evening"
            figures[f"fig_map_{p_short}_{h}"] = plot_spatial_comparison_map(
                sensor_scores_df=sensor_scores_df,
                horizon=h,
                period=period,
                output_path=out_dir / f"fig_map_{p_short}_{h}.png",
            )
            figures[f"fig_congestion_map_{p_short}_{h}"] = (
                plot_congestion_spatial_map(
                    sensor_scores_df=sensor_scores_df,
                    horizon=h,
                    period=period,
                    output_path=out_dir / f"fig_congestion_map_{p_short}_{h}.png",
                )
            )

    # Optional attention example
    if attention_path:
        attn_fig = plot_attention_example(
            attention_path, out_dir / "fig_attention_example.png"
        )
        if attn_fig:
            figures["fig_attention_example"] = attn_fig

    return figures

