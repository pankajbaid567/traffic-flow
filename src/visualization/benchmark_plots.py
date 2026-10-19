"""Benchmark comparison plots and visual gap analysis for Phase 1 models."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "figures"


MODEL_COLORS = {
    "persistence": "#94a3b8",            # Slate gray
    "historical_average": "#f59e0b",     # Amber
    "random_forest_32": "#10b981",       # Emerald green
    "xgboost_32": "#3b82f6",             # Blue
    "xgboost_ha": "#8b5cf6",             # Purple
}

MODEL_LABELS = {
    "persistence": "Persistence (t)",
    "historical_average": "Historical Average",
    "random_forest_32": "Random Forest (Full 32)",
    "xgboost_32": "XGBoost (Shared 32)",
    "xgboost_ha": "XGBoost + Hist. Avg (36)",
}


def plot_horizon_comparison(
    df_test: pd.DataFrame,
    output_path: Path | None = None,
) -> Path:
    """Plot MAE, RMSE, and MAPE across forecast horizons (5m, 15m, 30m, 60m)."""
    if output_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        output_path = FIGURES_DIR / "person_e_horizon_metrics_comparison.png"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    fig.patch.set_facecolor("#0f172a")

    horizons = ["5min", "15min", "30min", "60min"]
    x = np.arange(len(horizons))

    metrics = [
        ("mae", "Mean Absolute Error (MAE)", "Speed Units (mph)", axes[0]),
        ("rmse", "Root Mean Squared Error (RMSE)", "Speed Units (mph)", axes[1]),
        ("mape", "Mean Absolute Percentage Error (MAPE)", "Percentage (%)", axes[2]),
    ]

    for metric, title, ylabel, ax in metrics:
        ax.set_facecolor("#1e293b")
        for model in ["persistence", "historical_average", "random_forest_32", "xgboost_32", "xgboost_ha"]:
            sub = df_test[df_test["model"] == model].set_index("horizon").reindex(horizons)
            ax.plot(
                x,
                sub[metric],
                marker="o",
                linewidth=2.2,
                markersize=7,
                label=MODEL_LABELS.get(model, model),
                color=MODEL_COLORS.get(model, "#ffffff"),
            )

        ax.set_title(title, color="#f8fafc", fontsize=12, fontweight="bold", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(["+5 min", "+15 min", "+30 min", "+60 min"], color="#f1f5f9", fontsize=10)
        ax.set_ylabel(ylabel, color="#cbd5e1", fontsize=10)
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, linestyle="--", alpha=0.25, color="#64748b")
        for spine in ax.spines.values():
            spine.set_color("#475569")

    # Unified Legend on first axis
    axes[0].legend(loc="upper left", facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc", fontsize=9)
    plt.suptitle("Phase 1 Benchmark: Multi-Horizon Forecast Performance (Test Set)", color="#f8fafc", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    return output_path


def plot_peak_vs_offpeak(
    df_slices: pd.DataFrame,
    output_path: Path | None = None,
) -> Path:
    """Plot forecast error across temporal slices and rush-hour regimes."""
    if output_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        output_path = FIGURES_DIR / "person_e_peak_vs_offpeak_analysis.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    fig.patch.set_facecolor("#0f172a")
    ax1.set_facecolor("#1e293b")
    ax2.set_facecolor("#1e293b")

    # Panel 1: 15-min horizon error across temporal slices
    h15 = df_slices[df_slices["horizon"] == "15min"].copy()
    slice_order = ["all", "off_peak", "peak", "morning_peak", "evening_peak"]
    slice_labels = ["Overall (All)", "Off-Peak", "Peak (Combined)", "Morning Peak\n(07:00-10:00)", "Evening Peak\n(16:00-19:00)"]
    h15 = h15.set_index("period").reindex(slice_order)

    x = np.arange(len(slice_order))
    width = 0.22

    ax1.bar(x - width*1.5, h15["persistence_mae"], width, label="Persistence", color=MODEL_COLORS["persistence"], alpha=0.9)
    ax1.bar(x - width*0.5, h15["historical_average_mae"], width, label="Hist. Avg", color=MODEL_COLORS["historical_average"], alpha=0.9)
    ax1.bar(x + width*0.5, h15["xgboost_mae"], width, label="XGBoost (32)", color=MODEL_COLORS["xgboost_32"], alpha=0.9)
    ax1.bar(x + width*1.5, h15["xgboost_ha_mae"], width, label="XGBoost + HA", color=MODEL_COLORS["xgboost_ha"], alpha=0.9)

    ax1.set_title("15-Minute Forecast MAE by Temporal Period (Test)", color="#f8fafc", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(slice_labels, color="#f1f5f9", fontsize=9)
    ax1.set_ylabel("MAE (Speed Units)", color="#cbd5e1", fontsize=10)
    ax1.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.25, color="#64748b")
    ax1.tick_params(colors="#94a3b8")

    # Panel 2: Peak vs Off-Peak Error Escalation across Horizons (XGBoost + HA)
    horizons = ["5min", "15min", "30min", "60min"]
    hx = np.arange(len(horizons))

    offpeak_mae = [df_slices.loc[(df_slices["horizon"] == h) & (df_slices["period"] == "off_peak"), "xgboost_ha_mae"].values[0] for h in horizons]
    morning_mae = [df_slices.loc[(df_slices["horizon"] == h) & (df_slices["period"] == "morning_peak"), "xgboost_ha_mae"].values[0] for h in horizons]
    evening_mae = [df_slices.loc[(df_slices["horizon"] == h) & (df_slices["period"] == "evening_peak"), "xgboost_ha_mae"].values[0] for h in horizons]

    ax2.plot(hx, offpeak_mae, marker="s", linewidth=2.2, label="Off-Peak", color="#38bdf8")
    ax2.plot(hx, morning_mae, marker="^", linewidth=2.2, label="Morning Peak (07-10h)", color="#facc15")
    ax2.plot(hx, evening_mae, marker="D", linewidth=2.5, label="Evening Peak (16-19h)", color="#f43f5e")

    ax2.set_title("Rush-Hour Error Escalation by Horizon (XGBoost + HA)", color="#f8fafc", fontsize=12, fontweight="bold")
    ax2.set_xticks(hx)
    ax2.set_xticklabels(["+5 min", "+15 min", "+30 min", "+60 min"], color="#f1f5f9", fontsize=10)
    ax2.set_ylabel("MAE (Speed Units)", color="#cbd5e1", fontsize=10)
    ax2.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.25, color="#64748b")
    ax2.tick_params(colors="#94a3b8")

    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_color("#475569")

    plt.suptitle("Peak vs Off-Peak Error Analysis: Evening Rush-Hour Vulnerability", color="#f8fafc", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    return output_path


def plot_feature_importance_comparison(
    c_data: dict[str, Any],
    d_data: dict[str, Any],
    output_path: Path | None = None,
) -> Path:
    """Compare top influential features in Random Forest (Gini) vs XGBoost (Gain)."""
    if output_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        output_path = FIGURES_DIR / "person_e_feature_importance_comparison.png"

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=300)
    fig.patch.set_facecolor("#0f172a")

    # 5-min Horizon
    rf_5 = d_data["experiments"]["D"]["5min"]["feature_importances"]
    xgb_5 = c_data["horizons"]["5min"]["feature_importance_gain"]

    rf_5_top = sorted(rf_5.items(), key=lambda x: x[1], reverse=True)[:5]
    xgb_5_top = sorted(xgb_5.items(), key=lambda x: x[1], reverse=True)[:5]

    # 60-min Horizon
    rf_60 = d_data["experiments"]["D"]["60min"]["feature_importances"]
    xgb_60 = c_data["horizons"]["60min"]["feature_importance_gain"]

    rf_60_top = sorted(rf_60.items(), key=lambda x: x[1], reverse=True)[:5]
    xgb_60_top = sorted(xgb_60.items(), key=lambda x: x[1], reverse=True)[:5]

    plots_cfg = [
        (axes[0, 0], rf_5_top, "Random Forest: Top 5 Features (+5 min)", "Gini Impurity Importance", "#10b981"),
        (axes[0, 1], xgb_5_top, "XGBoost: Top 5 Features (+5 min)", "Normalized Gain", "#3b82f6"),
        (axes[1, 0], rf_60_top, "Random Forest: Top 5 Features (+60 min)", "Gini Impurity Importance", "#10b981"),
        (axes[1, 1], xgb_60_top, "XGBoost: Top 5 Features (+60 min)", "Normalized Gain", "#3b82f6"),
    ]

    for ax, data, title, xlabel, color in plots_cfg:
        ax.set_facecolor("#1e293b")
        names, values = zip(*data[::-1])  # Ascending for horizontal bar
        ax.barh(names, values, color=color, alpha=0.9, height=0.55)
        ax.set_title(title, color="#f8fafc", fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel, color="#cbd5e1", fontsize=9)
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, linestyle="--", alpha=0.25, color="#64748b")
        for spine in ax.spines.values():
            spine.set_color("#475569")

    plt.suptitle("Feature Importance Shift: Immediate Lag Dominance (5m) vs Temporal Dynamics (60m)", color="#f8fafc", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    return output_path


def plot_rf_ablation_progression(
    d_data: dict[str, Any],
    output_path: Path | None = None,
) -> Path:
    """Plot Person D's 4 feature ablation tiers across horizons."""
    if output_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        output_path = FIGURES_DIR / "person_e_rf_ablation_progression.png"

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    horizons = ["5min", "15min", "30min", "60min"]
    x = np.arange(len(horizons))
    exp_colors = {"A": "#f59e0b", "B": "#38bdf8", "C": "#818cf8", "D": "#10b981"}
    exp_labels = {
        "A": "Exp A: 12 Lags Only",
        "B": "Exp B: Lags + Temporal (19 feats)",
        "C": "Exp C: Lags + Rolling (23 feats)",
        "D": "Exp D: Full 32 Features",
    }

    # Plot persistence baseline as reference
    pers_mae = [d_data["experiments"]["A"][h]["test_persistence_mae"] for h in horizons]
    ax.plot(x, pers_mae, linestyle="--", marker="x", label="Persistence Baseline", color="#94a3b8", linewidth=1.8)

    for code in ["A", "B", "C", "D"]:
        maes = [d_data["experiments"][code][h]["test_metrics"]["mae"] for h in horizons]
        ax.plot(x, maes, marker="o", linewidth=2.2, label=exp_labels[code], color=exp_colors[code])

    ax.set_title("Person D Random Forest Ablation Progression (Test Set MAE)", color="#f8fafc", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["+5 min", "+15 min", "+30 min", "+60 min"], color="#f1f5f9", fontsize=10)
    ax.set_ylabel("MAE (Speed Units)", color="#cbd5e1", fontsize=10)
    ax.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.25, color="#64748b")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#475569")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    return output_path


def plot_graph_gap_diagram(output_path: Path | None = None) -> Path:
    """Render conceptual diagram demonstrating the gap between independent AML and Graph Learning."""
    if output_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        output_path = FIGURES_DIR / "person_e_graph_propagation_gap.png"

    fig, ax = plt.subplots(figsize=(13, 7), dpi=300)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    ax.axis("off")

    # Diagram Elements: Freeway segment with 4 sensors
    ax.text(0.5, 0.93, "The AML-to-DL Gap: Shockwave Propagation & Graph Blindness", color="#f8fafc", fontsize=14, fontweight="bold", ha="center")
    ax.text(0.5, 0.86, "Why independent segment models fail at 30-60 min and why GAT/DCRNN are required in Phase 2", color="#94a3b8", fontsize=10, ha="center")

    # Draw Roadway
    ax.plot([0.1, 0.9], [0.55, 0.55], color="#475569", linewidth=18, zorder=1)
    ax.plot([0.1, 0.9], [0.55, 0.55], color="#cbd5e1", linestyle="--", linewidth=1.5, zorder=2)

    # Traffic Flow Arrow
    ax.annotate("Traffic Direction (Eastbound)", xy=(0.85, 0.65), xytext=(0.15, 0.65),
                arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=3),
                color="#38bdf8", fontsize=11, fontweight="bold", ha="center")

    # Congestion Shockwave Arrow (Propagating Upstream!)
    ax.annotate("Shockwave Spillback Propagation (Westbound / Upstream)", xy=(0.20, 0.42), xytext=(0.80, 0.42),
                arrowprops=dict(arrowstyle="->", color="#ef4444", lw=3.5, linestyle=":"),
                color="#ef4444", fontsize=11, fontweight="bold", ha="center")

    # Sensors: S1 (Free flow), S2 (Transition), S3 (Queue tail), S4 (Bottleneck origin)
    sensors = [
        (0.20, "Sensor A (Upstream)\nv = 62 mph\nBlind to downstream jam", "#10b981"),
        (0.40, "Sensor B\nv = 58 mph\nIncoming shockwave", "#38bdf8"),
        (0.60, "Sensor C\nv = 38 mph\nQueue buildup", "#f59e0b"),
        (0.80, "Sensor D (Bottleneck)\nv = 18 mph\nAccident / Ramp meter", "#ef4444"),
    ]

    for x_pos, label, color in sensors:
        circle = plt.Circle((x_pos, 0.55), 0.035, color=color, zorder=4)
        ax.add_patch(circle)
        ax.text(x_pos, 0.55, "S", color="#0f172a", fontsize=11, fontweight="bold", ha="center", va="center", zorder=5)
        ax.text(x_pos, 0.28, label, color="#f1f5f9", fontsize=9, ha="center", va="top")

    # Explanatory Box
    gap_text = (
        "Phase 1 Applied ML Limitation (Random Forest & XGBoost):\n"
        "• Each sensor model uses only its own historical lags (speed_lag_0 ... speed_lag_11).\n"
        "• At t+15 min, Sensor A still measures 62 mph and predicts 61 mph.\n"
        "• But by t+30 min, the shockwave from Sensor D hits Sensor A, dropping speed to 25 mph!\n"
        "• Result: Massive forecast error at 30m & 60m because local AML models cannot observe adjacent links.\n\n"
        "Phase 2 Deep Learning Solution (Graph Attention Networks / Diffusion GNN):\n"
        "• Network aware: Propagates hidden states along directed road graph adjacency matrix A_ij.\n"
        "• Dynamic attention α_ij: Learns time-varying influence of downstream congestion on upstream links."
    )
    props = dict(boxstyle="round,pad=0.8", facecolor="#1e293b", edgecolor="#6366f1", linewidth=1.5)
    ax.text(0.5, 0.02, gap_text, color="#e2e8f0", fontsize=9.5, ha="center", va="bottom", bbox=props)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    return output_path
