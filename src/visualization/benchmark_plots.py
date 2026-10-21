"""Print-readable figures generated solely from verified result tables."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "figures"
MODEL_COLORS = {
    "persistence": "#6b7280",
    "historical_average": "#d97706",
    "random_forest_32": "#00876c",
    "xgboost_32": "#2166ac",
    "xgboost_ha": "#762a83",
}
MODEL_LABELS = {
    "persistence": "Persistence",
    "historical_average": "Historical average",
    "random_forest_32": "Random Forest 32",
    "xgboost_32": "XGBoost 32",
    "xgboost_ha": "XGBoost + HA 36",
}
HORIZONS = ["5min", "15min", "30min", "60min"]


def save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_horizon_comparison(df_test, output_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), constrained_layout=True)
    for ax, metric in zip(axes, ["mae", "rmse", "mape"]):
        for model in MODEL_COLORS:
            q = df_test[df_test.model == model].set_index("horizon").reindex(HORIZONS)
            ax.plot(
                [5, 15, 30, 60],
                q[metric],
                marker="o",
                ms=3,
                lw=1.4,
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
            )
        ax.set(
            xlabel="Forecast horizon (min)",
            ylabel=metric.upper() + (" (%)" if metric == "mape" else " (source units)"),
        )
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=7, loc="upper left")
    return save(
        fig, output_path or FIGURES_DIR / "person_e_horizon_metrics_comparison.png"
    )


def plot_peak_vs_offpeak(df_slices, output_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    periods = ["off_peak", "morning_peak", "evening_peak"]
    q = df_slices[df_slices.horizon == "15min"].set_index("period").reindex(periods)
    names = [
        ("persistence", "Persistence"),
        ("historical_average", "Historical average"),
        ("random_forest", "RF 32"),
        ("xgboost", "XGB 32"),
        ("xgboost_ha", "XGB + HA"),
    ]
    for i, (prefix, label) in enumerate(names):
        if f"{prefix}_mae" in q:
            axes[0].bar(
                np.arange(3) + (i - 2) * 0.16, q[f"{prefix}_mae"], 0.15, label=label
            )
    axes[0].set_xticks(range(3), ["Off peak", "Morning", "Evening"])
    axes[0].set(ylabel="MAE (source units)", title="15-minute forecasts")
    axes[0].legend(fontsize=7, ncol=2)
    for period in periods:
        vals = (
            df_slices[df_slices.period == period].set_index("horizon").reindex(HORIZONS)
        )
        axes[1].plot(
            [5, 15, 30, 60],
            vals.xgboost_ha_mae,
            marker="o",
            label=period.replace("_", " "),
        )
    axes[1].set(
        xlabel="Forecast horizon (min)",
        ylabel="MAE (source units)",
        title="XGBoost + HA by origin period",
    )
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    return save(
        fig, output_path or FIGURES_DIR / "person_e_peak_vs_offpeak_analysis.png"
    )


def plot_feature_importance_comparison(c_data, d_data, output_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.5), constrained_layout=True)
    for i, h in enumerate(["5min", "60min"]):
        for j, (values, label) in enumerate(
            [
                (
                    d_data["experiments"]["D"][h]["feature_importances"],
                    "RF squared-error impurity decrease",
                ),
                (
                    c_data["horizons"][h]["feature_importance_gain"],
                    "XGBoost normalized gain",
                ),
            ]
        ):
            top = sorted(values.items(), key=lambda x: x[1])[-5:]
            ax = axes[i, j]
            ax.barh(
                [x[0] for x in top],
                [x[1] for x in top],
                color=["#00876c", "#2166ac"][j],
            )
            ax.set_title(f"{label} | {h}", fontsize=9)
            ax.tick_params(labelsize=8)
    return save(
        fig, output_path or FIGURES_DIR / "person_e_feature_importance_comparison.png"
    )


def plot_rf_ablation_progression(d_data, output_path=None):
    fig, ax = plt.subplots(figsize=(6, 3.8), constrained_layout=True)
    for code, label in [
        ("A", "A: 12 lags"),
        ("B", "B: 19 lags + time"),
        ("C", "C: 27 lags + time + rolling"),
        ("D", "D: 32 full features"),
    ]:
        values = [
            d_data["experiments"][code][h]["test_metrics"]["mae"] for h in HORIZONS
        ]
        ax.plot([5, 15, 30, 60], values, marker="o", label=label)
    ax.set(xlabel="Forecast horizon (min)", ylabel="Test MAE (source units)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    return save(
        fig, output_path or FIGURES_DIR / "person_e_rf_ablation_progression.png"
    )


def plot_graph_gap_diagram(output_path=None):
    fig, ax = plt.subplots(figsize=(7, 2.8), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.plot([0.1, 0.9], [0.55, 0.55], color="#cccccc", lw=12)
    for i, x in enumerate([0.16, 0.38, 0.60, 0.82]):
        ax.scatter(
            [x],
            [0.55],
            s=180,
            color=["#00876c", "#2166ac", "#d97706", "#b2182b"][i],
            zorder=3,
        )
        ax.text(x, 0.33, f"Sensor {chr(65+i)}", ha="center", fontsize=10)
    ax.annotate(
        "Traffic direction",
        xy=(0.9, 0.82),
        xytext=(0.1, 0.82),
        arrowprops={"arrowstyle": "->"},
        ha="left",
        fontsize=10,
    )
    ax.annotate(
        "Possible upstream queue propagation",
        xy=(0.12, 0.12),
        xytext=(0.9, 0.12),
        arrowprops={"arrowstyle": "->", "color": "#b2182b"},
        ha="right",
        fontsize=9,
    )
    ax.set_title("Conceptual motivation for future graph comparison", fontsize=11)
    fig.text(
        0.5,
        0,
        "Illustrative topology only; no propagation speed or graph-model improvement measured in Phase 1.",
        ha="center",
        fontsize=8,
    )
    return save(fig, output_path or FIGURES_DIR / "person_e_graph_propagation_gap.png")
