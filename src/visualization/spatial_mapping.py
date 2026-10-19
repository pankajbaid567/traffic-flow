"""Spatial congestion mapping and corridor error analysis for METR-LA highway network."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
METADATA_DIR = ROOT / "data" / "metadata"
FIGURES_DIR = ROOT / "figures"


def load_sensor_geo_metadata() -> pd.DataFrame:
    """Load the 207 sensors with geospatial coordinates and traffic statistics."""
    sensors_path = METADATA_DIR / "sensors.csv"
    if not sensors_path.exists():
        raise FileNotFoundError(f"sensors.csv not found at: {sensors_path}")
    df_sensors = pd.read_csv(sensors_path, dtype={"sensor_id": str})

    # Join hourly profile if available to calculate empirical peak speeds
    hourly_path = METADATA_DIR / "hourly_mean_by_sensor.csv"
    if hourly_path.exists():
        df_hourly = pd.read_csv(hourly_path)
        # Hours 7, 8, 9 = Morning peak; Hours 16, 17, 18 = Evening peak
        morning_hours = [7, 8, 9]
        evening_hours = [16, 17, 18]

        morning_means = {}
        evening_means = {}
        for s_id in df_sensors["sensor_id"]:
            if s_id in df_hourly.columns:
                morning_means[s_id] = df_hourly.loc[df_hourly["stored_clock_hour"].isin(morning_hours), s_id].mean()
                evening_means[s_id] = df_hourly.loc[df_hourly["stored_clock_hour"].isin(evening_hours), s_id].mean()
            else:
                morning_means[s_id] = np.nan
                evening_means[s_id] = np.nan

        df_sensors["morning_peak_speed"] = df_sensors["sensor_id"].map(morning_means)
        df_sensors["evening_peak_speed"] = df_sensors["sensor_id"].map(evening_means)
    else:
        df_sensors["morning_peak_speed"] = df_sensors["valid_mean"] * 0.85
        df_sensors["evening_peak_speed"] = df_sensors["valid_mean"] * 0.78

    return assign_freeway_corridors(df_sensors)


def assign_freeway_corridors(df: pd.DataFrame) -> pd.DataFrame:
    """Classify 207 sensors into major Los Angeles highway corridors by coordinates."""
    corridors = []
    for _, row in df.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]

        # Known LA Freeway Coordinate Bounding Boxes
        if lon < -118.43:
            # West Los Angeles / Sepulveda Pass corridor
            corridors.append("I-405 (San Diego Fwy)")
        elif lat > 34.18:
            # Northern corridor / Foothills
            corridors.append("I-210 (Foothill Fwy)")
        elif 34.09 <= lat <= 34.18 and -118.40 <= lon <= -118.28:
            # Hollywood / Cahuenga Pass / San Fernando corridor
            corridors.append("US-101 (Hollywood Fwy)")
        elif 34.02 <= lat <= 34.08 and -118.43 <= lon <= -118.26:
            # Central East-West corridor
            corridors.append("I-10 (Santa Monica Fwy)")
        elif 34.03 <= lat <= 34.09 and -118.26 < lon <= -118.22:
            # Downtown Los Angeles junction
            corridors.append("I-110 (Harbor Fwy / DTLA)")
        elif lat < 34.08 and lon > -118.22:
            # East LA / Gateway Cities
            corridors.append("I-5 / I-710 Gateway")
        else:
            corridors.append("I-5 / SR-134 Connectors")

    df["corridor"] = corridors
    return df


def compute_corridor_statistics(df_sensors: pd.DataFrame) -> pd.DataFrame:
    """Aggregate speed and congestion statistics by highway corridor."""
    agg = df_sensors.groupby("corridor").agg(
        sensor_count=("sensor_id", "count"),
        mean_free_speed=("valid_mean", "mean"),
        std_free_speed=("valid_std", "mean"),
        mean_morning_speed=("morning_peak_speed", "mean"),
        mean_evening_speed=("evening_peak_speed", "mean"),
        mean_zero_readings=("zero_count", "mean"),
        mean_valid_fraction=("valid_fraction", "mean"),
    ).reset_index()

    agg["morning_speed_drop_pct"] = round(
        ((agg["mean_free_speed"] - agg["mean_morning_speed"]) / agg["mean_free_speed"]) * 100.0, 2
    )
    agg["evening_speed_drop_pct"] = round(
        ((agg["mean_free_speed"] - agg["mean_evening_speed"]) / agg["mean_free_speed"]) * 100.0, 2
    )
    return agg.sort_values(by="evening_speed_drop_pct", ascending=False)


def plot_spatial_congestion_map(
    df_sensors: pd.DataFrame,
    peak_type: str = "morning",
    output_path: Path | None = None,
) -> Path:
    """Plot physical 2D highway network map with sensor speeds during peak congestion."""
    if output_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        output_path = FIGURES_DIR / f"person_e_congestion_map_{peak_type}_peak.png"

    speed_col = "morning_peak_speed" if peak_type == "morning" else "evening_peak_speed"
    title_label = "Morning Peak (07:00–10:00)" if peak_type == "morning" else "Evening Peak (16:00–19:00)"

    fig, ax = plt.subplots(figsize=(12, 9), dpi=300)
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    # Scatter plot with speed colormap: Red = Congested (<35), Yellow = Moderate (35-55), Green = Free Flow (>55)
    speeds = df_sensors[speed_col].fillna(df_sensors["valid_mean"])
    scatter = ax.scatter(
        df_sensors["longitude"],
        df_sensors["latitude"],
        c=speeds,
        cmap="RdYlGn",
        s=75,
        alpha=0.92,
        edgecolors="#334155",
        linewidths=0.8,
        vmin=25,
        vmax=65,
    )

    # Highlight major freeway bottlenecks
    severe_mask = speeds < 40.0
    if severe_mask.any():
        ax.scatter(
            df_sensors.loc[severe_mask, "longitude"],
            df_sensors.loc[severe_mask, "latitude"],
            facecolors="none",
            edgecolors="#ef4444",
            s=180,
            linewidths=2.0,
            label="Severe Bottleneck (<40 mph)",
        )

    # Formatting and annotations
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.032, pad=0.04)
    cbar.set_label("Mean Traffic Speed (mph, unverified)", color="#f8fafc", fontsize=11, fontweight="bold")
    cbar.ax.yaxis.set_tick_params(color="#94a3b8")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#e2e8f0")

    ax.set_title(
        f"METR-LA Road Network Congestion Map: {title_label}\n207 Highway Sensors across Los Angeles County",
        fontsize=13,
        color="#f8fafc",
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("Longitude (West to East)", color="#cbd5e1", fontsize=11, labelpad=8)
    ax.set_ylabel("Latitude (South to North)", color="#cbd5e1", fontsize=11, labelpad=8)
    ax.tick_params(colors="#94a3b8", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#475569")
    ax.grid(True, linestyle="--", alpha=0.25, color="#64748b")

    # Add reference corridor annotations
    ax.text(-118.49, 34.18, "I-405 Corridor\n(Sepulveda Pass)", color="#67e8f9", fontsize=9, fontweight="bold")
    ax.text(-118.25, 34.05, "Downtown LA\n(I-10 / I-110 Hub)", color="#fbcfe8", fontsize=9, fontweight="bold")
    ax.text(-118.36, 34.14, "US-101 (Cahuenga Pass)", color="#fde047", fontsize=9, fontweight="bold")
    ax.text(-118.22, 34.22, "I-210 Foothill", color="#86efac", fontsize=9, fontweight="bold")

    ax.legend(loc="lower right", facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    return output_path


def plot_corridor_error_distribution(
    df_sensors: pd.DataFrame,
    output_path: Path | None = None,
) -> Path:
    """Plot speed variability and expected forecast difficulty across freeway corridors."""
    if output_path is None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        output_path = FIGURES_DIR / "person_e_spatial_error_corridors.png"

    corridor_stats = compute_corridor_statistics(df_sensors)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    fig.patch.set_facecolor("#0f172a")
    ax1.set_facecolor("#1e293b")
    ax2.set_facecolor("#1e293b")

    # Left plot: Morning vs Evening Speed Drop %
    y_pos = np.arange(len(corridor_stats))
    bar_width = 0.38

    b1 = ax1.barh(y_pos - bar_width/2, corridor_stats["morning_speed_drop_pct"], bar_width, label="Morning Peak Drop %", color="#38bdf8", alpha=0.9)
    b2 = ax1.barh(y_pos + bar_width/2, corridor_stats["evening_speed_drop_pct"], bar_width, label="Evening Peak Drop %", color="#f43f5e", alpha=0.9)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(corridor_stats["corridor"], color="#f1f5f9", fontsize=10)
    ax1.set_xlabel("Speed Drop vs Baseline Free Flow (%)", color="#cbd5e1", fontsize=10)
    ax1.set_title("Peak Congestion Severity by Highway Corridor", color="#f8fafc", fontsize=12, fontweight="bold")
    ax1.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")
    ax1.grid(True, linestyle="--", alpha=0.25, color="#64748b")
    ax1.tick_params(colors="#94a3b8")

    # Right plot: Standard deviation (volatility) by corridor
    corridor_order = corridor_stats["corridor"].tolist()
    sensor_vols = [df_sensors.loc[df_sensors["corridor"] == c, "valid_std"].values for c in corridor_order]

    bp = ax2.boxplot(sensor_vols, vert=False, patch_artist=True, tick_labels=corridor_order)
    for box in bp["boxes"]:
        box.set_facecolor("#818cf8")
        box.set_edgecolor("#c7d2fe")
    for whisker in bp["whiskers"]:
        whisker.set_color("#cbd5e1")
    for cap in bp["caps"]:
        cap.set_color("#cbd5e1")
    for median in bp["medians"]:
        median.set_color("#facc15")
        median.set_linewidth(2)

    ax2.set_title(r"Sensor Speed Volatility ($\sigma_{speed}$) by Corridor", color="#f8fafc", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Standard Deviation of Valid Speeds (mph)", color="#cbd5e1", fontsize=10)
    ax2.set_yticklabels([])  # Shared with left plot
    ax2.grid(True, linestyle="--", alpha=0.25, color="#64748b")
    ax2.tick_params(colors="#94a3b8")

    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_color("#475569")

    plt.suptitle("Spatial Analysis of Highway Corridors: Volatility & Peak Vulnerability", color="#f8fafc", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    return output_path
