"""Observed and predicted traffic maps with explicit geographic limitations.

Only coordinates are verified. Regions are reproducible coordinate bins; no
freeway membership, direction, or graph connectivity is inferred from them.
"""

from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.models.prediction_evaluation import geographic_regions, temporal_masks

ROOT = Path(__file__).resolve().parents[2]
METADATA_DIR = ROOT / "data/metadata"
FIGURES_DIR = ROOT / "figures"


def assign_freeway_corridors(df):
    """Compatibility name: return geographic regions, never unverified freeway IDs."""
    out = geographic_regions(df)
    out["corridor"] = out.region
    return out


def load_sensor_geo_metadata():
    sensors = pd.read_csv(METADATA_DIR / "sensors.csv", dtype={"sensor_id": str})
    raw = pd.read_hdf(ROOT / "data/raw/metr-la.h5")
    raw.columns = raw.columns.astype(str)
    valid = raw.where(np.isfinite(raw) & (raw > 0))
    contract = json.loads((METADATA_DIR / "split_contract.json").read_text())[
        "partitions"
    ]
    train = valid.loc[
        (valid.index >= pd.Timestamp(contract["train"]["start_inclusive"]))
        & (valid.index <= pd.Timestamp(contract["train"]["end_inclusive"]))
    ]
    test = valid.loc[
        (valid.index >= pd.Timestamp(contract["test"]["start_inclusive"]))
        & (valid.index <= pd.Timestamp(contract["test"]["end_inclusive"]))
    ]
    reference = train.quantile(0.85)
    sensors["training_reference_speed"] = sensors.sensor_id.map(reference)
    masks = temporal_masks(test.index)
    for period, name in [("morning_peak", "morning"), ("evening_peak", "evening")]:
        values = test.loc[masks[period]]
        sensors[f"{name}_peak_speed"] = sensors.sensor_id.map(values.mean())
        sensors[f"{name}_count"] = sensors.sensor_id.map(values.count())
        sensors[f"{name}_sum"] = sensors.sensor_id.map(values.sum())
    return assign_freeway_corridors(sensors)


def compute_corridor_statistics(df_sensors):
    records = []
    for region, g in df_sensors.groupby("region"):
        row = {
            "region": region,
            "sensor_count": len(g),
            "training_reference_speed": float(g.training_reference_speed.mean()),
            "grouping": "coordinate proxy; freeway membership unverified",
            "peak_time_basis": "weekday raw observation timestamp on stored clock",
        }
        for name in ["morning", "evening"]:
            n = int(g[f"{name}_count"].sum())
            mean = float(g[f"{name}_sum"].sum() / n) if n else np.nan
            # Reference averaged with the same sensor observation weights.
            ref = (
                float(
                    np.average(g.training_reference_speed, weights=g[f"{name}_count"])
                )
                if n
                else np.nan
            )
            row[f"{name}_valid_targets"] = n
            row[f"mean_{name}_speed"] = mean
            row[f"{name}_reference_speed"] = ref
            row[f"{name}_speed_drop_pct"] = float(100 * (ref - mean) / ref)
        records.append(row)
    return (
        pd.DataFrame(records)
        .sort_values("evening_speed_drop_pct", ascending=False)
        .reset_index(drop=True)
    )


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_spatial_congestion_map(df_sensors, peak_type="morning", output_path=None):
    fig, ax = plt.subplots(figsize=(6, 4.7), constrained_layout=True)
    values = df_sensors[f"{peak_type}_peak_speed"]
    scatter = ax.scatter(
        df_sensors.longitude,
        df_sensors.latitude,
        c=values,
        cmap="RdYlGn",
        s=15,
        vmin=15,
        vmax=70,
    )
    fig.colorbar(scatter, ax=ax, label="Observed speed (source units)")
    ax.set(
        xlabel="Longitude",
        ylabel="Latitude",
        title=f"Observed weekday {peak_type} peak on test interval",
    )
    return _save(
        fig,
        output_path or FIGURES_DIR / f"person_e_congestion_map_{peak_type}_peak.png",
    )


def plot_corridor_error_distribution(df_sensors, output_path=None):
    stats = compute_corridor_statistics(df_sensors)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    x = np.arange(len(stats))
    ax.barh(x - 0.18, stats.morning_speed_drop_pct, 0.35, label="Morning")
    ax.barh(x + 0.18, stats.evening_speed_drop_pct, 0.35, label="Evening")
    ax.set_yticks(x, stats.region)
    ax.set(
        xlabel="Drop from training 85th-percentile reference (%)",
        title="Observed test peak speeds by geographic region",
    )
    ax.legend()
    ax.grid(axis="x", alpha=0.2)
    return _save(
        fig, output_path or FIGURES_DIR / "person_e_spatial_error_corridors.png"
    )


def plot_prediction_map(
    sensor_scores,
    metadata,
    model="xgboost_ha",
    horizon="15min",
    period="evening_peak",
    output_path=None,
):
    q = sensor_scores[
        (sensor_scores.split == "test")
        & (sensor_scores.model == model)
        & (sensor_scores.horizon == horizon)
        & (sensor_scores.period == period)
    ]
    q = q.merge(
        metadata[["sensor_id", "latitude", "longitude"]],
        on="sensor_id",
        validate="one_to_one",
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), constrained_layout=True)
    for ax, col, title, cmap, lo, hi in zip(
        axes,
        ["actual_mean", "predicted_mean", "mae"],
        ["Observed mean", "Predicted mean", "Forecast MAE"],
        ["RdYlGn", "RdYlGn", "magma_r"],
        [10, 10, 0],
        [70, 70, 12],
    ):
        scatter = ax.scatter(
            q.longitude,
            q.latitude,
            c=q[col],
            cmap=cmap,
            s=12,
            vmin=lo,
            vmax=hi,
            rasterized=True,
        )
        ax.set(title=title, xlabel="Longitude")
        ax.tick_params(labelsize=8)
        fig.colorbar(scatter, ax=ax, shrink=0.8, label="Source speed units")
    axes[0].set_ylabel("Latitude")
    fig.suptitle(
        f'{model} | {horizon} | {period.replace("_"," ")} | valid test targets',
        fontsize=11,
    )
    return _save(
        fig, output_path or FIGURES_DIR / f"person_e_forecast_map_{model}_{period}.png"
    )


def plot_congestion_prediction_map(
    sensor_scores,
    metadata,
    model="xgboost_ha",
    horizon="15min",
    period="evening_peak",
    output_path=None,
):
    q = sensor_scores[
        (sensor_scores.split == "test")
        & (sensor_scores.model == model)
        & (sensor_scores.horizon == horizon)
        & (sensor_scores.period == period)
    ]
    q = q.merge(
        metadata[["sensor_id", "latitude", "longitude"]],
        on="sensor_id",
        validate="one_to_one",
    )
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), constrained_layout=True)
    for ax, col, title in zip(
        axes,
        ["actual_congestion_rate", "predicted_congestion_rate"],
        ["Observed congestion frequency", "Predicted congestion frequency"],
    ):
        scatter = ax.scatter(
            q.longitude,
            q.latitude,
            c=q[col],
            cmap="YlOrRd",
            s=14,
            vmin=0,
            vmax=1,
            rasterized=True,
        )
        ax.set(title=title, xlabel="Longitude")
        fig.colorbar(scatter, ax=ax, shrink=0.8)
    axes[0].set_ylabel("Latitude")
    fig.suptitle(
        "Congestion proxy: speed < 60% of sensor training 85th percentile", fontsize=10
    )
    return _save(
        fig, output_path or FIGURES_DIR / "person_e_predicted_vs_actual_congestion.png"
    )
