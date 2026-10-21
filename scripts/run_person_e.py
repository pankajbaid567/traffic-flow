#!/usr/bin/env python3
"""Recompute Phase 1 evaluation from aligned saved predictions.

Run after dataset_builder, run_person_c --predict-only, and predict_person_d.
Produces traceable benchmark, temporal, sensor, geographic, bootstrap evidence,
figures and reports. Missing forecasts are a hard error, never invented values.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.data.dataset_builder import load_partition
from src.models.prediction_evaluation import (
    HORIZONS,
    MODELS,
    align_predictions,
    score_frame,
    region_scores,
    paired_daily_bootstrap,
    geographic_regions,
)
from src.visualization.spatial_mapping import (
    load_sensor_geo_metadata,
    compute_corridor_statistics,
    plot_spatial_congestion_map,
    plot_corridor_error_distribution,
    plot_prediction_map,
    plot_congestion_prediction_map,
)
from src.models.evaluation import build_rf_ablation_summary, compute_horizon_degradation
from src.visualization.benchmark_plots import (
    plot_horizon_comparison,
    plot_peak_vs_offpeak,
    plot_feature_importance_comparison,
    plot_rf_ablation_progression,
    plot_graph_gap_diagram,
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def peak_wide(metrics, split="test"):
    q = metrics[metrics.split == split]
    records = []
    for (h, period), g in q.groupby(["horizon", "period"], sort=False):
        if len(set(g.n)) != 1:
            raise ValueError("Models have unequal valid-target counts")
        row = {"horizon": h, "period": period, "rows": int(g.n.iloc[0])}
        for _, r in g.iterrows():
            # Preserve historical plotting names while adding RF.
            m = {"xgboost_32": "xgboost", "random_forest_32": "random_forest"}.get(
                r.model, r.model
            )
            for metric in ["mae", "rmse", "mape"]:
                row[f"{m}_{metric}"] = r[metric]
        records.append(row)
    return pd.DataFrame(records)


def compare_previous(metrics, c_data, d_data, tolerance=0.002):
    checks = []
    for split in ["val", "test"]:
        for h in HORIZONS:
            q = metrics[
                (metrics.split == split)
                & (metrics.horizon == h)
                & (metrics.period == "all")
            ]
            for _, r in q.iterrows():
                old = (
                    d_data["experiments"]["D"][h][f"{split}_metrics"]
                    if r.model == "random_forest_32"
                    else c_data["horizons"][h][split][
                        {"xgboost_32": "xgboost"}.get(r.model, r.model)
                    ]["all"]
                )
                delta = max(abs(r[k] - old[k]) for k in ["mae", "rmse", "mape"])
                checks.append(
                    {
                        "split": split,
                        "horizon": h,
                        "model": r.model,
                        "maximum_metric_difference": float(delta),
                    }
                )
                if delta > tolerance:
                    raise ValueError(
                        f"Recomputed metrics differ from model-run evidence: {checks[-1]}"
                    )
    return checks


def plot_example(df, sensor_scores):
    """Retrospective error example, explicitly not selected for model tuning."""
    import matplotlib.pyplot as plt

    q = sensor_scores[
        (sensor_scores.split == "test")
        & (sensor_scores.horizon == "60min")
        & (sensor_scores.model == "xgboost_ha")
        & (sensor_scores.period == "all")
    ]
    sensor = (
        q.sort_values(["mae", "sensor_id"], ascending=[False, True]).iloc[0].sensor_id
    )
    sub = df[df.sensor_id.astype(str) == sensor].copy()
    mask = sub.target_valid_60min.to_numpy(bool)
    # Date of largest observed absolute error; diagnostic selection is disclosed.
    errors = np.where(mask, np.abs(sub.target_60min - sub.pred_xgboost_ha_60min), -1)
    center = pd.Timestamp(
        sub.iloc[int(np.argmax(errors))].origin_timestamp
    ) + pd.Timedelta(hours=1)
    target_time = pd.DatetimeIndex(sub.origin_timestamp) + pd.Timedelta(hours=1)
    keep = (target_time >= center - pd.Timedelta(hours=6)) & (
        target_time <= center + pd.Timedelta(hours=6)
    )
    fig, ax = plt.subplots(figsize=(8, 3), constrained_layout=True)
    ax.plot(
        target_time[keep],
        sub.target_60min.where(sub.target_valid_60min).to_numpy()[keep],
        label="Observed",
        color="black",
        lw=1.5,
    )
    for model, prefix in [
        ("Persistence", "persistence"),
        ("Random Forest", "random_forest"),
        ("XGBoost + HA", "xgboost_ha"),
    ]:
        ax.plot(
            target_time[keep],
            sub[f"pred_{prefix}_60min"].to_numpy()[keep],
            label=model,
            lw=1,
        )
    ax.set(
        xlabel="Target timestamp on stored clock",
        ylabel="Speed (source units)",
        title=f"Sensor {sensor}: retrospective 60-minute forecast example",
    )
    ax.legend(ncol=4, fontsize=8)
    ax.grid(alpha=0.2)
    path = ROOT / "figures/person_e_prediction_example.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {
        "sensor_id": sensor,
        "center_target_time": str(center),
        "selection": "largest-error event at highest-MAE XGBoost+HA sensor; retrospective test diagnostic only",
    }


def run_person_e_pipeline(save_figures=True):
    metadata_dir = ROOT / "data/metadata"
    pred_dir = ROOT / "data/processed/predictions"
    c_data = json.loads((metadata_dir / "person_c_results.json").read_text())
    d_data = json.loads((metadata_dir / "person_d_results.json").read_text())
    geo = load_sensor_geo_metadata()
    reference = geo.set_index("sensor_id").training_reference_speed.to_dict()
    metrics = []
    sensors = []
    daily = []
    inputs = []
    test_frame = None
    for split in ["val", "test"]:
        cpath = pred_dir / f"person_c_{split}_predictions.parquet"
        dpath = pred_dir / f"person_d_{split}_predictions.parquet"
        c = pd.read_parquet(cpath)
        d = pd.read_parquet(dpath)
        columns = ["origin_timestamp", "sensor_id"] + [
            f"{p}_{h}" for h in HORIZONS for p in ["target", "target_valid"]
        ]
        source = load_partition(split, columns=columns)
        aligned = align_predictions(c, d, source)
        m, s, a = score_frame(aligned, split, reference)
        metrics.append(m)
        sensors.append(s)
        daily.append(a)
        if split == "test":
            test_frame = aligned
        inputs.extend([cpath, dpath])
        print(
            "Independently scored",
            split,
            len(aligned),
            "rows across",
            len(MODELS),
            "models",
            flush=True,
        )
    m = pd.concat(metrics, ignore_index=True)
    s = pd.concat(sensors, ignore_index=True)
    a = pd.concat(daily, ignore_index=True)
    checks = compare_previous(m, c_data, d_data)
    reg = region_scores(s, geo)
    intervals = []
    for h in HORIZONS:
        for baseline, candidate in [
            ("persistence", "xgboost_ha"),
            ("random_forest_32", "xgboost_32"),
            ("xgboost_32", "xgboost_ha"),
        ]:
            intervals.append(paired_daily_bootstrap(a, baseline, candidate, h))
    ci = pd.DataFrame(intervals)
    bench = (
        m[(m.split == "test") & (m.period == "all")]
        .copy()
        .rename(columns={"n": "n_samples"})
    )
    slices = peak_wide(m)
    abl = build_rf_ablation_summary(d_data)
    corridors = compute_corridor_statistics(geo)
    # Actual forecast error rankings, not speed volatility.
    ranks = s[(s.split == "test") & (s.period == "all")].merge(
        geo[["sensor_id", "latitude", "longitude", "region"]],
        on="sensor_id",
        validate="many_to_one",
    )
    ranks = ranks.sort_values(
        ["horizon", "model", "mae"], ascending=[True, True, False]
    )
    for name, table in [
        ("phase1_metrics", m),
        ("sensor_forecast_metrics", s),
        ("daily_forecast_errors", a),
        ("region_forecast_metrics", reg),
        ("paired_bootstrap", ci),
        ("corridor_performance", corridors),
        ("sensor_error_rankings", ranks),
    ]:
        table.to_csv(metadata_dir / f"{name}.csv", index=False)
    geo.to_csv(metadata_dir / "sensor_spatial_summary.csv", index=False)
    example = plot_example(test_frame, s) if save_figures else None
    provenance_paths = inputs + [
        ROOT / "data/raw/manifest.json",
        ROOT / "data/metadata/split_contract.json",
        ROOT / "data/processed/dataset_summary.json",
    ]
    provenance_paths += sorted((ROOT / "models/person_c").glob("*.json")) + sorted(
        (ROOT / "models/person_d").glob("*.joblib")
    )
    provenance_paths += sorted((ROOT / "src").rglob("*.py")) + sorted(
        (ROOT / "scripts").glob("*.py")
    )
    package_names = [
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "pyarrow",
        "tables",
        "matplotlib",
    ]
    versions = {}
    for name in package_names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": "independent aligned prediction-row evaluation",
            "packages": versions,
            "speed_units": "source units; physical units unverified",
            "timezone": "stored clock; timezone unverified",
            "peak_definition": "weekday origin hours [07,10) and [16,19)",
            "mape_denominator_floor": 1e-5,
            "congestion_proxy": "speed < 0.6 * sensor training 85th percentile",
            "spatial_grouping": "coordinate regions; no verified freeway IDs or direction",
            "bootstrap": "2000 paired circular moving-block resamples of two stored days, seed 42",
            "inference_scope": "frozen models; no test-based tuning during completion",
        },
        "unified_test_benchmark": bench.to_dict("records"),
        "temporal_slices": slices.to_dict("records"),
        "rf_ablations": abl.to_dict("records"),
        "horizon_degradation": compute_horizon_degradation(bench).to_dict("records"),
        "corridor_performance": corridors.to_dict("records"),
        "paired_comparisons": intervals,
        "metric_reproduction_checks": checks,
        "example": example,
        "input_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in provenance_paths},
    }
    (metadata_dir / "person_e_results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    if save_figures:
        for call in [
            lambda: plot_horizon_comparison(bench),
            lambda: plot_peak_vs_offpeak(slices),
            lambda: plot_rf_ablation_progression(d_data),
            lambda: plot_feature_importance_comparison(c_data, d_data),
            lambda: plot_graph_gap_diagram(),
            lambda: plot_spatial_congestion_map(geo, "morning"),
            lambda: plot_spatial_congestion_map(geo, "evening"),
            lambda: plot_corridor_error_distribution(geo),
        ]:
            call()
        for model in ["random_forest_32", "xgboost_ha"]:
            for period in ["morning_peak", "evening_peak"]:
                plot_prediction_map(s, geo, model=model, period=period)
        plot_congestion_prediction_map(s, geo)
    from src.reporting.phase1_reports import write_reports

    write_reports(ROOT, result, m, s, reg, ci, corridors)
    print(
        "Saved independently recomputed evaluation, spatial errors, uncertainty and reports",
        flush=True,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plots", action="store_true")
    run_person_e_pipeline(not parser.parse_args().no_plots)
