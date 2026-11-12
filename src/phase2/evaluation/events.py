"""Congestion onset diagnostics and stratified evaluation slices for Phase 2 (Member 4).

Contract: reports/phase2_plan.md section 4.8, 8.4.

CONGESTION PROXY DEFINITIONS (Fixed ex-ante):
---------------------------------------------
1. Congestion State Proxy:
   A sensor at time t is in a proxy congested state if its observed (or predicted) speed
   is strictly below 60% of that sensor's historical training 85th percentile speed:
       congested = (speed < 0.6 * training_reference_speed)
   where training_reference_speed is derived from valid training partition cells.

2. Congestion Onset Event:
   A true congestion onset occurs at horizon h if:
   (a) The sensor is NOT congested at the forecast origin t (speed_origin >= 0.6 * ref), AND
   (b) The sensor transitions to CONGESTED at the target timestamp t+h (target_speed < 0.6 * ref),
       evaluated on valid, finite readings.

3. Availability Slice:
   A sample origin has 'normal' input availability if at least 11 of the 12 history readings
   in the window [t-11 ... t] are valid (>0). Origins with >= 2 missing readings in the 12-step
   window are categorized as 'low' availability.

Labels must explicitly state 'proxy congestion' and 'coordinate regions'.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.models.prediction_evaluation import HORIZONS, geographic_regions, region_scores, temporal_masks
from src.phase2.common import HORIZON_NAMES, ROOT

METADATA_DIR = ROOT / "data" / "metadata"


def load_sensor_reference_speeds(
    sensors_csv_path: Path | str = METADATA_DIR / "sensors.csv",
    raw_h5_path: Path | str = ROOT / "data" / "raw" / "metr-la.h5",
    contract_path: Path | str = METADATA_DIR / "split_contract.json",
) -> Dict[str, float]:
    """Calculate the 85th percentile training reference speed per sensor."""
    import json
    from src.data.load_data import load_raw_speed_data

    raw = load_raw_speed_data(raw_h5_path)
    contract = json.loads(Path(contract_path).read_text())["partitions"]["train"]
    start = pd.Timestamp(contract["start_inclusive"])
    end = pd.Timestamp(contract["end_inclusive"])

    train_data = raw.loc[(raw.index >= start) & (raw.index <= end)]
    train_valid = train_data.where(np.isfinite(train_data) & (train_data > 0))
    ref = train_valid.quantile(0.85)
    return {str(k).strip(): float(v) for k, v in ref.items()}


def compute_congestion_event_metrics(
    df: pd.DataFrame,
    models: Sequence[str],
    horizons: Sequence[str] = ("15min", "30min", "60min"),
    reference_speeds: Optional[Mapping[str, float]] = None,
) -> pd.DataFrame:
    """Compute precision, recall, and onset recall of proxy congestion.

    Outputs DataFrame for data/metadata/phase2/events.csv.
    """
    if reference_speeds is None:
        reference_speeds = load_sensor_reference_speeds()

    ref_array = df["sensor_id"].astype(str).map(reference_speeds).to_numpy(dtype=float)
    threshold = 0.6 * ref_array

    # Origin speed is speed_lag_0 if present, or speed at t
    if "speed_lag_0" in df.columns:
        origin_speed = df["speed_lag_0"].to_numpy(dtype=float)
    else:
        origin_speed = df["target_5min"].to_numpy(dtype=float)

    origin_not_congested = origin_speed >= threshold
    records = []

    for h in horizons:
        valid = df[f"target_valid_{h}"].to_numpy(dtype=bool)
        y = df[f"target_{h}"].to_numpy(dtype=float)
        y_congested = (y < threshold) & valid
        true_onset = origin_not_congested & y_congested

        for model in models:
            pred_col = f"pred_{model}_{h}"
            if pred_col not in df.columns:
                continue

            p = df[pred_col].to_numpy(dtype=float)
            p_congested = (p < threshold) & valid
            pred_onset = origin_not_congested & p_congested

            # Congestion detection precision & recall
            tp_cong = np.sum(p_congested & y_congested)
            fp_cong = np.sum(p_congested & ~y_congested & valid)
            fn_cong = np.sum(~p_congested & y_congested)

            prec_cong = float(tp_cong / (tp_cong + fp_cong)) if (tp_cong + fp_cong) > 0 else np.nan
            rec_cong = float(tp_cong / (tp_cong + fn_cong)) if (tp_cong + fn_cong) > 0 else np.nan

            # Congestion onset recall
            tp_onset = np.sum(pred_onset & true_onset)
            fn_onset = np.sum(~pred_onset & true_onset)
            rec_onset = float(tp_onset / (tp_onset + fn_onset)) if (tp_onset + fn_onset) > 0 else np.nan

            records.append({
                "model": model,
                "horizon": h,
                "n_valid_targets": int(valid.sum()),
                "actual_congested_count": int(y_congested.sum()),
                "predicted_congested_count": int(p_congested.sum()),
                "precision_congestion": prec_cong,
                "recall_congestion": rec_cong,
                "actual_onset_count": int(true_onset.sum()),
                "predicted_onset_count": int(pred_onset.sum()),
                "recall_congestion_onset": rec_onset,
                "definition": "proxy speed < 0.6 * training 85th percentile",
            })

    return pd.DataFrame(records)


def compute_stratified_slices(
    df: pd.DataFrame,
    sensor_scores_df: pd.DataFrame,
    sensor_metadata_path: Path | str = METADATA_DIR / "sensors.csv",
    adj_path: Path | str = ROOT / "data" / "raw" / "adj_mx.pkl",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute sensor_metrics.csv and region_metrics.csv stratified by regions and degree."""
    import pickle
    from src.models.prediction_evaluation import region_scores

    sensors_meta = pd.read_csv(sensor_metadata_path, dtype={"sensor_id": str})
    sensors_meta["sensor_id"] = sensors_meta["sensor_id"].astype(str)

    # Calculate graph degree from adj_mx.pkl if available
    if Path(adj_path).is_file():
        try:
            with open(adj_path, "rb") as f:
                _, _, adj = pickle.load(f, encoding="latin1")
            # Adjacency has self-loops (diagonal = 1), subtract self loop for degree
            degrees = np.sum(adj > 0, axis=1) - 1
            sensors_meta["graph_degree"] = degrees
        except Exception:
            sensors_meta["graph_degree"] = np.nan
    else:
        sensors_meta["graph_degree"] = np.nan

    # Add geographic regions
    sensors_meta = geographic_regions(sensors_meta)

    # Merge degree into sensor_scores
    merged_sensor_scores = sensor_scores_df.merge(
        sensors_meta[["sensor_id", "region", "graph_degree"]],
        on="sensor_id",
        how="left",
    )

    # Compute region aggregates using Phase 1 region_scores logic
    reg_scores = region_scores(sensor_scores_df, sensors_meta)

    return merged_sensor_scores, reg_scores

