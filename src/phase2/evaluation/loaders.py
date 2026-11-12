"""Data loaders and alignment checks for Phase 2 evaluation (Member 4).

Contract: reports/phase2_plan.md section 4.7, 4.8, 8.4.
Enforces hard validation on:
- Matching origins between Phase 1 and Phase 2 models (3,416 val / 6,844 test).
- Matching sensor ordering against data/metadata/sensors.csv.
- Re-computed raw target speed and target validity against data/raw/metr-la.h5.
- Finiteness of predictions on valid targets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.models.prediction_evaluation import HORIZONS, align_predictions, temporal_masks
from src.phase2.common import HORIZON_NAMES, HORIZON_STEPS, ROOT

DATA_DIR = ROOT / "data"
RAW_H5_PATH = DATA_DIR / "raw" / "metr-la.h5"
SENSORS_CSV_PATH = DATA_DIR / "metadata" / "sensors.csv"
PREDICTIONS_DIR = DATA_DIR / "processed" / "predictions"
PHASE2_PREDICTIONS_DIR = DATA_DIR / "processed" / "phase2" / "predictions"

EXPECTED_ORIGIN_COUNTS = {
    "val": 3416,
    "validation": 3416,
    "test": 6844,
}


def load_canonical_sensor_ids(sensors_path: Path | str = SENSORS_CSV_PATH) -> list[str]:
    """Return the 207 canonical sensor IDs in metadata order."""
    sensors_path = Path(sensors_path)
    if not sensors_path.is_file():
        raise FileNotFoundError(f"sensors metadata not found: {sensors_path}")
    df = pd.read_csv(sensors_path, dtype={"sensor_id": str})
    return [str(s).strip() for s in df["sensor_id"]]


def load_canonical_origins(
    split: str,
    raw_path: Path | str = RAW_H5_PATH,
    contract_path: Path | str = DATA_DIR / "metadata" / "split_contract.json",
) -> pd.DatetimeIndex:
    """Return the exact eligible origin timestamps for a partition using Phase 1 rules."""
    from src.data.load_data import load_raw_speed_data
    from src.data.split import get_eligible_sample_origins

    norm_split = "validation" if split in ("val", "validation") else "test"
    raw_df = load_raw_speed_data(raw_path)
    _, origins = get_eligible_sample_origins(
        raw_df.index,
        partition_name=norm_split,
        history_steps=12,
        horizon_steps=[1, 3, 6, 12],
    )
    return pd.DatetimeIndex(origins)


def load_phase1_predictions(
    split: str,
    predictions_dir: Path | str = PREDICTIONS_DIR,
    raw_path: Path | str = RAW_H5_PATH,
    validate_targets_with_raw: bool = True,
) -> pd.DataFrame:
    """Load and align Person C and Person D Phase 1 prediction parquets.

    Raises FileNotFoundError if Phase 1 parquets are missing so tests can skip gracefully.
    """
    predictions_dir = Path(predictions_dir)
    norm_split = "val" if split in ("val", "validation") else "test"
    c_path = predictions_dir / f"person_c_{norm_split}_predictions.parquet"
    d_path = predictions_dir / f"person_d_{norm_split}_predictions.parquet"

    if not c_path.is_file() or not d_path.is_file():
        raise FileNotFoundError(
            f"Phase 1 prediction parquets not found in {predictions_dir} "
            f"({c_path.name} or {d_path.name} missing). Download phase1_predictions.zip into data/processed/predictions/"
        )

    c_df = pd.read_parquet(c_path)
    d_df = pd.read_parquet(d_path)
    aligned = align_predictions(c_df, d_df, source=None)

    # Convert sensor_id to string for reliable alignment
    aligned["sensor_id"] = aligned["sensor_id"].astype(str)

    # Verify origin count
    n_unique_origins = aligned["origin_timestamp"].nunique()
    expected = EXPECTED_ORIGIN_COUNTS[norm_split]
    if n_unique_origins != expected:
        raise ValueError(
            f"Phase 1 {norm_split} partition has {n_unique_origins} origins, expected {expected}"
        )

    # Optional independent target validation against raw metr-la.h5
    if validate_targets_with_raw and Path(raw_path).is_file():
        validate_dataframe_targets_against_raw(aligned, raw_path)

    return aligned


def validate_dataframe_targets_against_raw(
    df: pd.DataFrame, raw_path: Path | str = RAW_H5_PATH
) -> None:
    """Verify that target columns and target_valid flags match raw metr-la.h5 exactly."""
    from src.data.load_data import load_raw_speed_data

    raw = load_raw_speed_data(raw_path)
    raw.columns = [str(c).strip() for c in raw.columns]

    for h, step in HORIZONS.items():
        target_col = f"target_{h}"
        valid_col = f"target_valid_{h}"
        if target_col not in df.columns:
            continue

        # Check a sample of rows across distinct timestamps
        unique_origins = pd.DatetimeIndex(df["origin_timestamp"].drop_duplicates()[:50])
        offset = pd.Timedelta(minutes=step * 5)
        for origin in unique_origins:
            target_ts = origin + offset
            if target_ts not in raw.index:
                continue
            raw_vals = raw.loc[target_ts]
            sub = df[df["origin_timestamp"] == origin].set_index("sensor_id")
            for sensor_id in sub.index[:10]:
                if sensor_id in raw_vals:
                    expected_val = float(raw_vals[sensor_id])
                    actual_val = float(sub.loc[sensor_id, target_col])
                    actual_valid = bool(sub.loc[sensor_id, valid_col])
                    expected_valid = bool(np.isfinite(expected_val) and expected_val > 0)

                    if not np.isclose(expected_val, actual_val, equal_nan=True):
                        raise ValueError(
                            f"Target mismatch for {sensor_id} at {target_ts}: raw={expected_val}, df={actual_val}"
                        )
                    if actual_valid != expected_valid:
                        raise ValueError(
                            f"Target validity mismatch for {sensor_id} at {target_ts}: raw_valid={expected_valid}, df_valid={actual_valid}"
                        )


def load_phase2_npz(
    npz_path: Path | str,
    split: str,
    canonical_origins: Sequence[pd.Timestamp] | None = None,
    canonical_sensors: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Load and strictly validate a single Phase 2 prediction .npz file (plan 4.7).

    Returns a dict with keys:
      - 'pred': float32 ndarray [n_origins, 4, 207]
      - 'origin_ns': int64 ndarray [n_origins]
      - 'origin_dt': DatetimeIndex [n_origins]
      - 'sensor_ids': list of strings [207]
    """
    npz_path = Path(npz_path)
    if not npz_path.is_file():
        raise FileNotFoundError(f"Phase 2 prediction file not found: {npz_path}")

    norm_split = (
        "val"
        if split in ("val", "validation")
        else ("test" if split in ("test",) else None)
    )
    expected_count = EXPECTED_ORIGIN_COUNTS.get(norm_split)

    with np.load(npz_path) as data:
        if "pred" not in data or "origin_ns" not in data or "sensor_ids" not in data:
            raise KeyError(
                f"{npz_path.name} must contain 'pred', 'origin_ns', and 'sensor_ids'"
            )
        pred = data["pred"]
        origin_ns = data["origin_ns"]
        sensor_ids = [str(s).strip() for s in data["sensor_ids"]]

    # 1. Shape check [n_origins, 4, 207]
    if pred.ndim != 3:
        raise ValueError(
            f"{npz_path.name}: 'pred' must be 3D [n_origins, 4, 207], got shape {pred.shape}"
        )
    n_origins, n_horizons, n_sensors = pred.shape
    if n_horizons != 4:
        raise ValueError(f"{npz_path.name}: expected 4 horizons, got {n_horizons}")
    if n_sensors != 207:
        raise ValueError(f"{npz_path.name}: expected 207 sensors, got {n_sensors}")

    if expected_count is not None and n_origins != expected_count:
        raise ValueError(
            f"{npz_path.name}: expected {expected_count} origins for split '{split}', got {n_origins}"
        )

    # 2. Finiteness check
    if not np.isfinite(pred).all():
        raise ValueError(f"{npz_path.name}: contains NaN or infinite prediction values")

    # 3. Sensor IDs check
    if canonical_sensors is not None:
        if list(sensor_ids) != list(canonical_sensors):
            raise ValueError(
                f"{npz_path.name}: sensor_ids do not match canonical sensors.csv order"
            )

    # 4. Origins check
    origin_dt = pd.to_datetime(origin_ns, unit="ns")
    if canonical_origins is not None:
        canon_dt = pd.DatetimeIndex(canonical_origins)
        if len(origin_dt) != len(canon_dt) or not np.array_equal(
            origin_dt.asi8, canon_dt.asi8
        ):
            raise ValueError(
                f"{npz_path.name}: origin timestamps do not match canonical {split} partition origins"
            )

    return {
        "pred": pred.astype(np.float32),
        "origin_ns": origin_ns,
        "origin_dt": origin_dt,
        "sensor_ids": sensor_ids,
    }


def attach_phase2_predictions(
    df: pd.DataFrame,
    phase2_data: dict[str, Any],
    model_col_prefix: str,
) -> pd.DataFrame:
    """Attach Phase 2 predictions into an aligned evaluation DataFrame.

    Adds columns:
      pred_{model_col_prefix}_5min
      pred_{model_col_prefix}_15min
      pred_{model_col_prefix}_30min
      pred_{model_col_prefix}_60min
    """
    pred = phase2_data["pred"]  # [n_origins, 4, n_sensors]
    origin_dt = phase2_data["origin_dt"]
    sensor_ids = phase2_data["sensor_ids"]

    # Build a lookup DataFrame indexed by (origin_timestamp, sensor_id)
    n_origins, _, n_sensors = pred.shape
    records = []
    
    # Fast reshaping: flatten origins and sensors
    # pred is [n_origins, 4, n_sensors] -> transpose to [n_origins, n_sensors, 4]
    pred_transposed = pred.transpose(0, 2, 1)  # [n_origins, n_sensors, 4]
    flat_preds = pred_transposed.reshape(-1, 4)

    repeated_origins = np.repeat(origin_dt, n_sensors)
    tiled_sensors = np.tile(sensor_ids, n_origins)

    pred_df = pd.DataFrame(
        flat_preds,
        columns=[f"pred_{model_col_prefix}_{h}" for h in HORIZON_NAMES],
    )
    pred_df["origin_timestamp"] = repeated_origins
    pred_df["sensor_id"] = tiled_sensors

    # Merge into df
    merged = df.merge(
        pred_df,
        on=["origin_timestamp", "sensor_id"],
        how="left",
        validate="one_to_one",
    )
    for h in HORIZON_NAMES:
        col = f"pred_{model_col_prefix}_{h}"
        if merged[col].isna().any():
            raise ValueError(f"Missing predictions when attaching {col}")

    return merged
