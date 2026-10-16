"""Builds the shared train/val/test parquets for Persons C and D (Person B).

Run from the project root:

    python -m src.data.dataset_builder

Writes data/processed/{train,val,test}.parquet and dataset_summary.json.
One row = one (origin timestamp, sensor). History length and horizons come
from Person A's split_contract.json so everyone uses the same setup.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .features import (
    ALL_FEATURE_COLUMNS,
    LAG_COLUMNS,
    MISSINGNESS_COLUMNS,
    ROLLING_COLUMNS,
    TARGET_COLUMNS,
    TARGET_HORIZONS,
    TARGET_VALID_COLUMNS,
    TEMPORAL_COLUMNS,
    TREND_COLUMNS,
    compute_lag_features,
    compute_missingness_features,
    compute_rolling_features,
    compute_temporal_features,
    compute_trend_features,
    extract_targets,
)
from .load_data import (
    apply_causal_imputation,
    check_sensor_order,
    create_missingness_mask,
    fit_training_sensor_medians,
    load_raw_speed_data,
    load_sensor_metadata,
)
from .split import get_eligible_sample_origins, get_partition_info, load_split_contract

ROOT = Path(__file__).resolve().parents[2]

PARQUET_NAMES = {"train": "train.parquet", "validation": "val.parquet", "test": "test.parquet"}

FEATURE_GROUPS = {
    "lag_only": LAG_COLUMNS,
    "temporal": TEMPORAL_COLUMNS,
    "rolling": ROLLING_COLUMNS,
    "trend": TREND_COLUMNS,
    "missingness": MISSINGNESS_COLUMNS,
    # Person D ablations A-D
    "lag_temporal": LAG_COLUMNS + TEMPORAL_COLUMNS,
    "lag_temporal_rolling": LAG_COLUMNS + TEMPORAL_COLUMNS + ROLLING_COLUMNS,
    "full_features": ALL_FEATURE_COLUMNS,
}


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def assemble_partition_table(
    origins: np.ndarray,
    timestamps: pd.DatetimeIndex,
    sensor_ids: list[str],
    feature_frames: dict[str, pd.DataFrame],
    temporal_df: pd.DataFrame,
    target_values: dict[str, pd.DataFrame],
    target_valids: dict[str, pd.DataFrame],
) -> pa.Table:
    """Turn the (time x sensor) frames into a long table, row order = origin then sensor."""
    n_sensors = len(sensor_ids)
    if len(origins) == 0:
        raise ValueError("no origins for this partition")

    def per_sensor(frame: pd.DataFrame, dtype) -> pa.Array:
        return pa.array(frame.to_numpy()[origins, :].ravel().astype(dtype))

    codes = np.tile(np.arange(n_sensors, dtype=np.int16), len(origins))
    cols: dict[str, pa.Array] = {
        "origin_timestamp": pa.array(np.repeat(timestamps[origins].to_numpy(), n_sensors)),
        "sensor_id": pa.DictionaryArray.from_arrays(pa.array(codes), pa.array(sensor_ids, type=pa.string())),
    }

    for name in LAG_COLUMNS + ROLLING_COLUMNS + TREND_COLUMNS:
        cols[name] = per_sensor(feature_frames[name], np.float32)

    # time features are the same for every sensor at a given origin
    t = temporal_df.iloc[origins]
    for name in TEMPORAL_COLUMNS:
        dtype = np.int8 if name.startswith("is_") else np.float32
        cols[name] = pa.array(np.repeat(t[name].to_numpy(), n_sensors).astype(dtype))

    for name in MISSINGNESS_COLUMNS:
        cols[name] = per_sensor(feature_frames[name], np.int8)

    for name in TARGET_COLUMNS:
        cols[name] = per_sensor(target_values[name], np.float32)
    for name in TARGET_VALID_COLUMNS:
        cols[name] = per_sensor(target_valids[name], np.bool_)

    return pa.Table.from_pydict(cols)


def build_dataset(
    raw_path: Path | str = ROOT / "data/raw/metr-la.h5",
    contract_path: Path | str = ROOT / "data/metadata/split_contract.json",
    sensors_path: Path | str = ROOT / "data/metadata/sensors.csv",
    output_dir: Path | str = ROOT / "data/processed",
    save_parquet: bool = True,
    compression: str = "snappy",
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_speed_data(raw_path)
    check_sensor_order(raw_df, load_sensor_metadata(sensors_path))
    contract = load_split_contract(contract_path)

    history = int(contract["history_steps"])
    horizons = [int(h) for h in contract["horizon_steps"]]
    if history != len(LAG_COLUMNS) or horizons != list(TARGET_HORIZONS):
        raise ValueError(
            f"contract wants history={history}, horizons={horizons} but features.py is set up "
            f"for {len(LAG_COLUMNS)} lags and horizons {list(TARGET_HORIZONS)}"
        )

    timestamps = raw_df.index
    sensor_ids = list(raw_df.columns)
    partitions = get_partition_info(timestamps, contract=contract)

    # missing mask + imputation (medians from train only)
    mask = create_missingness_mask(raw_df)
    medians = fit_training_sensor_medians(raw_df, mask, partitions["train"].end_inclusive)
    speed = apply_causal_imputation(raw_df, mask, medians)

    feature_frames = {
        **compute_lag_features(speed, lags=range(history)),
        **compute_rolling_features(speed),
        **compute_trend_features(speed),
        **compute_missingness_features(mask, history_window=history),
    }
    temporal_df = compute_temporal_features(timestamps)
    # targets from the raw table, never the imputed one
    target_values, target_valids = extract_targets(raw_df)

    summary: dict[str, Any] = {
        "dataset": "METR-LA",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "history_steps": history,
        "horizon_steps": horizons,
        "horizon_names": TARGET_HORIZONS,
        "total_grid_timestamps": len(timestamps),
        "absent_timestamps": raw_df.attrs.get("absent_timestamps", 0),
        "sensor_count": len(sensor_ids),
        "sensor_order": sensor_ids,
        "feature_columns": ALL_FEATURE_COLUMNS,
        "feature_groups": FEATURE_GROUPS,
        "target_columns": TARGET_COLUMNS,
        "target_valid_columns": TARGET_VALID_COLUMNS,
        "preprocessing": {
            "missing_rule": "NaN, inf, exact zero and negative speeds are missing",
            "input_imputation": "forward fill, then per-sensor training median for leading gaps",
            "targets_imputed": False,
            "train_median_fit_end": str(partitions["train"].end_inclusive),
            "training_medians": {s: round(float(v), 6) for s, v in medians.items()},
        },
        "partitions": {},
    }

    for name, filename in PARQUET_NAMES.items():
        origins, origin_times = get_eligible_sample_origins(
            timestamps, name, history_steps=history, horizon_steps=horizons, contract=contract
        )
        table = assemble_partition_table(
            origins, timestamps, sensor_ids, feature_frames, temporal_df, target_values, target_valids
        )

        target_stats = {}
        for h in TARGET_HORIZONS.values():
            n_valid = pc.sum(table[f"target_valid_{h}"].cast(pa.int64())).as_py()
            target_stats[h] = {
                "valid_targets": n_valid,
                "total_rows": table.num_rows,
                "valid_fraction": n_valid / table.num_rows,
            }

        info = {
            "eligible_origins": len(origins),
            "total_rows": table.num_rows,
            "first_origin": str(origin_times[0]),
            "last_origin": str(origin_times[-1]),
            "target_statistics": target_stats,
        }
        if save_parquet:
            out = output_dir / filename
            pq.write_table(table, out, compression=compression)
            info["file_path"] = out.relative_to(ROOT).as_posix() if out.is_relative_to(ROOT) else str(out)
            info["file_bytes"] = out.stat().st_size
            info["file_sha256"] = compute_file_sha256(out)
        summary["partitions"][name] = info

    summary_path = output_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    summary["summary_path"] = str(summary_path)
    return summary


def load_partition(
    partition: str = "train",
    processed_dir: Path | str = ROOT / "data/processed",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load one split. Accepts "train", "val"/"validation" or "test"."""
    key = "validation" if partition == "val" else partition
    if key not in PARQUET_NAMES:
        raise ValueError(f"unknown partition {partition!r}")
    path = Path(processed_dir) / PARQUET_NAMES[key]
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found, run `python -m src.data.dataset_builder` first")
    return pd.read_parquet(path, columns=columns)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Person B processed datasets.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--compression", default="snappy")
    args = parser.parse_args()

    summary = build_dataset(output_dir=args.output_dir, compression=args.compression)
    for name, p in summary["partitions"].items():
        print(f"{name:<10} origins={p['eligible_origins']:,} rows={p['total_rows']:,} "
              f"{p['first_origin']} -> {p['last_origin']}")
        if "file_path" in p:
            print(f"{'':<10} {p['file_path']} ({p['file_bytes'] / 2**20:.1f} MB) sha256={p['file_sha256'][:12]}...")
    print(f"summary written to {summary['summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
