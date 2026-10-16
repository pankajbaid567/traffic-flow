"""Non-destructive METR-LA quality checks.

Zeros are suspect observations, not evidence of stopped traffic. The returned
``valid_speed`` masks zeros, nonfinite values and negative speeds. It is an
analysis view, not an imputed or resampled canonical dataset. All run and change
checks use chronological order and break at duplicate timestamps or time gaps.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


DEFAULT_CONFIG = {
    "expected_sensors": 207,
    "frequency": "5min",
    "low_variance_std_threshold": 1.0,
    "zero_run_min_steps": 12,
    "flatline_min_steps": 12,
    "discontinuity_threshold": 30.0,
}


def _iso(timestamp: pd.Timestamp) -> str:
    return timestamp.isoformat()


def _runs(mask: np.ndarray, consecutive: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive runs, never joining across a missing time interval."""
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1] & consecutive[1:]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:] & consecutive[1:], False])
    return list(zip(starts.tolist(), ends.tolist()))


def analyze_frame(
    df: pd.DataFrame,
    locations: pd.DataFrame | None = None,
    graph_sensor_ids: list[str] | None = None,
    adjacency: np.ndarray | None = None,
    config: dict | None = None,
) -> dict[str, Any]:
    """Inspect speed measurements without changing ``df`` or its observations.

    A nonempty numeric DataFrame with a DatetimeIndex and distinct sensor IDs is
    required. IDs are compared as strings. Invalid shape/type inputs raise
    ValueError; dataset anomalies are recorded in ``summary.issues``. Timestamp
    irregularities and graph coverage failures yield ``fail``; suspect readings
    and diagnostic sensor flags yield ``warning``. No model features, split,
    imputation parameters or graph-derived predictors are constructed here.

    Configuration keys are listed in DEFAULT_CONFIG. Speeds and thresholds use
    source units. Positive near-zero speeds remain valid. Positive constant runs
    are flagged for investigation, never removed.
    """
    settings = dict(DEFAULT_CONFIG)
    if config:
        unknown = set(config) - set(settings)
        if unknown:
            raise ValueError(f"Unknown quality configuration keys: {sorted(unknown)}")
        settings.update(config)
    for name in ("zero_run_min_steps", "flatline_min_steps"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        settings[name] = int(value)
    expected_sensors = settings["expected_sensors"]
    if expected_sensors is not None:
        if isinstance(expected_sensors, bool) or not isinstance(expected_sensors, (int, np.integer)) or expected_sensors < 1:
            raise ValueError("expected_sensors must be a positive integer or None")
        settings["expected_sensors"] = int(expected_sensors)
    try:
        cadence = pd.Timedelta(settings["frequency"])
    except (ValueError, TypeError) as exc:
        raise ValueError("frequency must be a positive fixed timedelta, e.g. 5min") from exc
    if pd.isna(cadence) or cadence <= pd.Timedelta(0):
        raise ValueError("frequency must be a positive fixed timedelta")
    settings["frequency"] = str(cadence)
    sampling_minutes = float(cadence / pd.Timedelta(minutes=1))
    for name in ("low_variance_std_threshold", "discontinuity_threshold"):
        value = settings[name]
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a nonnegative finite number")
        settings[name] = float(value)

    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Expected a nonempty speed DataFrame")
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.hasnans:
        raise ValueError("Speed data must have a DatetimeIndex without NaT")
    sensor_ids = [str(sensor_id) for sensor_id in df.columns]
    if len(set(sensor_ids)) != len(sensor_ids):
        raise ValueError("Sensor IDs must be unique, including after string conversion")
    bad_columns = [str(col) for col in df.columns
                   if not is_numeric_dtype(df[col]) or is_bool_dtype(df[col])
                   or pd.api.types.is_complex_dtype(df[col])]
    if bad_columns:
        raise ValueError(f"Non-numeric speed columns: {bad_columns}")
    values = df.to_numpy(dtype=float, na_value=np.nan, copy=True)
    # Retain the original labels and order in the analysis view.
    missing = np.isnan(values)
    infinite = np.isinf(values)
    zeros = values == 0
    negative = np.isfinite(values) & (values < 0)
    valid = np.isfinite(values) & (values > 0)
    valid_speed = pd.DataFrame(np.where(valid, values, np.nan),
                               index=df.index.copy(), columns=df.columns.copy())
    issues: list[str] = []
    failures: list[str] = []

    def issue(message: str, *, fail: bool = False) -> None:
        issues.append(message)
        if fail:
            failures.append(message)

    if expected_sensors is not None and len(sensor_ids) != expected_sensors:
        issue(f"Expected {expected_sensors} sensors but found {len(sensor_ids)}.", fail=True)
    index = df.index
    unique_index = index.unique().sort_values()
    start, end = unique_index[0], unique_index[-1]
    expected_index = pd.date_range(start, end, freq=cadence)
    missing_timestamps = expected_index.difference(unique_index)
    off_grid = unique_index.difference(expected_index)
    duplicate_count = int(index.duplicated().sum())
    sorted_unique_deltas = unique_index[1:] - unique_index[:-1]
    cadence_counts = pd.Series(sorted_unique_deltas).value_counts()
    mode_minutes = (float(cadence_counts.index[0] / pd.Timedelta(minutes=1))
                    if len(cadence_counts) else None)
    if duplicate_count:
        issue(f"{duplicate_count} duplicate timestamp rows; no rows were dropped.", fail=True)
    if not index.is_monotonic_increasing:
        issue("Timestamps are not chronological; run diagnostics use a sorted copy.", fail=True)
    if len(missing_timestamps):
        issue(f"{len(missing_timestamps)} timestamps missing from the expected grid.", fail=True)
    if len(off_grid):
        issue(f"{len(off_grid)} timestamps lie outside the expected sampling grid.", fail=True)
    gap_rows = []
    for before, after in zip(unique_index[:-1], unique_index[1:]):
        delta = after - before
        if delta > cadence:
            gap_rows.append({
                "previous_timestamp": before, "next_timestamp": after,
                "elapsed_minutes": float(delta / pd.Timedelta(minutes=1)),
                "expected_missing_slots": max(0, int(delta // cadence) - 1),
                "aligned_to_cadence": bool(delta % cadence == pd.Timedelta(0)),
            })
    timestamp_gaps = pd.DataFrame(gap_rows, columns=[
        "previous_timestamp", "next_timestamp", "elapsed_minutes",
        "expected_missing_slots", "aligned_to_cadence",
    ])

    # A duplicate invalidates continuity on both sides: two values at one time
    # cannot provide an unambiguous predecessor for an outage/change check.
    order = np.argsort(index.asi8, kind="stable")
    sorted_index = index[order]
    sorted_values = values[order]
    sorted_valid = valid[order]
    duplicated_time = sorted_index.duplicated(keep=False)
    # DatetimeIndex may use microseconds (pandas 3) or nanoseconds; compare
    # timedeltas rather than mixing native integer units with Timedelta.value.
    consecutive = np.r_[False, (sorted_index[1:] - sorted_index[:-1]) == cadence]
    consecutive[1:] &= ~(duplicated_time[:-1] | duplicated_time[1:])

    zero_records, flatline_records, change_records, sensor_rows = [], [], [], []
    for position, sensor_id in enumerate(sensor_ids):
        series = sorted_values[:, position]
        series_valid = sorted_valid[:, position]
        zero_intervals = _runs(series == 0, consecutive)
        zero_lengths = [finish - begin + 1 for begin, finish in zero_intervals]
        for begin, finish in zero_intervals:
            length = finish - begin + 1
            zero_records.append({
                "sensor_id": sensor_id, "position": position,
                "start_timestamp": sorted_index[begin], "end_timestamp": sorted_index[finish],
                "length": length, "duration_minutes": length * sampling_minutes,
                "possible_outage": length >= settings["zero_run_min_steps"],
            })

        same_previous = np.r_[False, (series[1:] == series[:-1])
                             & series_valid[1:] & series_valid[:-1]]
        # Every valid observation starts/continues a constant-value segment.
        constant_intervals = _runs(series_valid, consecutive & same_previous)
        constant_lengths = [finish - begin + 1 for begin, finish in constant_intervals]
        flatline_count = 0
        for begin, finish in constant_intervals:
            length = finish - begin + 1
            if length >= settings["flatline_min_steps"]:
                flatline_count += 1
                flatline_records.append({
                    "sensor_id": sensor_id, "position": position,
                    "start_timestamp": sorted_index[begin], "end_timestamp": sorted_index[finish],
                    "length": length, "duration_minutes": length * sampling_minutes,
                    "speed": float(series[begin]),
                })

        deltas = np.full(len(series), np.nan)
        usable_pairs = consecutive[1:] & series_valid[1:] & series_valid[:-1]
        np.subtract(series[1:], series[:-1], out=deltas[1:], where=usable_pairs)
        change_positions = np.flatnonzero(np.isfinite(deltas)
                                          & (np.abs(deltas) >= settings["discontinuity_threshold"]))
        for row in change_positions:
            change_records.append({
                "sensor_id": sensor_id, "position": position,
                "previous_timestamp": sorted_index[row - 1], "timestamp": sorted_index[row],
                "previous_speed": float(series[row - 1]), "speed": float(series[row]),
                "speed_change": float(deltas[row]), "absolute_change": float(abs(deltas[row])),
            })
        observed = values[valid[:, position], position]
        count = len(observed)
        std = float(observed.std(ddof=1)) if count > 1 else None
        sensor_rows.append({
            "sensor_id": sensor_id, "position": position,
            "observation_count": len(df), "valid_count": count,
            "missing_count": int(missing[:, position].sum()),
            "infinite_count": int(infinite[:, position].sum()),
            "nonfinite_count": int((missing[:, position] | infinite[:, position]).sum()),
            "zero_count": int(zeros[:, position].sum()),
            "negative_count": int(negative[:, position].sum()),
            "invalid_count": int((~valid[:, position]).sum()),
            "valid_fraction": float(count / len(df)),
            "valid_mean": float(observed.mean()) if count else None,
            "valid_std": std, "valid_min": float(observed.min()) if count else None,
            "valid_median": float(np.median(observed)) if count else None,
            "valid_max": float(observed.max()) if count else None,
            "is_constant": bool(count > 1 and observed.min() == observed.max()),
            "is_raw_constant": bool(len(series) > 1 and np.isfinite(series).all()
                                    and (series == series[0]).all()),
            "is_all_zero": bool((series == 0).all()),
            "is_low_variance": bool(std is not None and std < settings["low_variance_std_threshold"]),
            "zero_run_count": len(zero_intervals),
            "zero_outage_run_count": sum(length >= settings["zero_run_min_steps"] for length in zero_lengths),
            "max_zero_run": max(zero_lengths, default=0),
            "flatline_run_count": flatline_count,
            "max_flatline_run": max(constant_lengths, default=0),
            "discontinuity_count": len(change_positions),
        })
    sensors = pd.DataFrame(sensor_rows)
    zero_runs = pd.DataFrame(zero_records, columns=[
        "sensor_id", "position", "start_timestamp", "end_timestamp", "length",
        "duration_minutes", "possible_outage",
    ])
    flatline_runs = pd.DataFrame(flatline_records, columns=[
        "sensor_id", "position", "start_timestamp", "end_timestamp", "length",
        "duration_minutes", "speed",
    ])
    discontinuities = pd.DataFrame(change_records, columns=[
        "sensor_id", "position", "previous_timestamp", "timestamp", "previous_speed",
        "speed", "speed_change", "absolute_change",
    ])

    metadata_summary: dict[str, Any] = {"provided": locations is not None}
    if locations is not None:
        metadata = locations.copy()
        if "sensor_id" in metadata:
            metadata_ids = metadata["sensor_id"].astype(str)
        elif not isinstance(metadata.index, pd.RangeIndex):
            metadata_ids = pd.Series(metadata.index.astype(str), index=metadata.index)
        else:
            metadata_ids = pd.Series(dtype=str)
            issue("Location metadata has no sensor_id column or labeled sensor index.")
        duplicate_metadata = int(metadata_ids.duplicated().sum())
        id_set = set(metadata_ids)
        missing_ids = [sid for sid in sensor_ids if sid not in id_set]
        extra_ids = sorted(id_set - set(sensor_ids))
        metadata_summary.update({
            "sensor_count": len(metadata_ids), "duplicate_sensor_ids": duplicate_metadata,
            "missing_sensor_ids": missing_ids, "extra_sensor_ids": extra_ids,
            "all_sensors_covered": not missing_ids and not duplicate_metadata,
        })
        if duplicate_metadata:
            issue("Location metadata contains duplicate sensor IDs; coordinates were not joined.")
        elif len(metadata_ids):
            metadata.index = metadata_ids.to_numpy()
            for destination, aliases in {"latitude": ("latitude", "lat"),
                                         "longitude": ("longitude", "lon", "lng")}.items():
                source = next((name for name in aliases if name in metadata), None)
                if source is not None:
                    sensors[destination] = pd.to_numeric(
                        metadata[source].reindex(sensor_ids), errors="coerce"
                    ).to_numpy()
        if missing_ids:
            issue(f"Location metadata is missing {len(missing_ids)} data sensors.")
        if extra_ids:
            issue(f"Location metadata has {len(extra_ids)} sensors absent from speed data.")

    graph_summary: dict[str, Any] = {
        "sensor_ids_provided": graph_sensor_ids is not None,
        "adjacency_provided": adjacency is not None, "used_as_features": False,
    }
    if graph_sensor_ids is not None:
        graph_ids = [str(item) for item in graph_sensor_ids]
        graph_set = set(graph_ids)
        missing_ids = [sid for sid in sensor_ids if sid not in graph_set]
        extra_ids = sorted(graph_set - set(sensor_ids))
        graph_summary.update({
            "sensor_count": len(graph_ids), "duplicate_sensor_ids": len(graph_ids) - len(graph_set),
            "same_sensor_set": not missing_ids and not extra_ids and len(graph_ids) == len(sensor_ids),
            "same_sensor_order": graph_ids == sensor_ids,
            "missing_sensor_ids": missing_ids, "extra_sensor_ids": extra_ids,
        })
        if not graph_summary["same_sensor_set"]:
            issue("Graph sensor IDs do not match speed-data sensor coverage.", fail=True)
        elif not graph_summary["same_sensor_order"]:
            issue("Graph sensor order differs from speed columns; retain and align IDs explicitly.", fail=True)
    if adjacency is not None:
        matrix = np.asarray(adjacency)
        square = matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]
        shape_ok = square and matrix.shape[0] == len(sensor_ids)
        numeric = np.issubdtype(matrix.dtype, np.number)
        finite = bool(numeric and np.isfinite(matrix).all())
        graph_summary.update({"adjacency_shape": list(matrix.shape),
                              "adjacency_shape_valid": bool(shape_ok),
                              "adjacency_finite": finite})
        if not shape_ok or not finite:
            issue("Adjacency must be a finite numeric square matrix matching the sensor count.", fail=True)

    invalid_count = int((~valid).sum())
    if invalid_count:
        issue(f"{invalid_count} suspect/invalid speed values are masked in valid_speed; raw values are preserved.")
    if bool((sensors["valid_count"] == 0).any()):
        issue(f"{int((sensors['valid_count'] == 0).sum())} sensors have no valid positive observations.")
    if bool(sensors["is_low_variance"].any()):
        issue(f"{int(sensors['is_low_variance'].sum())} sensors have unusually low valid-speed variance.")
    if len(flatline_runs):
        issue(f"{len(flatline_runs)} positive constant runs warrant inspection; readings are retained.")
    if len(discontinuities):
        issue(f"{len(discontinuities)} large consecutive speed changes warrant inspection; readings are retained.")

    missingness_columns = ["sensor_id", "position", "observation_count", "valid_count", "missing_count",
                           "infinite_count", "nonfinite_count", "zero_count", "negative_count", "invalid_count"]
    missingness_summary = sensors[missingness_columns].copy()
    for kind in ("missing", "infinite", "nonfinite", "zero", "negative", "invalid"):
        missingness_summary[f"{kind}_fraction"] = missingness_summary[f"{kind}_count"] / len(df)
    missingness_over_time = pd.DataFrame({
        "missing_count": missing.sum(axis=1), "zero_count": zeros.sum(axis=1),
        "invalid_count": (~valid).sum(axis=1), "valid_count": valid.sum(axis=1),
        "missing_fraction": missing.mean(axis=1), "zero_fraction": zeros.mean(axis=1),
        "invalid_fraction": (~valid).mean(axis=1), "all_zero": zeros.all(axis=1),
        "all_invalid": (~valid).all(axis=1),
    }, index=index.copy())
    summary = {
        "validation_status": "fail" if failures else ("warning" if issues else "pass"),
        "issues": issues,
        "dimensions": {
            "timestamps": len(df), "unique_timestamps": len(unique_index), "sensors": len(sensor_ids),
            "observed_cells": int(values.size), "expected_timestamps": len(expected_index),
            "expected_grid_cells": len(expected_index) * len(sensor_ids),
            "missing_timestamp_cells": len(missing_timestamps) * len(sensor_ids),
        },
        "timestamp_validation": {
            "start": _iso(start), "end": _iso(end), "timezone": str(index.tz) if index.tz else None,
            "timezone_note": "Naive source timestamps; no timezone inferred." if index.tz is None else "Source timezone preserved.",
            "expected_sampling_minutes": sampling_minutes, "modal_sampling_minutes": mode_minutes,
            "duplicate_rows": duplicate_count, "chronologically_ordered": bool(index.is_monotonic_increasing),
            "missing_timestamp_count": len(missing_timestamps),
            "missing_timestamps": [_iso(item) for item in missing_timestamps],
            "off_grid_timestamp_count": len(off_grid),
            "off_grid_timestamps": [_iso(item) for item in off_grid], "gap_count": len(timestamp_gaps),
        },
        "value_counts": {
            "missing_nan": int(missing.sum()), "infinite": int(infinite.sum()),
            "zero": int(zeros.sum()), "negative_finite": int(negative.sum()),
            "valid_positive_finite": int(valid.sum()), "invalid_total": invalid_count,
        },
        "zero_evidence": {
            "all_zero_timestamp_rows": int(zeros.all(axis=1).sum()),
            "at_least_90pct_zero_timestamp_rows": int((zeros.mean(axis=1) >= 0.9).sum()),
            "all_invalid_timestamp_rows": int((~valid).all(axis=1).sum()),
            "total_zero_runs": len(zero_runs),
            "possible_outage_runs": int(sensors["zero_outage_run_count"].sum()),
            "maximum_zero_run_steps": int(sensors["max_zero_run"].max()),
            "interpretation": "Zero is suspect, not proof of an outage or genuinely stopped traffic.",
        },
        "sensor_quality": {
            "no_valid_observations": int((sensors["valid_count"] == 0).sum()),
            "constant_sensors": int(sensors["is_constant"].sum()),
            "raw_constant_sensors": int(sensors["is_raw_constant"].sum()),
            "all_zero_sensors": int(sensors["is_all_zero"].sum()),
            "low_variance_sensors": int(sensors["is_low_variance"].sum()),
            "positive_flatline_runs": len(flatline_runs), "large_discontinuities": len(discontinuities),
        },
        "metadata_validation": metadata_summary, "graph_validation": graph_summary,
        "policy": {**settings, "invalid_values": ["NaN", "infinity", "negative", "exact zero"],
                   "positive_near_zero": "retained", "imputation": "none", "raw_modified": False,
                   "valid_std_ddof": 1},
    }
    return {
        "summary": summary, "sensors": sensors, "missingness_summary": missingness_summary,
        "missingness_over_time": missingness_over_time, "timestamp_gaps": timestamp_gaps,
        "zero_runs": zero_runs, "flatline_runs": flatline_runs,
        "discontinuities": discontinuities, "valid_speed": valid_speed,
    }
