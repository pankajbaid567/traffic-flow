"""Reproducible descriptive findings for the Person A data-quality report.

These functions neither fit preprocessing nor change or save measurements.
Temporal means pool available sensor cells, so groups with fewer valid sensors
or timestamps are not given the same weight as more complete groups.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
PERIODS = ["Weekday morning peak", "Weekday evening peak", "Weekday off-peak", "Weekend"]


def _profile(
    groups: np.ndarray, labels: list, row_sums: np.ndarray,
    row_counts: np.ndarray, sensor_count: int, key: str,
) -> pd.DataFrame:
    records = []
    for label in labels:
        keep = groups == label
        n_timestamps = int(keep.sum())
        total = n_timestamps * sensor_count
        count = int(row_counts[keep].sum())
        records.append({key: label, "timestamp_count": n_timestamps,
                        "total_sensor_cells": total, "valid_count": count,
                        "excluded_count": total - count,
                        "excluded_fraction": (total - count) / total if total else None,
                        "mean_speed": float(row_sums[keep].sum() / count) if count else None})
    return pd.DataFrame(records)


def _network_zero_blocks(df: pd.DataFrame, zero_rows: np.ndarray) -> pd.DataFrame:
    """Identify all-sensor-zero blocks; duplicates and elapsed gaps break them."""
    order = np.argsort(df.index.asi8, kind="stable")
    index = df.index[order]
    zero = zero_rows[order]
    duplicates = index.duplicated(keep=False)
    cadence = pd.Timedelta(minutes=5)
    continuous = np.r_[False, (index[1:] - index[:-1]) == cadence]
    continuous[1:] &= ~(duplicates[:-1] | duplicates[1:])
    starts = np.flatnonzero(zero & ~np.r_[False, zero[:-1] & continuous[1:]])
    ends = np.flatnonzero(zero & ~np.r_[zero[1:] & continuous[1:], False])
    records = [{"start_timestamp": index[begin], "end_timestamp": index[end],
                "length": int(end - begin + 1),
                "slot_minutes": int(end - begin + 1) * 5,
                "elapsed_minutes": float((index[end] - index[begin]) / pd.Timedelta(minutes=1))}
               for begin, end in zip(starts, ends)]
    return pd.DataFrame(records, columns=["start_timestamp", "end_timestamp", "length",
                                          "slot_minutes", "elapsed_minutes"])


def _extreme(frame: pd.DataFrame, key: str, *, largest: bool) -> dict | None:
    observed = frame.loc[frame["valid_count"] > 0]
    if observed.empty:
        return None
    row = observed.sort_values("mean_speed", ascending=not largest, kind="stable").iloc[0]
    value = row[key]
    if key in ("hour", "day_of_week") or isinstance(value, np.integer):
        value = int(value)
    return {key: value, "mean_speed": float(row["mean_speed"]), "valid_count": int(row["valid_count"])}


def summarize_descriptive(df: pd.DataFrame, analysis: dict) -> dict[str, Any]:
    """Return JSON-safe findings and full evidence tables without mutations.

    ``analysis`` is the result of :func:`quality.analyze_frame`. Profiles are
    weighted by valid sensor-cell counts, not averages of timestamp/sensor means.
    Positive near-zero readings remain observed. Calendar groups use the stored
    clock; neither physical speed units nor a missing timezone are inferred.
    """
    valid_speed = analysis["valid_speed"]
    if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
        raise ValueError("Expected nonempty timestamp-indexed speed data")
    if not valid_speed.index.equals(df.index) or not valid_speed.columns.equals(df.columns):
        raise ValueError("valid_speed must retain the exact raw index and sensor order")
    values = valid_speed.to_numpy(dtype=float, na_value=np.nan, copy=True)
    valid = np.isfinite(values) & (values > 0)
    observed = values[valid]
    row_sums = np.where(valid, values, 0).sum(axis=1)
    row_counts = valid.sum(axis=1)
    sensor_count = len(df.columns)
    total_cells = int(values.size)
    count = int(valid.sum())
    quantiles = np.quantile(observed, [.05, .5, .95]) if count else [None] * 3
    hourly = _profile(np.asarray(df.index.hour), list(range(24)), row_sums, row_counts, sensor_count, "hour")
    daily = _profile(np.asarray(df.index.dayofweek), list(range(7)), row_sums, row_counts, sensor_count, "day_of_week")
    daily.insert(1, "day_name", DAY_NAMES)
    weekend = np.asarray(df.index.dayofweek >= 5)
    weekday_weekend = _profile(np.where(weekend, "Weekend", "Weekday"), ["Weekday", "Weekend"],
                               row_sums, row_counts, sensor_count, "day_type")
    hour = np.asarray(df.index.hour)
    periods = np.full(len(df), "Weekday off-peak", dtype=object)
    periods[(~weekend) & (hour >= 7) & (hour < 10)] = "Weekday morning peak"
    periods[(~weekend) & (hour >= 16) & (hour < 19)] = "Weekday evening peak"
    periods[weekend] = "Weekend"
    traffic = _profile(periods, PERIODS, row_sums, row_counts, sensor_count, "period")

    raw = df.to_numpy(dtype=float, na_value=np.nan, copy=True)
    zeros = raw == 0
    zero_count = int(zeros.sum())
    zero_rows = zeros.all(axis=1)
    zero_blocks = _network_zero_blocks(df, zero_rows)
    zero_sensor_table = pd.DataFrame({
        "sensor_id": [str(item) for item in df.columns], "position": range(sensor_count),
        "zero_count": zeros.sum(axis=0), "zero_fraction": zeros.mean(axis=0),
    })
    top_sensors = zero_sensor_table.sort_values(["zero_count", "position"], ascending=[False, True], kind="stable").head(3).reset_index(drop=True)
    longest_runs = analysis["zero_runs"].sort_values("length", ascending=False, kind="stable").head(3).reset_index(drop=True).copy()
    longest_network = zero_blocks.sort_values("length", ascending=False, kind="stable").head(1)
    if longest_network.empty:
        longest_network_summary = None
    else:
        longest = longest_network.iloc[0]
        longest_network_summary = {
            "start": longest["start_timestamp"].isoformat(),
            "end": longest["end_timestamp"].isoformat(),
            "steps": int(longest["length"]), "slot_minutes": int(longest["slot_minutes"]),
        }
    timestamps = analysis["summary"]["timestamp_validation"]
    network_zero_cells = int(zero_rows.sum()) * sensor_count
    summary = {
        "coverage": {
            "start": timestamps["start"], "end": timestamps["end"],
            "timestamp_count": len(df), "sensor_count": sensor_count,
            "timezone": str(df.index.tz) if df.index.tz else None,
            "speed_units": "unverified_source_units",
        },
        "overall_valid_speed": {
            "n": count, "mean": float(observed.mean()) if count else None,
            "median": float(quantiles[1]) if count else None,
            "p05": float(quantiles[0]) if count else None,
            "p95": float(quantiles[2]) if count else None,
        },
        "availability": {
            "total_sensor_cells": total_cells, "valid_sensor_cells": count,
            "excluded_sensor_cells": total_cells - count,
            "excluded_fraction": (total_cells - count) / total_cells,
            "exact_zero_cells": zero_count, "exact_zero_fraction": zero_count / total_cells,
            "nonzero_excluded_cells": total_cells - count - zero_count,
        },
        "temporal_associations": {
            "lowest_mean_hour": _extreme(hourly, "hour", largest=False),
            "highest_mean_hour": _extreme(hourly, "hour", largest=True),
            "lowest_mean_day": _extreme(daily, "day_name", largest=False),
            "highest_mean_day": _extreme(daily, "day_name", largest=True),
            "aggregation": "pooled_valid_sensor_cells",
        },
        "network_zero_evidence": {
            "all_zero_timestamp_rows": int(zero_rows.sum()),
            "all_zero_timestamp_fraction": float(zero_rows.mean()),
            "contiguous_blocks": len(zero_blocks),
            "longest_block": longest_network_summary,
            "all_zero_sensor_cells": network_zero_cells,
            "share_of_zero_cells_in_all_zero_rows": network_zero_cells / zero_count if zero_count else None,
        },
        "limitations": {
            "missingness_bias": "Means describe observed positive sensor cells; nonrandom unavailable readings may bias temporal and sensor comparisons.",
            "physical_cause": "Simultaneous zeros and long runs support a missingness investigation but do not establish stopped traffic, outages or corruption.",
            "preprocessing": "No trained preprocessing or imputation; full-period summaries are descriptive only.",
        },
    }
    return {"summary": summary, "tables": {
        "hourly_profile": hourly, "day_of_week_profile": daily,
        "weekday_weekend_profile": weekday_weekend, "traffic_period_profile": traffic,
        "all_network_zero_blocks": zero_blocks, "top_zero_sensors": top_sensors,
        "longest_zero_runs": longest_runs,
    }}


def _number(value: Any, decimals: int = 2) -> str:
    return "unavailable" if value is None or pd.isna(value) else f"{float(value):,.{decimals}f}"


def _markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "None detected."

    def cell(value: Any) -> str:
        if value is None or pd.isna(value):
            return "unavailable"
        if isinstance(value, (float, np.floating)):
            return _number(value)
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(map(str, frame.columns)) + " |",
             "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines.extend("| " + " | ".join(cell(item) for item in row) + " |"
                 for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def render_descriptive(summary: dict, tables: dict[str, pd.DataFrame]) -> str:
    """Render concise findings from measured values; never hardcode results."""
    coverage, overall = summary["coverage"], summary["overall_valid_speed"]
    availability = summary["availability"]
    network = summary["network_zero_evidence"]
    lines = ["## Measured descriptive findings",
             f"The source covers **{coverage['start']} through {coverage['end']}**, with {coverage['timestamp_count']:,} timestamps and {coverage['sensor_count']:,} sensors. Speeds use unverified source units. Calendar groups use the stored clock; no timezone is inferred or converted.",
             f"Across **{overall['n']:,} valid positive sensor cells**, mean speed is **{_number(overall['mean'])}**, median **{_number(overall['median'])}**, and the 5th–95th percentile range is **{_number(overall['p05'])}–{_number(overall['p95'])}**.",
             f"Excluded readings total **{availability['excluded_sensor_cells']:,} / {availability['total_sensor_cells']:,} ({100 * availability['excluded_fraction']:.2f}%)**; exact zeros account for {availability['exact_zero_cells']:,}. Positive near-zero speeds remain observed. No imputation or trained preprocessing is performed."]
    temporal = summary["temporal_associations"]
    low_hour, high_hour = temporal["lowest_mean_hour"], temporal["highest_mean_hour"]
    if low_hour is not None and high_hour is not None:
        low_day, high_day = temporal["lowest_mean_day"], temporal["highest_mean_day"]
        lines.append(f"Hourly pooled means are lowest at stored hour **{low_hour['hour']:02d}:00 ({_number(low_hour['mean_speed'])})** and highest at **{high_hour['hour']:02d}:00 ({_number(high_hour['mean_speed'])})**. Daily pooled means are lowest on **{low_day['day_name']} ({_number(low_day['mean_speed'])})** and highest on **{high_day['day_name']} ({_number(high_day['mean_speed'])})**.")
    lines += ["All temporal means pool valid sensor cells using their counts. They are descriptive associations; unavailable readings may be nonrandom, so these profiles can be biased. Counts and exclusion rates accompany the means."]
    for key, title in [("weekday_weekend_profile", "Weekday and weekend observations"),
                       ("traffic_period_profile", "Traffic-period observations")]:
        profile = tables[key]
        display = profile[[profile.columns[0], "valid_count", "mean_speed", "excluded_fraction"]].copy()
        display["excluded_fraction"] *= 100
        display.columns = [profile.columns[0].replace("_", " "), "Valid cells", "Mean speed", "Excluded (%)"]
        lines += [f"### {title}", _markdown(display)]
    lines.append("Peak windows are weekdays [07:00, 10:00) and [16:00, 19:00); weekday off-peak excludes both windows. Weekend observations are reported separately. Full hourly and day-of-week tables are exported alongside these tables.")
    lines += ["### Strongest zero-value evidence",
              f"All {coverage['sensor_count']:,} sensors are zero together in **{network['all_zero_timestamp_rows']:,} timestamp rows ({100 * network['all_zero_timestamp_fraction']:.2f}%)**, forming **{network['contiguous_blocks']:,}** consecutive five-minute blocks. Gaps and duplicate timestamps interrupt blocks."]
    if network["share_of_zero_cells_in_all_zero_rows"] is not None:
        lines.append(f"These simultaneous-zero rows account for **{100 * network['share_of_zero_cells_in_all_zero_rows']:.4f}%** of all exact-zero sensor cells.")
    longest = network["longest_block"]
    if longest is not None:
        lines.append(f"The longest simultaneous-zero block has **{longest['steps']:,} slots**, from **{longest['start']}** through **{longest['end']}** ({longest['slot_minutes']:,} minutes of five-minute slots; the elapsed first-to-last span is one slot shorter).")
    display = tables["top_zero_sensors"][["sensor_id", "zero_count", "zero_fraction"]].copy()
    display["zero_fraction"] *= 100
    display.columns = ["Sensor", "Zero cells", "Zero (%)"]
    lines += ["Sensors with the highest zero rates:", _markdown(display),
              "Longest individual zero runs (slot duration includes all observed five-minute slots):",
              _markdown(tables["longest_zero_runs"]),
              "Simultaneous zeros and long zero runs are evidence for investigating unavailable measurements. They do not establish a physical cause or prove an outage. Raw values are preserved, exact zeros are flagged as benchmark missing/suspect, and no sensor is automatically removed."]
    return "\n\n".join(lines) + "\n"
