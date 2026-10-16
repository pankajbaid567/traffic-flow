"""Reproducible, descriptive Person A plots (no model or graph features).

All speed summaries use ``analysis['valid_speed']``: zero, negative and
nonfinite measurements are unavailable under the Phase 1 quality rule.  These
plots describe the supplied period, including held-out dates; their statistics
must not be reused as fitted preprocessing parameters or model-selection data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_FOOTNOTE = (
    "Descriptive EDA across the supplied period; not training-only statistics. "
    "Speed units and time zone are unverified.\n"
    "Zero, nonfinite and negative values are excluded; exclusion alone does not establish sensor outages."
)


def _profile(valid: pd.DataFrame, groups: Any, levels: Any) -> pd.Series:
    """Mean over observed sensor cells, rather than mean of unequal-size means."""
    row_sums = valid.sum(axis=1, min_count=1)
    row_counts = valid.count(axis=1)
    numer = row_sums.groupby(groups).sum(min_count=1)
    denom = row_counts.groupby(groups).sum()
    return numer.div(denom.where(denom > 0)).reindex(levels)


def _ticks(length: int, limit: int = 18) -> np.ndarray:
    return np.unique(np.linspace(0, length - 1, min(length, limit), dtype=int))


def _no_data(ax: Any, message: str = "No valid positive speed observations") -> None:
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center")


def _draw_profile(ax: Any, profile: pd.Series, *, label: str | None = None,
                  color: str | None = None) -> bool:
    if not profile.notna().any():
        return False
    ax.plot(profile.index, profile.to_numpy(), label=label, color=color, linewidth=1.8)
    return True


def _series_with_time_gaps(series: pd.Series) -> pd.Series:
    """Insert plot-only separators without removing actual observations.

    Repeated x coordinates place a NaN immediately before the first point after
    any non-five-minute interval. Unlike assigning NaN to that point, this keeps
    both observed endpoints. The source Series and raw timestamps stay intact.
    """
    timestamps: list[pd.Timestamp] = []
    values: list[float] = []
    previous: pd.Timestamp | None = None
    for timestamp, value in series.items():
        if previous is not None and timestamp - previous != pd.Timedelta(minutes=5):
            timestamps.append(timestamp)
            values.append(np.nan)
        timestamps.append(timestamp)
        values.append(value)
        previous = timestamp
    return pd.Series(values, index=pd.DatetimeIndex(timestamps), dtype=float)


def generate_eda(
    df: pd.DataFrame, analysis: dict[str, Any], output_dir: Path,
) -> dict[str, str]:
    """Save ten PNG figures and return ``figure_key -> relative PNG filename``.

    ``valid_speed`` must have the same timestamp and sensor order as ``df``.
    All sensors enter distribution and quality plots.  Sensor labels are thinned
    only to make axes legible; no sensors are omitted.  Aggregate profiles pool
    valid sensor cells. Missingness includes excluded zero/negative/nonfinite
    measurements at existing timestamps. Absent calendar days are shown as gray
    heatmap cells and gaps in the daily profile, never as zero exclusion. Absent
    individual timestamps are audited by the quality module; sample time-series
    lines break at every non-five-minute interval without dropping observed
    endpoints or filling the raw data.

    Representative sensors are chosen deterministically by lowest exclusion,
    highest exclusion and highest valid sample variance, with original column
    order breaking ties. Their selection is recorded in the sidecar
    ``representative_sensor_selection.csv``. A first-seven-calendar-days window
    is plotted at a bounded resolution. Empty frames raise a clear error;
    nonempty but entirely unavailable frames produce explanatory figures.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("generate_eda requires a pandas DatetimeIndex.")
    if df.empty or df.shape[1] == 0:
        raise ValueError("generate_eda requires at least one timestamp and one sensor.")
    if df.index.hasnans:
        raise ValueError("generate_eda does not accept missing timestamp labels (NaT).")
    if not df.index.is_monotonic_increasing:
        raise ValueError("generate_eda requires timestamps in chronological order.")
    valid = analysis.get("valid_speed")
    if not isinstance(valid, pd.DataFrame):
        raise ValueError("analysis['valid_speed'] must be a pandas DataFrame.")
    if not valid.index.equals(df.index) or not valid.columns.equals(df.columns):
        raise ValueError("valid_speed must preserve the raw timestamp and sensor order.")
    valid = valid.astype(float, copy=False)
    # Defensive masking also keeps externally supplied analysis consistent with
    # the documented rule; raw data are never modified by this function.
    valid = valid.where(np.isfinite(valid) & (valid > 0))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    sensor_count = valid.shape[1]
    sensor_labels = [str(sensor) for sensor in valid.columns]
    positions = np.arange(sensor_count)
    sensor_ticks = _ticks(sensor_count)
    valid_counts = valid.count()
    has_valid = bool(valid_counts.sum())
    missing = valid.isna()
    exclusion_rate = missing.mean()
    variances = valid.var(ddof=1)

    def save(key: str, fig: Any, title: str) -> None:
        filename = f"{key}.png"
        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.text(0.01, 0.012, _FOOTNOTE, fontsize=7.5, color="#4b5563", va="bottom")
        fig.tight_layout(rect=(0, 0.085, 1, 0.94))
        try:
            fig.savefig(output_dir / filename, dpi=160, facecolor="white")
        finally:
            plt.close(fig)
        artifacts[key] = filename

    def label_sensor_axis(ax: Any, *, horizontal: bool = True) -> None:
        if horizontal:
            ax.set_xticks(sensor_ticks)
            ax.set_xticklabels([sensor_labels[i] for i in sensor_ticks], rotation=60, ha="right", fontsize=8)
            ax.set_xlabel("Sensor ID (all sensors, source column order)")
        else:
            ax.set_yticks(sensor_ticks)
            ax.set_yticklabels([sensor_labels[i] for i in sensor_ticks], fontsize=7)
            ax.set_ylabel("Sensor ID (all sensors)")

    # Count every available measurement in chunks; avoid subsampling the tails.
    fig, ax = plt.subplots(figsize=(10, 5))
    if has_valid:
        lower = float(valid.min().min())
        upper = float(valid.max().max())
        if lower == upper:
            margin = max(abs(lower) * 0.02, 0.5)
            lower, upper = lower - margin, upper + margin
        edges = np.linspace(lower, upper, 61)
        counts = np.zeros(len(edges) - 1, dtype=np.int64)
        for start in range(0, len(valid), 2048):
            chunk = valid.iloc[start:start + 2048].to_numpy().ravel()
            chunk = chunk[np.isfinite(chunk)]
            counts += np.histogram(chunk, bins=edges)[0]
        ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge", color="#2563eb")
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.set_ylabel("Valid measurements (count)")
    else:
        _no_data(ax)
    ax.set_xlabel("Traffic speed (source units; unverified)")
    save("overall_speed_distribution", fig, "Overall distribution of valid traffic speed")

    fig, ax = plt.subplots(figsize=(14, 5))
    if has_valid:
        quantiles = valid.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#d1d5db")
        shown = ax.imshow(np.ma.masked_invalid(quantiles.to_numpy()), aspect="auto", cmap=cmap,
                          interpolation="nearest")
        fig.colorbar(shown, ax=ax, label="Traffic speed (source units; unverified)")
        ax.set_yticks(np.arange(5))
        ax.set_yticklabels(["5th", "25th", "Median", "75th", "95th"])
        ax.set_ylabel("Within-sensor quantile")
    else:
        _no_data(ax)
    label_sensor_axis(ax)
    save("sensor_speed_distribution", fig, "Speed distributions across every sensor (gray = unavailable)")

    hourly = _profile(valid, valid.index.hour, range(24))
    fig, ax = plt.subplots(figsize=(10, 5))
    if not _draw_profile(ax, hourly, color="#2563eb"):
        _no_data(ax)
    ax.set(xticks=range(0, 24, 2), xlabel="Hour in source timestamp", ylabel="Mean valid speed (source units)")
    ax.grid(alpha=0.2)
    save("average_speed_by_hour", fig, "Average traffic speed by hour (pooled observed sensor values)")

    daily = _profile(valid, valid.index.dayofweek, range(7))
    fig, ax = plt.subplots(figsize=(10, 5))
    if daily.notna().any():
        ax.bar(np.arange(7), daily.to_numpy(), color="#2563eb")
    else:
        _no_data(ax)
    ax.set_xticks(np.arange(7))
    ax.set_xticklabels(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    ax.set_ylabel("Mean valid speed (source units)")
    save("average_speed_by_day_of_week", fig, "Average traffic speed by day of week")

    minutes = valid.index.hour * 60 + valid.index.minute
    fig, ax = plt.subplots(figsize=(11, 5))
    drawn = False
    for is_weekend, label, color in [(False, "Weekday", "#2563eb"), (True, "Weekend", "#d97706")]:
        keep = (valid.index.dayofweek >= 5) == is_weekend
        if keep.any():
            profile = _profile(valid.loc[keep], minutes[keep], range(0, 1440, 5))
            profile.index = profile.index / 60
            drawn = _draw_profile(ax, profile, label=label, color=color) or drawn
    if drawn:
        ax.legend()
    else:
        _no_data(ax)
    ax.set(xticks=range(0, 25, 2), xlabel="Hour in source timestamp", ylabel="Mean valid speed (source units)")
    ax.grid(alpha=0.2)
    save("weekday_weekend_profile", fig, "Weekday and weekend profiles (five-minute time slots)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, start, end, label in [(axes[0], 7, 10, "Morning: 07:00 ≤ time < 10:00"),
                                 (axes[1], 16, 19, "Evening: 16:00 ≤ time < 19:00")]:
        keep = (valid.index.dayofweek < 5) & (minutes >= start * 60) & (minutes < end * 60)
        drawn = False
        if keep.any():
            profile = _profile(valid.loc[keep], minutes[keep], range(start * 60, end * 60, 5))
            profile.index = profile.index / 60
            drawn = _draw_profile(ax, profile, color="#b45309")
        if not drawn:
            _no_data(ax, "No valid weekday measurements in this window")
        ax.set(title=label, xlabel="Hour in source timestamp", xticks=np.arange(start, end + 0.1, 0.5))
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Mean valid speed (source units)")
    save("morning_evening_congestion", fig, "Weekday peak-window speed profiles (descriptive, not a congestion threshold)")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(positions, exclusion_rate.to_numpy() * 100, color="#7c3aed", width=0.9)
    label_sensor_axis(ax)
    ax.set(ylim=(0, 100), ylabel="Unavailable / excluded measurements (%)")
    save("missingness_by_sensor", fig, "Unavailable measurements by sensor (includes zero/negative/nonfinite values)")

    daily_missing = missing.groupby(missing.index.normalize()).mean()
    calendar_days = pd.date_range(daily_missing.index[0], daily_missing.index[-1], freq="D")
    # Reindex this plotting summary only. NaN denotes an absent date, while an
    # observed day with zero excluded values must retain its genuine 0% value.
    daily_missing = daily_missing.reindex(calendar_days)
    fig, (ax_heat, ax_line) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [2, 1]})
    missing_cmap = plt.get_cmap("magma").copy()
    missing_cmap.set_bad("#d1d5db")
    shown = ax_heat.imshow(np.ma.masked_invalid(daily_missing.to_numpy().T * 100),
                           aspect="auto", vmin=0, vmax=100, cmap=missing_cmap,
                           interpolation="nearest")
    date_ticks = _ticks(len(daily_missing), limit=10)
    ax_heat.set_xticks(date_ticks)
    ax_heat.set_xticklabels([daily_missing.index[i].strftime("%Y-%m-%d") for i in date_ticks], rotation=30, ha="right")
    label_sensor_axis(ax_heat, horizontal=False)
    fig.colorbar(shown, ax=ax_heat, label="Unavailable / excluded (%)", pad=0.01)
    ax_heat.set_title("Per-sensor daily exclusion fractions; gray = no timestamp rows that day")
    ax_line.plot(daily_missing.index, daily_missing.mean(axis=1) * 100, color="#7c3aed",
                 linewidth=1.5, marker="o", markersize=3)
    ax_line.set(ylim=(0, 100), ylabel="All-sensor exclusion (%)", xlabel="Date; fractions use existing timestamps only")
    ax_line.set_xticks(daily_missing.index[date_ticks])
    ax_line.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax_line.tick_params(axis="x", rotation=25)
    ax_line.grid(alpha=0.2)
    save("missingness_over_time", fig, "Unavailable measurements over time (absent days remain unavailable)")

    fig, ax = plt.subplots(figsize=(10, 5))
    finite_variances = variances.dropna()
    if len(finite_variances):
        ax.hist(finite_variances.to_numpy(), bins=min(30, max(1, len(finite_variances))), color="#059669")
    else:
        _no_data(ax, "No sensor has at least two valid observations")
    ax.set(xlabel="Within-sensor sample variance (source speed units squared)", ylabel="Sensors (count)")
    save("sensor_variance_distribution", fig, "Distribution of sensor speed variance (ddof = 1)")

    # Descriptive selections use stable source order for ties, never randomness.
    candidates: list[tuple[int, str]] = [
        (int(np.argmin(exclusion_rate.to_numpy())), "lowest_exclusion_rate"),
        (int(np.argmax(exclusion_rate.to_numpy())), "highest_exclusion_rate"),
    ]
    if variances.notna().any():
        candidates.append((int(np.nanargmax(variances.to_numpy())), "highest_valid_variance"))
    selected: dict[int, list[str]] = {}
    for sensor_pos, reason in candidates:
        selected.setdefault(sensor_pos, []).append(reason)
    window_end = valid.index[0] + pd.Timedelta(days=7)
    window = valid.loc[valid.index < window_end]
    # Keep plotting bounded, even if a nonbenchmark sampling cadence is supplied.
    sample_positions = _ticks(len(window), limit=2500)
    rows = []
    fig, axes = plt.subplots(len(selected), 1, figsize=(13, 3 * len(selected) + 1), squeeze=False)
    for ax, (sensor_pos, reasons) in zip(axes[:, 0], selected.items()):
        sample = window.iloc[sample_positions, sensor_pos]
        if sample.notna().any():
            plotted = _series_with_time_gaps(sample)
            # Small markers keep isolated observations visible on either side
            # of a time gap; NaN separators prevent misleading connecting lines.
            ax.plot(plotted.index, plotted.to_numpy(), color="#2563eb", linewidth=0.7,
                    marker=".", markersize=1.5)
        else:
            _no_data(ax, "No valid observations in displayed window")
        ax.set_title(f"Sensor {sensor_labels[sensor_pos]}: {', '.join(reasons).replace('_', ' ')}", fontsize=10)
        ax.set_ylabel("Speed (source units)")
        display_end = window.index[-1]
        if display_end == window.index[0]:
            display_end += pd.Timedelta(minutes=5)
        ax.set_xlim(window.index[0], display_end)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.tick_params(axis="x", rotation=20)
        ax.grid(alpha=0.2)
        rows.append({
            "sensor_id": sensor_labels[sensor_pos],
            "sensor_column_position": sensor_pos,
            "selection_reasons": ";".join(reasons),
            "unavailable_fraction": float(exclusion_rate.iloc[sensor_pos]),
            "valid_sample_variance": float(variances.iloc[sensor_pos]),
            "display_start": window.index[0].isoformat(),
            "display_end_inclusive": window.index[-1].isoformat(),
            "displayed_rows": len(sample),
            "selection_scope": "whole-period descriptive EDA; not model selection",
        })
    axes[-1, 0].set_xlabel("Source timestamp; gaps mark excluded speeds or non-five-minute intervals")
    pd.DataFrame(rows).to_csv(output_dir / "representative_sensor_selection.csv", index=False)
    save("sample_sensor_timeseries", fig, "Deterministic sensor examples (first seven calendar days; gaps retained)")
    return artifacts
