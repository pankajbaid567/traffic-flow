"""Feature engineering for the Phase 1 local-sensor models (Person B).

Everything here is computed per sensor from its own speed only, and every
window ends at the origin t (pandas rolling/shift are backward looking).
No neighbour sensors or graph info, that is for Phase 2.

32 features:
  12 lags          speed_lag_0 (t) ... speed_lag_11 (t-11)
   8 rolling       mean/std over 15, 30, 60 min, min/max over 30 min
   3 trend         speed_t - speed_{t-1}, t-3, t-6
   7 time          hour sin/cos, day-of-week sin/cos, weekend, AM/PM peak
   2 missingness   reading at t was missing, number missing in last hour

Targets are the raw speed at t+1, t+3, t+6, t+12 plus a valid flag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LAG_COLUMNS = [f"speed_lag_{i}" for i in range(12)]

# name -> (window in 5-min steps, pandas rolling method)
ROLLING_SPECS = {
    "rolling_mean_15min": (3, "mean"),
    "rolling_mean_30min": (6, "mean"),
    "rolling_mean_60min": (12, "mean"),
    "rolling_std_15min": (3, "std"),
    "rolling_std_30min": (6, "std"),
    "rolling_std_60min": (12, "std"),
    "rolling_min_30min": (6, "min"),
    "rolling_max_30min": (6, "max"),
}
ROLLING_COLUMNS = list(ROLLING_SPECS)

TREND_STEPS = {"speed_change_5min": 1, "speed_change_15min": 3, "speed_change_30min": 6}
TREND_COLUMNS = list(TREND_STEPS)

TEMPORAL_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "is_morning_peak",
    "is_evening_peak",
]

MISSINGNESS_COLUMNS = ["speed_missing_flag", "missing_count_60min"]

ALL_FEATURE_COLUMNS = LAG_COLUMNS + ROLLING_COLUMNS + TREND_COLUMNS + TEMPORAL_COLUMNS + MISSINGNESS_COLUMNS

# horizon in steps -> name used in column names
TARGET_HORIZONS = {1: "5min", 3: "15min", 6: "30min", 12: "60min"}
TARGET_COLUMNS = [f"target_{h}" for h in TARGET_HORIZONS.values()]
TARGET_VALID_COLUMNS = [f"target_valid_{h}" for h in TARGET_HORIZONS.values()]


def compute_lag_features(speed_df: pd.DataFrame, lags=range(12)) -> dict[str, pd.DataFrame]:
    # shift(k) puts the value from row t-k on row t
    return {f"speed_lag_{k}": speed_df.shift(k) for k in lags}


def compute_rolling_features(speed_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for name, (window, how) in ROLLING_SPECS.items():
        roll = speed_df.rolling(window=window, min_periods=1)
        if how == "std":
            # population std so a single value gives 0 instead of NaN
            out[name] = roll.std(ddof=0).fillna(0.0)
        else:
            out[name] = getattr(roll, how)()
    return out


def compute_trend_features(speed_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {name: speed_df - speed_df.shift(k) for name, k in TREND_STEPS.items()}


def compute_temporal_features(
    timestamps: pd.DatetimeIndex,
    morning_peak_range: tuple[int, int] = (7, 10),
    evening_peak_range: tuple[int, int] = (16, 19),
) -> pd.DataFrame:
    """Calendar features on the stored clock (timezone is unverified, see Person A).

    Peaks follow the shared definition: weekdays only, [07:00, 10:00) and
    [16:00, 19:00).
    """
    hour = timestamps.hour + timestamps.minute / 60.0
    dow = timestamps.dayofweek  # Monday = 0
    weekday = dow < 5

    def in_hours(lo, hi):
        return (weekday & (timestamps.hour >= lo) & (timestamps.hour < hi)).astype(np.int8)

    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24).astype(np.float32),
            "hour_cos": np.cos(2 * np.pi * hour / 24).astype(np.float32),
            "dow_sin": np.sin(2 * np.pi * dow / 7).astype(np.float32),
            "dow_cos": np.cos(2 * np.pi * dow / 7).astype(np.float32),
            "is_weekend": (dow >= 5).astype(np.int8),
            "is_morning_peak": in_hours(*morning_peak_range),
            "is_evening_peak": in_hours(*evening_peak_range),
        },
        index=timestamps,
    )


def compute_missingness_features(raw_mask: pd.DataFrame, history_window: int = 12) -> dict[str, pd.DataFrame]:
    """Built from the raw mask, so the model can tell a real reading from a forward-filled one."""
    flag = raw_mask.astype(np.int8)
    count = flag.rolling(window=history_window, min_periods=1).sum().astype(np.int8)
    return {"speed_missing_flag": flag, "missing_count_60min": count}


def extract_targets(
    raw_df: pd.DataFrame,
    horizons: dict[int, str] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Raw future speed for each horizon and whether it's a usable target.

    Pass the RAW speed table here, not the imputed one. A target is valid
    only if it is finite and > 0, zeros stay in the column but are marked
    invalid so C/D/E can filter them out.
    """
    horizons = horizons or TARGET_HORIZONS
    values, valids = {}, {}
    for step, name in horizons.items():
        future = raw_df.shift(-step)  # row t gets the value from t+step
        values[f"target_{name}"] = future
        valids[f"target_valid_{name}"] = (future > 0.0) & np.isfinite(future)
    return values, valids
