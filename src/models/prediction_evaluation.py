"""Independent scoring from aligned forecast rows, with spatial and daily evidence.

No metric is copied from a previous report. All models are scored on exactly the
same observed targets and forecast origins. Geographic regions are descriptive
coordinate bins, not verified freeway identities or connectivity.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from src.models.metrics import MAPE_EPSILON

HORIZONS = {"5min": 1, "15min": 3, "30min": 6, "60min": 12}
MODELS = {
    "persistence": "persistence",
    "historical_average": "historical_average",
    "random_forest_32": "random_forest",
    "xgboost_32": "xgboost",
    "xgboost_ha": "xgboost_ha",
}


def temporal_masks(timestamps):
    t = pd.DatetimeIndex(timestamps)
    morning = (t.dayofweek < 5) & (t.hour >= 7) & (t.hour < 10)
    evening = (t.dayofweek < 5) & (t.hour >= 16) & (t.hour < 19)
    return {
        "all": np.ones(len(t), dtype=bool),
        "peak": morning | evening,
        "off_peak": ~(morning | evening),
        "morning_peak": morning,
        "evening_peak": evening,
    }


def align_predictions(c, d, source=None):
    """Reject row, target, flag or prediction corruption before any scoring."""
    if len(c) != len(d) or c.duplicated(["origin_timestamp", "sensor_id"]).any():
        raise ValueError("Prediction rows must be unique and equally sized")
    for name in ["origin_timestamp", "sensor_id"]:
        if not np.array_equal(
            c[name].astype(str).to_numpy(), d[name].astype(str).to_numpy()
        ):
            raise ValueError(f"Prediction row alignment mismatch: {name}")
        if source is not None and not np.array_equal(
            c[name].astype(str).to_numpy(), source[name].astype(str).to_numpy()
        ):
            raise ValueError(f"Prediction rows differ from shared partition: {name}")
    out = c.copy()
    for h in HORIZONS:
        for name in [f"target_{h}", f"target_valid_{h}"]:
            if not np.array_equal(
                c[name].to_numpy(), d[name].to_numpy(), equal_nan=True
            ):
                raise ValueError(f"Prediction target alignment mismatch: {name}")
            if source is not None and not np.array_equal(
                c[name].to_numpy(), source[name].to_numpy(), equal_nan=True
            ):
                raise ValueError(
                    f"Prediction targets differ from shared partition: {name}"
                )
        y = c[f"target_{h}"].to_numpy(dtype=float)
        valid = c[f"target_valid_{h}"].to_numpy(dtype=bool)
        if not np.array_equal(valid, np.isfinite(y) & (y > 0)):
            raise ValueError(f"Invalid target-validity flags for {h}")
        out[f"pred_random_forest_{h}"] = d[f"pred_random_forest_{h}"].to_numpy()
        for prefix in MODELS.values():
            if not np.isfinite(out[f"pred_{prefix}_{h}"].to_numpy()).all():
                raise ValueError(f"Nonfinite prediction for {prefix} {h}")
    masks = temporal_masks(out.origin_timestamp)
    for flag, period in [
        ("is_morning_peak", "morning_peak"),
        ("is_evening_peak", "evening_peak"),
    ]:
        if not np.array_equal(out[flag].astype(bool).to_numpy(), masks[period]):
            raise ValueError("Saved peak flags differ from stored-clock origin times")
    return out


def geographic_regions(sensors):
    """Deterministic location groups, explicitly not inferred freeway names."""
    out = sensors.copy()
    lat, lon = out.latitude, out.longitude
    out["region"] = np.select(
        [
            lon < -118.43,
            lat > 34.18,
            lat.between(34.09, 34.18) & lon.between(-118.40, -118.28),
            lat.between(34.02, 34.08) & lon.between(-118.43, -118.26),
            lat.between(34.03, 34.09) & lon.gt(-118.26) & lon.le(-118.22),
            lat.lt(34.08) & lon.gt(-118.22),
        ],
        [
            "Western",
            "Northern",
            "North central",
            "Central western",
            "Central eastern",
            "Southeastern",
        ],
        default="Remaining locations",
    )
    return out


def score_frame(df, split, training_free_speed=None):
    """Return overall, per-sensor and per-day sufficient statistics."""
    masks = temporal_masks(df.origin_timestamp)
    ids = sorted(df.sensor_id.astype(str).unique())
    codes = pd.Categorical(df.sensor_id.astype(str), categories=ids).codes
    dates = pd.DatetimeIndex(df.origin_timestamp).normalize()
    days = pd.DatetimeIndex(sorted(dates.unique()))
    day_codes = days.get_indexer(dates)
    rows = []
    sensors = []
    daily = []
    for h in HORIZONS:
        valid = df[f"target_valid_{h}"].to_numpy(bool)
        y = df[f"target_{h}"].to_numpy(float)
        for model, prefix in MODELS.items():
            p = df[f"pred_{prefix}_{h}"].to_numpy(float)
            err = np.abs(y - p)
            for period, pmask in masks.items():
                m = valid & pmask
                count = int(m.sum())
                if not count:
                    rows.append(
                        dict(
                            split=split,
                            horizon=h,
                            model=model,
                            period=period,
                            n=0,
                            mae=None,
                            rmse=None,
                            mape=None,
                        )
                    )
                    continue
                rows.append(
                    dict(
                        split=split,
                        horizon=h,
                        model=model,
                        period=period,
                        n=count,
                        mae=float(err[m].mean()),
                        rmse=float(np.sqrt(np.square(err[m]).mean())),
                        mape=float(
                            (err[m] / np.maximum(y[m], MAPE_EPSILON)).mean() * 100
                        ),
                    )
                )
                counts = np.bincount(codes[m], minlength=len(ids))
                sums = {
                    "absolute_error_sum": np.bincount(
                        codes[m], weights=err[m], minlength=len(ids)
                    ),
                    "squared_error_sum": np.bincount(
                        codes[m], weights=np.square(err[m]), minlength=len(ids)
                    ),
                    "percentage_error_sum": np.bincount(
                        codes[m],
                        weights=err[m] / np.maximum(y[m], MAPE_EPSILON) * 100,
                        minlength=len(ids),
                    ),
                    "actual_sum": np.bincount(
                        codes[m], weights=y[m], minlength=len(ids)
                    ),
                    "predicted_sum": np.bincount(
                        codes[m], weights=p[m], minlength=len(ids)
                    ),
                }
                if training_free_speed is not None:
                    threshold = np.array([0.6 * training_free_speed[s] for s in ids])[
                        codes[m]
                    ]
                    sums["actual_congested_count"] = np.bincount(
                        codes[m], weights=y[m] < threshold, minlength=len(ids)
                    )
                    sums["predicted_congested_count"] = np.bincount(
                        codes[m], weights=p[m] < threshold, minlength=len(ids)
                    )
                for i in np.flatnonzero(counts):
                    n = int(counts[i])
                    rec = dict(
                        split=split,
                        horizon=h,
                        model=model,
                        period=period,
                        sensor_id=ids[i],
                        n=n,
                    )
                    rec.update({k: float(v[i]) for k, v in sums.items()})
                    rec.update(
                        mae=sums["absolute_error_sum"][i] / n,
                        rmse=np.sqrt(sums["squared_error_sum"][i] / n),
                        mape=sums["percentage_error_sum"][i] / n,
                        actual_mean=sums["actual_sum"][i] / n,
                        predicted_mean=sums["predicted_sum"][i] / n,
                    )
                    if training_free_speed is not None:
                        rec.update(
                            actual_congestion_rate=sums["actual_congested_count"][i]
                            / n,
                            predicted_congestion_rate=sums["predicted_congested_count"][
                                i
                            ]
                            / n,
                        )
                    sensors.append(rec)
            counts = np.bincount(day_codes[valid], minlength=len(days))
            sums = np.bincount(
                day_codes[valid], weights=err[valid], minlength=len(days)
            )
            for i in range(len(days)):
                daily.append(
                    dict(
                        split=split,
                        horizon=h,
                        model=model,
                        day=str(days[i].date()),
                        n=int(counts[i]),
                        absolute_error_sum=float(sums[i]),
                    )
                )
    return pd.DataFrame(rows), pd.DataFrame(sensors), pd.DataFrame(daily)


def paired_daily_bootstrap(
    daily, baseline, candidate, horizon, repetitions=2000, seed=42, block_days=2
):
    """Paired moving-day-block bootstrap; all sensors within each day stay together.

    Positive difference means lower MAE for the candidate. This estimates
    uncertainty on this held-out period, not model-selection-adjusted evidence.
    """
    q = daily[(daily.split == "test") & (daily.horizon == horizon)]
    b = q[q.model == baseline].set_index("day").sort_index()
    c = q[q.model == candidate].set_index("day").sort_index()
    if not b.index.equals(c.index) or not np.array_equal(b.n, c.n):
        raise ValueError(
            "Paired bootstrap requires identical days and valid-target counts"
        )
    if len(b) < 2:
        raise ValueError("At least two observed days are required")
    difference = (b.absolute_error_sum - c.absolute_error_sum).to_numpy()
    weights = b.n.to_numpy()
    rng = np.random.default_rng(seed)
    starts = rng.integers(
        0, len(b), size=(repetitions, int(np.ceil(len(b) / block_days)))
    )
    indexes = ((starts[:, :, None] + np.arange(block_days)) % len(b)).reshape(
        repetitions, -1
    )[:, : len(b)]
    denominator = weights[indexes].sum(axis=1)
    observed = denominator > 0
    draws = difference[indexes][observed].sum(axis=1) / denominator[observed]
    if len(draws) < min(100, repetitions):
        raise ValueError("Too few nonempty bootstrap samples")
    low, high = np.quantile(draws, [0.025, 0.975])
    return dict(
        baseline=baseline,
        candidate=candidate,
        horizon=horizon,
        days=len(b),
        block_days=block_days,
        repetitions=repetitions,
        nonempty_repetitions=len(draws),
        seed=seed,
        mae_difference=float(difference.sum() / weights.sum()),
        ci95_low=float(low),
        ci95_high=float(high),
    )


def region_scores(sensor_scores, sensor_metadata):
    meta = geographic_regions(sensor_metadata)[["sensor_id", "region"]].copy()
    meta.sensor_id = meta.sensor_id.astype(str)
    merged = sensor_scores.merge(meta, on="sensor_id", validate="many_to_one")
    keys = ["split", "horizon", "model", "period", "region"]
    sums = [
        "n",
        "absolute_error_sum",
        "squared_error_sum",
        "percentage_error_sum",
        "actual_sum",
        "predicted_sum",
    ]
    agg = merged.groupby(keys, sort=False)[sums].sum().reset_index()
    agg["mae"] = agg.absolute_error_sum / agg.n
    agg["rmse"] = np.sqrt(agg.squared_error_sum / agg.n)
    agg["mape"] = agg.percentage_error_sum / agg.n
    agg["actual_mean"] = agg.actual_sum / agg.n
    agg["predicted_mean"] = agg.predicted_sum / agg.n
    return agg
