"""Common Phase 1 metrics on finite, positive, observed targets.

Unavailable targets are excluded; invalid predictions on usable targets raise.
An empty evaluation has NaN metrics, never a misleading perfect zero. MAPE uses
max(target, 1e-5) as a declared denominator floor, retaining positive near-zero
observations in every metric.
"""

from __future__ import annotations
import numpy as np

MAPE_EPSILON = 1e-5


def valid_pairs(y_true, y_pred, null_val=0.0):
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError(
            "Targets and predictions must be equally shaped one-dimensional arrays"
        )
    mask = np.isfinite(y) & (y > 0)
    if not np.isnan(null_val):
        mask &= y != null_val
    if not np.isfinite(p[mask]).all():
        raise ValueError("Predictions must be finite on every valid target")
    return y[mask], p[mask]


def masked_mae(y_true, y_pred, null_val=0.0):
    y, p = valid_pairs(y_true, y_pred, null_val)
    return float(np.abs(y - p).mean()) if len(y) else float("nan")


def masked_rmse(y_true, y_pred, null_val=0.0):
    y, p = valid_pairs(y_true, y_pred, null_val)
    return float(np.sqrt(np.square(y - p).mean())) if len(y) else float("nan")


def masked_mape(y_true, y_pred, null_val=0.0, epsilon=MAPE_EPSILON):
    if epsilon <= 0:
        raise ValueError("MAPE denominator floor must be positive")
    y, p = valid_pairs(y_true, y_pred, null_val)
    return (
        float((np.abs(y - p) / np.maximum(y, epsilon)).mean() * 100)
        if len(y)
        else float("nan")
    )


def compute_metrics(y_true, y_pred, null_val=0.0):
    return {
        "mae": round(masked_mae(y_true, y_pred, null_val), 4),
        "rmse": round(masked_rmse(y_true, y_pred, null_val), 4),
        "mape": round(masked_mape(y_true, y_pred, null_val), 4),
    }


def compute_relative_gain(baseline_score, model_score):
    return (
        round((baseline_score - model_score) / baseline_score * 100, 2)
        if baseline_score
        else float("nan")
    )


def error_distribution_quantiles(
    y_true, y_pred, null_val=0.0, quantiles=(0.50, 0.75, 0.90, 0.95, 0.99)
):
    y, p = valid_pairs(y_true, y_pred, null_val)
    values = (
        np.quantile(np.abs(y - p), quantiles)
        if len(y)
        else [float("nan")] * len(quantiles)
    )
    return {f"p{int(q*100)}": round(float(v), 4) for q, v in zip(quantiles, values)}
