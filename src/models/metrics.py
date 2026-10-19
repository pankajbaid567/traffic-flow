"""Evaluation metrics matching DCRNN benchmark and Person E standards.

Masked MAE, RMSE, and MAPE exclude null values (zeros or invalid values)
consistent with Li et al. (ICLR 2018, DCRNN benchmark).

Person E owns the unified evaluation suite, statistical comparison tools,
and error distribution calculations across forecast horizons and temporal periods.
"""
from __future__ import annotations

from typing import Any
import numpy as np


def masked_mae(y_true: np.ndarray, y_pred: np.ndarray, null_val: float = 0.0) -> float:
    """Compute Mean Absolute Error excluding null_val."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if np.isnan(null_val):
        mask = ~np.isnan(y_true)
    else:
        mask = (~np.isnan(y_true)) & (y_true != null_val)
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def masked_rmse(y_true: np.ndarray, y_pred: np.ndarray, null_val: float = 0.0) -> float:
    """Compute Root Mean Squared Error excluding null_val."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if np.isnan(null_val):
        mask = ~np.isnan(y_true)
    else:
        mask = (~np.isnan(y_true)) & (y_true != null_val)
    if not np.any(mask):
        return 0.0
    return float(np.sqrt(np.mean(np.square(y_true[mask] - y_pred[mask]))))


def masked_mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    null_val: float = 0.0,
    epsilon: float = 1e-5,
) -> float:
    """Compute Mean Absolute Percentage Error (as percentage, e.g. 7.5 for 7.5%) excluding null_val.

    Excludes targets where y_true <= 0 or equal to null_val to avoid division by zero or negative speed anomalies.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if np.isnan(null_val):
        mask = ~np.isnan(y_true) & (y_true > epsilon)
    else:
        mask = (~np.isnan(y_true)) & (y_true != null_val) & (y_true > epsilon)
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    null_val: float = 0.0,
) -> dict[str, float]:
    """Compute MAE, RMSE, and MAPE."""
    return {
        "mae": round(masked_mae(y_true, y_pred, null_val), 4),
        "rmse": round(masked_rmse(y_true, y_pred, null_val), 4),
        "mape": round(masked_mape(y_true, y_pred, null_val), 4),
    }


def compute_relative_gain(baseline_score: float, model_score: float) -> float:
    """Compute relative percentage reduction in error (higher is better).

    E.g., baseline MAE 5.0 -> model MAE 4.0 yields +20.0% gain.
    """
    if baseline_score == 0.0:
        return 0.0
    return round(((baseline_score - model_score) / baseline_score) * 100.0, 2)


def error_distribution_quantiles(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    null_val: float = 0.0,
    quantiles: tuple[float, ...] = (0.50, 0.75, 0.90, 0.95, 0.99),
) -> dict[str, float]:
    """Compute percentiles of absolute errors |y_true - y_pred| on valid readings."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if np.isnan(null_val):
        mask = ~np.isnan(y_true)
    else:
        mask = (~np.isnan(y_true)) & (y_true != null_val)
    if not np.any(mask):
        return {f"p{int(q*100)}": 0.0 for q in quantiles}
    abs_err = np.abs(y_true[mask] - y_pred[mask])
    q_vals = np.quantile(abs_err, quantiles)
    return {f"p{int(q*100)}": round(float(val), 4) for q, val in zip(quantiles, q_vals)}
