"""XGBoost forecasters and period-wise evaluation (Person C).

One model per horizon, trained on the 32 local features. Two choices came out of
validation-only experiments (see reports/person_c_report.md):

  - the model predicts the CHANGE from the current speed (target - speed_lag_0)
    and the current speed is added back afterwards
  - the loss is absolute error, because MAE is the headline metric

Only the training split is used to fit; the validation split decides the number
of trees (early stopping). Test is scored once at the end.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError:
    xgb = None

from src.models.metrics import compute_metrics

DEFAULT_PARAMS: dict[str, Any] = {
    "tree_method": "hist",
    "objective": "reg:absoluteerror",
    "eval_metric": "mae",
    "max_depth": 8,
    "eta": 0.1,
    "min_child_weight": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
}

BASE_COLUMN = "speed_lag_0"


def _xy(
    df: pd.DataFrame, features: list[str], horizon: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Valid rows only: features, target, and the current speed used as the base."""
    valid = df[f"target_valid_{horizon}"].to_numpy()
    X = df.loc[valid, features].to_numpy(np.float32)
    y = df.loc[valid, f"target_{horizon}"].to_numpy(np.float64)
    base = df.loc[valid, BASE_COLUMN].to_numpy(np.float64)
    return X, y, base


def train_horizon(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str],
    horizon: str,
    params: dict[str, Any] | None = None,
    max_rounds: int = 2000,
    early_stopping_rounds: int = 50,
    n_jobs: int = 4,
) -> tuple[xgb.Booster, dict[str, Any]]:
    if xgb is None:
        raise ImportError(
            "XGBoost training requires the xgboost package. Saved-model inference also supports the offline numeric JSON reader."
        )
    params = {
        **DEFAULT_PARAMS,
        **(params or {}),
        "nthread": n_jobs if n_jobs > 0 else _all_cores(),
    }

    X_tr, y_tr, b_tr = _xy(train_df, features, horizon)
    X_va, y_va, b_va = _xy(val_df, features, horizon)
    d_tr = xgb.DMatrix(X_tr, y_tr - b_tr, feature_names=features)
    d_va = xgb.DMatrix(X_va, y_va - b_va, feature_names=features)

    t0 = time.time()
    booster = xgb.train(
        params,
        d_tr,
        max_rounds,
        evals=[(d_va, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )
    info = {
        "horizon": horizon,
        "fit_time_seconds": round(time.time() - t0, 1),
        "train_samples": int(len(y_tr)),
        "best_iteration": int(booster.best_iteration) + 1,
        "params": {k: v for k, v in params.items() if k != "nthread"},
    }
    return booster, info


def predict(booster: xgb.Booster, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Predicted speed for EVERY row of df (validity is applied when scoring)."""
    if hasattr(booster, "predict_frame"):
        return booster.predict_frame(df, features)
    d = xgb.DMatrix(df[features].to_numpy(np.float32), feature_names=features)
    delta = booster.predict(d, iteration_range=(0, booster.best_iteration + 1))
    return df[BASE_COLUMN].to_numpy(np.float64) + delta


def _all_cores() -> int:
    import os

    return os.cpu_count() or 1


# ---- period-wise evaluation -------------------------------------------------


def period_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Peak / off-peak groups from the origin time, using Person B's peak flags.

    Peak = weekday 07:00-10:00 or 16:00-19:00. Off-peak is everything else,
    including weekends.
    """
    morning = df["is_morning_peak"].to_numpy().astype(bool)
    evening = df["is_evening_peak"].to_numpy().astype(bool)
    peak = morning | evening
    return {
        "all": np.ones(len(df), dtype=bool),
        "peak": peak,
        "off_peak": ~peak,
        "morning_peak": morning,
        "evening_peak": evening,
    }


def score_by_period(
    df: pd.DataFrame, horizon: str, preds: dict[str, np.ndarray]
) -> dict[str, dict[str, dict[str, float]]]:
    """{model: {period: {mae, rmse, mape, n}}} on valid targets only."""
    valid = df[f"target_valid_{horizon}"].to_numpy()
    y = df[f"target_{horizon}"].to_numpy(np.float64)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for period, pmask in period_masks(df).items():
        m = valid & pmask
        for name, p in preds.items():
            res = compute_metrics(y[m], np.asarray(p)[m])
            res["n"] = int(m.sum())
            out.setdefault(name, {})[period] = res
    return out
