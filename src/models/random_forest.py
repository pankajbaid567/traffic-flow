"""Random Forest regression models and ablation runner for traffic speed forecasting (Person D).

Covers:
- Exp A: lag_only (12 lags)
- Exp B: lag_temporal (lags + time)
- Exp C: lag_temporal_rolling (lags + time + rolling stats)
- Exp D: full_features (all 32 engineered features)

Across horizons: 5min, 15min, 30min, 60min.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.models.metrics import compute_metrics

ROOT = Path(__file__).resolve().parents[2]


def load_feature_groups(summary_path: Path | str = ROOT / "data/processed/dataset_summary.json") -> dict[str, list[str]]:
    """Load feature group definitions from dataset_summary.json."""
    summary_path = Path(summary_path)
    if not summary_path.is_file():
        raise FileNotFoundError(f"dataset summary not found: {summary_path}")
    with open(summary_path) as f:
        summary = json.load(f)
    return summary["feature_groups"]


def build_model(
    n_estimators: int = 100,
    max_depth: int = 15,
    max_samples: float = 0.1,
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestRegressor:
    """Instantiate a standardized RandomForestRegressor for Person D ablations."""
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def train_and_evaluate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    features: list[str],
    horizon: str,
    n_estimators: int = 100,
    max_depth: int = 15,
    max_samples: float = 0.1,
    random_state: int = 42,
    evaluate_test: bool = False,
) -> dict[str, Any]:
    """Train a Random Forest model on valid training targets and evaluate on val (and optionally test)."""
    valid_col = f"target_valid_{horizon}"
    target_col = f"target_{horizon}"

    train_mask = train_df[valid_col]
    X_train = train_df.loc[train_mask, features]
    y_train = train_df.loc[train_mask, target_col]

    val_mask = val_df[valid_col]
    X_val = val_df.loc[val_mask, features]
    y_val = val_df.loc[val_mask, target_col]

    model = build_model(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_samples=max_samples,
        random_state=random_state,
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0

    val_preds = model.predict(X_val)
    val_metrics = compute_metrics(y_val.to_numpy(), val_preds)

    # Persistence baseline on validation set for direct comparison
    val_persistence = val_df.loc[val_mask, "speed_lag_0"].to_numpy()
    val_persist_metrics = compute_metrics(y_val.to_numpy(), val_persistence)

    # Feature importances
    importances = dict(
        sorted(
            zip(features, [round(float(x), 5) for x in model.feature_importances_]),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    result = {
        "horizon": horizon,
        "n_features": len(features),
        "fit_time_seconds": round(fit_time, 2),
        "train_samples": int(len(X_train)),
        "val_metrics": val_metrics,
        "val_persistence_mae": val_persist_metrics["mae"],
        "val_improvement_mae": round(val_persist_metrics["mae"] - val_metrics["mae"], 4),
        "feature_importances": importances,
    }

    if evaluate_test and test_df is not None:
        test_mask = test_df[valid_col]
        X_test = test_df.loc[test_mask, features]
        y_test = test_df.loc[test_mask, target_col]
        test_preds = model.predict(X_test)
        test_metrics = compute_metrics(y_test.to_numpy(), test_preds)
        test_persistence = test_df.loc[test_mask, "speed_lag_0"].to_numpy()
        test_persist_metrics = compute_metrics(y_test.to_numpy(), test_persistence)

        result["test_metrics"] = test_metrics
        result["test_persistence_mae"] = test_persist_metrics["mae"]
        result["test_improvement_mae"] = round(test_persist_metrics["mae"] - test_metrics["mae"], 4)

    return result, model
