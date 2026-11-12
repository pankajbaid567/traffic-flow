"""Scoring and metric generation for Phase 2 evaluation (Member 4).

Contract: reports/phase2_plan.md section 4.8, 8.4.
- Pooled errors (sum of errors / valid count).
- metrics.csv: per-model and per-seed rows.
- metrics_summary.csv: mean and sd across seeds, params and train seconds.
- patch_models context manager safely wrapping Phase 1's score_frame.
- Phase 1 reproduction verification (within 1e-3).
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.models import prediction_evaluation
from src.phase2.common import ROOT

METADATA_DIR = ROOT / "data" / "metadata"
PHASE1_METRICS_PATH = METADATA_DIR / "phase1_metrics.csv"


@contextmanager
def patch_models(models_map: Mapping[str, str]) -> Iterator[None]:
    """Temporarily patch prediction_evaluation.MODELS during scoring."""
    original = prediction_evaluation.MODELS.copy()
    try:
        prediction_evaluation.MODELS = dict(models_map)
        yield
    finally:
        prediction_evaluation.MODELS = original


def score_dataframe(
    df: pd.DataFrame,
    split: str,
    models_map: Mapping[str, str],
    training_free_speed: Optional[Mapping[str, float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score a DataFrame using Phase 1's score_frame with patched model mapping.

    Returns:
      overall_df: summary rows [split, horizon, model, period, n, mae, rmse, mape]
      sensors_df: per-sensor rows
      daily_df: per-day sufficient statistics for bootstrapping
    """
    with patch_models(models_map):
        overall, sensors, daily = prediction_evaluation.score_frame(
            df, split=split, training_free_speed=training_free_speed
        )
    return overall, sensors, daily


def format_metrics_table(
    overall_df: pd.DataFrame,
    model_metadata_map: Optional[Mapping[str, Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Format overall scoring output into the exact schema for metrics.csv.

    Columns: split,horizon,model,seed,period,n,mae,rmse,mape
    """
    df = overall_df.copy()
    model_metadata_map = model_metadata_map or {}

    models = []
    seeds = []
    for m in df["model"]:
        meta = model_metadata_map.get(m, {})
        base_model = meta.get("model_id", m)
        seed_val = meta.get("seed", "")
        models.append(base_model)
        seeds.append(seed_val)

    df["model"] = models
    df["seed"] = seeds

    cols = ["split", "horizon", "model", "seed", "period", "n", "mae", "rmse", "mape"]
    return df[cols]


def compute_metrics_summary(
    metrics_df: pd.DataFrame,
    meta_stats: Optional[Mapping[Tuple[str, Any], Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """Aggregate per-seed rows into metrics_summary.csv (plan 4.8).

    Columns:
      split,horizon,model,period,n_seeds,n,
      mae_mean,mae_sd,rmse_mean,rmse_sd,mape_mean,mape_sd,
      params,train_seconds_mean
    """
    meta_stats = meta_stats or {}
    group_cols = ["split", "horizon", "model", "period"]
    records = []

    for (split, horizon, model, period), group in metrics_df.groupby(group_cols, sort=False):
        n_seeds = len(group)
        n_count = int(group["n"].iloc[0])

        maes = group["mae"].to_numpy(dtype=float)
        rmses = group["rmse"].to_numpy(dtype=float)
        mapes = group["mape"].to_numpy(dtype=float)

        mae_mean = float(np.mean(maes))
        mae_sd = float(np.std(maes, ddof=1)) if n_seeds > 1 else 0.0

        rmse_mean = float(np.mean(rmses))
        rmse_sd = float(np.std(rmses, ddof=1)) if n_seeds > 1 else 0.0

        mape_mean = float(np.mean(mapes))
        mape_sd = float(np.std(mapes, ddof=1)) if n_seeds > 1 else 0.0

        # Retrieve param count and training seconds from metadata if available
        meta = meta_stats.get((model, split), {})
        params = meta.get("params", np.nan)
        train_sec = meta.get("train_seconds_mean", np.nan)

        records.append({
            "split": split,
            "horizon": horizon,
            "model": model,
            "period": period,
            "n_seeds": n_seeds,
            "n": n_count,
            "mae_mean": mae_mean,
            "mae_sd": mae_sd,
            "rmse_mean": rmse_mean,
            "rmse_sd": rmse_sd,
            "mape_mean": mape_mean,
            "mape_sd": mape_sd,
            "params": params,
            "train_seconds_mean": train_sec,
        })

    return pd.DataFrame(records)


def verify_phase1_reproduction(
    computed_metrics_df: pd.DataFrame,
    expected_csv_path: Path | str = PHASE1_METRICS_PATH,
    tol: float = 1e-3,
) -> Dict[str, float]:
    """Verify that re-scoring Phase 1 models matches phase1_metrics.csv within tol.

    Returns dict of maximum absolute deviations: {'mae': max_diff, 'rmse': max_diff, 'mape': max_diff}.
    Raises AssertionError if any row differs by more than tol.
    """
    expected_path = Path(expected_csv_path)
    if not expected_path.is_file():
        raise FileNotFoundError(f"Expected metrics file not found: {expected_path}")

    expected = pd.read_csv(expected_path)
    computed = computed_metrics_df.copy()

    # Filter computed to Phase 1 models only if other models are present
    phase1_models = [
        "persistence",
        "historical_average",
        "random_forest_32",
        "xgboost_32",
        "xgboost_ha",
    ]
    computed = computed[computed["model"].isin(phase1_models)]

    keys = ["split", "horizon", "model", "period"]
    merged = expected.merge(
        computed, on=keys, suffixes=("_expected", "_computed"), how="inner"
    )

    if len(merged) != len(expected):
        missing = set(zip(*[expected[k] for k in keys])) - set(zip(*[merged[k] for k in keys]))
        raise AssertionError(
            f"Reproduction missing {len(expected) - len(merged)} expected rows: {missing}"
        )

    # Check counts match exactly
    if not (merged["n_expected"] == merged["n_computed"]).all():
        mismatched = merged[merged["n_expected"] != merged["n_computed"]]
        raise AssertionError(f"Valid target count mismatch in rows:\n{mismatched[keys]}")

    max_diffs = {}
    for metric in ["mae", "rmse", "mape"]:
        exp = merged[f"{metric}_expected"].to_numpy(dtype=float)
        act = merged[f"{metric}_computed"].to_numpy(dtype=float)
        diff = np.abs(exp - act)
        max_d = float(np.max(diff))
        max_diffs[metric] = max_d
        if max_d > tol:
            worst_idx = np.argmax(diff)
            worst_row = merged.iloc[worst_idx]
            raise AssertionError(
                f"Metric {metric} deviation {max_d:.6f} exceeds tolerance {tol} "
                f"at {worst_row[keys].to_dict()}: expected {worst_row[f'{metric}_expected']}, computed {worst_row[f'{metric}_computed']}"
            )

    return max_diffs
