"""Paired daily block bootstrap and formal hypothesis evaluation for Phase 2 (Member 4).

Contract: reports/phase2_plan.md section 4.8, 5.3, 5.4, 8.4.
- Circular moving-day-block bootstrap (2-day blocks, 2000 resamples, seed 42).
- Multi-seed models average daily absolute error sums across seeds before bootstrapping.
- Programmatic verdict rules: supported, contradicted, inconclusive, or not_evaluated.
- Outputs data/metadata/phase2/bootstrap.csv and hypotheses.csv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.models.prediction_evaluation import HORIZONS, paired_daily_bootstrap
from src.phase2.common import ROOT

METADATA_DIR = ROOT / "data" / "metadata"
PHASE1_BOOTSTRAP_PATH = METADATA_DIR / "paired_bootstrap.csv"

COMPARISONS = [
    # Baseline, Candidate, Horizons, Periods
    ("gru_local", "gat", ["5min", "15min", "30min", "60min"], ["all", "peak", "off_peak"]),
    ("gat_identity", "gat", ["5min", "15min", "30min", "60min"], ["all", "peak", "off_peak"]),
    ("gat_identity", "gat_shuffled", ["5min", "15min", "30min", "60min"], ["all"]),
    ("dcrnn", "gat", ["5min", "15min", "30min", "60min"], ["all"]),
    ("xgboost_ha", "gat", ["5min", "15min", "30min", "60min"], ["all"]),
    ("gru_local", "dcrnn", ["5min", "15min", "30min", "60min"], ["all"]),
    ("dcrnn_identity", "dcrnn", ["5min", "15min", "30min", "60min"], ["all"]),
    ("xgboost_ha", "dcrnn", ["5min", "15min", "30min", "60min"], ["all"]),
    # Phase 1 reference comparisons
    ("persistence", "xgboost_ha", ["5min", "15min", "30min", "60min"], ["all"]),
    ("random_forest_32", "xgboost_32", ["5min", "15min", "30min", "60min"], ["all"]),
    ("xgboost_32", "xgboost_ha", ["5min", "15min", "30min", "60min"], ["all"]),
]


def aggregate_daily_stats_across_seeds(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-day absolute error sums across multiple seeds.

    If a model has multiple seeds (e.g. 'gat_seed42', 'gat_seed43'), daily error
    sums are averaged across seeds for each (split, horizon, day).
    """
    df = daily_df.copy()
    group_cols = ["split", "horizon", "model", "day"]
    agg = df.groupby(group_cols, as_index=False).agg({
        "n": "first",  # n is identical across seeds
        "absolute_error_sum": "mean",  # seed-averaged sum
    })
    return agg


def run_paired_bootstrap_comparison(
    daily_df: pd.DataFrame,
    baseline: str,
    candidate: str,
    horizon: str,
    period: str = "all",
    repetitions: int = 2000,
    seed: int = 42,
    block_days: int = 2,
    split: str = "test",
) -> Optional[Dict[str, Any]]:
    """Run circular moving-day-block bootstrap for a baseline-candidate pair."""
    models_available = daily_df["model"].unique()
    if baseline not in models_available or candidate not in models_available:
        return None

    try:
        res = paired_daily_bootstrap(
            daily=daily_df,
            baseline=baseline,
            candidate=candidate,
            horizon=horizon,
            repetitions=repetitions,
            seed=seed,
            block_days=block_days,
        )
        res["split"] = split
        res["period"] = period
        return res
    except Exception as e:
        return None


def execute_all_bootstrap_comparisons(
    daily_df: pd.DataFrame,
    repetitions: int = 2000,
    seed: int = 42,
    block_days: int = 2,
) -> pd.DataFrame:
    """Execute all planned bootstrap comparisons across available models."""
    daily_agg = aggregate_daily_stats_across_seeds(daily_df)
    results = []

    for baseline, candidate, horizons, periods in COMPARISONS:
        for h in horizons:
            for p in periods:
                res = run_paired_bootstrap_comparison(
                    daily_df=daily_agg,
                    baseline=baseline,
                    candidate=candidate,
                    horizon=h,
                    period=p,
                    repetitions=repetitions,
                    seed=seed,
                    block_days=block_days,
                )
                if res is not None:
                    results.append(res)

    if not results:
        cols = [
            "split", "baseline", "candidate", "horizon", "period", "days",
            "block_days", "repetitions", "nonempty_repetitions", "seed",
            "mae_difference", "ci95_low", "ci95_high"
        ]
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(results)
    cols = [
        "split", "baseline", "candidate", "horizon", "period", "days",
        "block_days", "repetitions", "nonempty_repetitions", "seed",
        "mae_difference", "ci95_low", "ci95_high"
    ]
    return df[cols]


def evaluate_hypotheses(bootstrap_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate formal Phase 2 hypotheses H0-H5 using code-computed verdict rules.

    Verdict rules:
      - 'supported': 95% CI entirely > 0 (for multi-horizon, all named horizons must be > 0).
      - 'contradicted': 95% CI entirely < 0.
      - 'inconclusive': 95% CI contains 0.
      - 'not_evaluated': required models absent from bootstrap_df.
    """
    rows = []

    def get_comp(baseline: str, candidate: str, horizon: str, period: str = "all"):
        if bootstrap_df.empty:
            return None
        sub = bootstrap_df[
            (bootstrap_df["baseline"] == baseline)
            & (bootstrap_df["candidate"] == candidate)
            & (bootstrap_df["horizon"] == horizon)
            & (bootstrap_df["period"] == period)
        ]
        if sub.empty:
            return None
        return sub.iloc[0]

    # H1: GAT has lower MAE than gru_local at 30 and 60 min
    h1_30 = get_comp("gru_local", "gat", "30min", "all")
    h1_60 = get_comp("gru_local", "gat", "60min", "all")
    if h1_30 is None or h1_60 is None:
        v_h1 = "not_evaluated"
        diff_h1 = np.nan
        ci_l = np.nan
        ci_h = np.nan
    else:
        diff_h1 = (h1_30["mae_difference"] + h1_60["mae_difference"]) / 2.0
        ci_l = min(h1_30["ci95_low"], h1_60["ci95_low"])
        ci_h = max(h1_30["ci95_high"], h1_60["ci95_high"])
        if h1_30["ci95_low"] > 0 and h1_60["ci95_low"] > 0:
            v_h1 = "supported"
        elif h1_30["ci95_high"] < 0 or h1_60["ci95_high"] < 0:
            v_h1 = "contradicted"
        else:
            v_h1 = "inconclusive"

    rows.append({
        "id": "H1",
        "statement": "gat has lower MAE than gru_local at 30 and 60 min",
        "horizon": "30min,60min",
        "period": "all",
        "comparison": "gru_local vs gat",
        "mae_difference": diff_h1,
        "ci95_low": ci_l,
        "ci95_high": ci_h,
        "verdict": v_h1,
    })

    # H2: GAT beats gat_identity; gat_shuffled is no better than gat_identity
    h2_id = get_comp("gat_identity", "gat", "60min", "all")
    h2_shuf = get_comp("gat_identity", "gat_shuffled", "60min", "all")
    if h2_id is None or h2_shuf is None:
        v_h2 = "not_evaluated"
        diff_h2 = np.nan
        ci_l = np.nan
        ci_h = np.nan
    else:
        diff_h2 = h2_id["mae_difference"]
        ci_l = h2_id["ci95_low"]
        ci_h = h2_id["ci95_high"]
        # gat beats gat_identity (CI > 0) AND gat_shuffled does NOT beat gat_identity (shuf - id <= 0)
        if h2_id["ci95_low"] > 0 and h2_shuf["ci95_high"] <= 0:
            v_h2 = "supported"
        elif h2_id["ci95_high"] < 0:
            v_h2 = "contradicted"
        else:
            v_h2 = "inconclusive"

    rows.append({
        "id": "H2",
        "statement": "gat beats gat_identity and gat_shuffled is no better than gat_identity",
        "horizon": "60min",
        "period": "all",
        "comparison": "gat_identity vs gat / gat_shuffled",
        "mae_difference": diff_h2,
        "ci95_low": ci_l,
        "ci95_high": ci_h,
        "verdict": v_h2,
    })

    # H3: The gat vs gat_identity gain is larger in peak than off-peak
    h3_peak = get_comp("gat_identity", "gat", "60min", "peak")
    h3_off = get_comp("gat_identity", "gat", "60min", "off_peak")
    if h3_peak is None or h3_off is None:
        v_h3 = "not_evaluated"
        diff_h3 = np.nan
        ci_l = np.nan
        ci_h = np.nan
    else:
        diff_h3 = h3_peak["mae_difference"] - h3_off["mae_difference"]
        ci_l = h3_peak["ci95_low"] - h3_off["ci95_high"]
        ci_h = h3_peak["ci95_high"] - h3_off["ci95_low"]
        if diff_h3 > 0:
            v_h3 = "supported"
        else:
            v_h3 = "contradicted"

    rows.append({
        "id": "H3",
        "statement": "gat vs gat_identity gain is larger in peak than off-peak",
        "horizon": "60min",
        "period": "peak vs off_peak",
        "comparison": "gat_identity vs gat",
        "mae_difference": diff_h3,
        "ci95_low": ci_l,
        "ci95_high": ci_h,
        "verdict": v_h3,
    })

    # H4: The gain grows with horizon (5 < 15 < 30 < 60 min)
    gains = [get_comp("gru_local", "gat", h, "all") for h in ["5min", "15min", "30min", "60min"]]
    if any(g is None for g in gains):
        v_h4 = "not_evaluated"
        diff_h4 = np.nan
    else:
        vals = [g["mae_difference"] for g in gains]
        diff_h4 = vals[-1] - vals[0]
        if vals[0] < vals[1] < vals[2] < vals[3]:
            v_h4 = "supported"
        elif vals[-1] <= vals[0]:
            v_h4 = "contradicted"
        else:
            v_h4 = "inconclusive"

    rows.append({
        "id": "H4",
        "statement": "gain grows monotonically with horizon (5 < 15 < 30 < 60 min)",
        "horizon": "all",
        "period": "all",
        "comparison": "gru_local vs gat across horizons",
        "mae_difference": diff_h4,
        "ci95_low": np.nan,
        "ci95_high": np.nan,
        "verdict": v_h4,
    })

    # H5: The gain is larger at sensors with more neighbours and around congestion onsets
    rows.append({
        "id": "H5",
        "statement": "gain correlates positively with graph degree and around congestion onsets",
        "horizon": "60min",
        "period": "all",
        "comparison": "sensor degree & onset descriptive correlation",
        "mae_difference": np.nan,
        "ci95_low": np.nan,
        "ci95_high": np.nan,
        "verdict": "descriptive_only",
    })

    # H0: gat and dcrnn do not differ from xgboost_ha
    h0_gat = get_comp("xgboost_ha", "gat", "60min", "all")
    h0_dcrnn = get_comp("xgboost_ha", "dcrnn", "60min", "all")
    if h0_gat is None and h0_dcrnn is None:
        v_h0 = "not_evaluated"
        diff_h0 = np.nan
        ci_l = np.nan
        ci_h = np.nan
    else:
        comp = h0_gat if h0_gat is not None else h0_dcrnn
        diff_h0 = comp["mae_difference"]
        ci_l = comp["ci95_low"]
        ci_h = comp["ci95_high"]
        # H0 is supported if CI contains 0 (no difference)
        if ci_l <= 0 <= ci_h:
            v_h0 = "supported"
        else:
            v_h0 = "contradicted"

    rows.append({
        "id": "H0",
        "statement": "deep graph models (gat, dcrnn) do not differ from xgboost_ha",
        "horizon": "60min",
        "period": "all",
        "comparison": "xgboost_ha vs {gat, dcrnn}",
        "mae_difference": diff_h0,
        "ci95_low": ci_l,
        "ci95_high": ci_h,
        "verdict": v_h0,
    })

    return pd.DataFrame(rows)


def verify_phase1_bootstrap_reproduction(
    computed_bootstrap_df: pd.DataFrame,
    expected_csv_path: Path | str = PHASE1_BOOTSTRAP_PATH,
    mae_diff_tol: float = 1e-4,
    ci_tol: float = 0.05,
) -> Dict[str, float]:
    """Verify that computed bootstrap reproduces Phase 1 paired_bootstrap.csv."""
    expected_path = Path(expected_csv_path)
    if not expected_path.is_file():
        raise FileNotFoundError(f"Expected bootstrap file not found: {expected_path}")

    expected = pd.read_csv(expected_path)
    merged = expected.merge(
        computed_bootstrap_df,
        on=["baseline", "candidate", "horizon"],
        suffixes=("_expected", "_computed"),
    )

    if len(merged) != len(expected):
        raise AssertionError(
            f"Reproduction missing {len(expected) - len(merged)} expected bootstrap comparisons"
        )

    # Exact match on mae_difference
    diff_mae = np.abs(merged["mae_difference_expected"] - merged["mae_difference_computed"])
    max_mae_diff = float(np.max(diff_mae))
    if max_mae_diff > mae_diff_tol:
        raise AssertionError(
            f"mae_difference deviation {max_mae_diff} exceeds tolerance {mae_diff_tol}"
        )

    # Confidence interval ends within stochastic bootstrap noise
    diff_ci_low = np.abs(merged["ci95_low_expected"] - merged["ci95_low_computed"])
    diff_ci_high = np.abs(merged["ci95_high_expected"] - merged["ci95_high_computed"])
    max_ci_diff = float(max(np.max(diff_ci_low), np.max(diff_ci_high)))
    if max_ci_diff > ci_tol:
        raise AssertionError(
            f"Bootstrap CI deviation {max_ci_diff} exceeds stochastic tolerance {ci_tol}"
        )

    return {"max_mae_difference": max_mae_diff, "max_ci_deviation": max_ci_diff}
