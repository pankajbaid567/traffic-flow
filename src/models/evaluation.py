"""Core evaluation and benchmark synthesis engine for Person E.

Ingests results from Person C (baselines + XGBoost) and Person D (Random Forest),
verifies metric alignment, and compiles unified cross-model benchmark tables,
temporal slice breakdowns, and horizon degradation dynamics.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
METADATA_DIR = ROOT / "data" / "metadata"
PERSON_C_JSON = METADATA_DIR / "person_c_results.json"
PERSON_D_JSON = METADATA_DIR / "person_d_results.json"


def load_person_c_results(path: Path = PERSON_C_JSON) -> dict[str, Any]:
    """Load and parse Person C benchmark results JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Person C results not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_person_d_results(path: Path = PERSON_D_JSON) -> dict[str, Any]:
    """Load and parse Person D benchmark results JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Person D results not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_unified_test_benchmark(
    c_data: dict[str, Any] | None = None,
    d_data: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build unified test set benchmark table across all models and horizons.

    Models included:
      - persistence (Person C)
      - historical_average (Person C)
      - random_forest (Person D, Exp D full 32 features)
      - xgboost_32 (Person C, 32 shared features)
      - xgboost_ha (Person C, 32 features + 4 HA features)
    """
    if c_data is None:
        c_data = load_person_c_results()
    if d_data is None:
        d_data = load_person_d_results()

    horizons = ["5min", "15min", "30min", "60min"]
    records = []

    for h in horizons:
        c_h_test = c_data["horizons"][h]["test"]
        d_exp_d = d_data["experiments"]["D"][h]["test_metrics"]

        # Persistence
        records.append({
            "horizon": h,
            "model": "persistence",
            "model_type": "Naive Baseline",
            "features": "speed_lag_0",
            "mae": c_h_test["persistence"]["all"]["mae"],
            "rmse": c_h_test["persistence"]["all"]["rmse"],
            "mape": c_h_test["persistence"]["all"]["mape"],
            "n_samples": c_h_test["persistence"]["all"]["n"],
        })
        # Historical Average
        records.append({
            "horizon": h,
            "model": "historical_average",
            "model_type": "Lookup Baseline",
            "features": "time_of_week_lookup",
            "mae": c_h_test["historical_average"]["all"]["mae"],
            "rmse": c_h_test["historical_average"]["all"]["rmse"],
            "mape": c_h_test["historical_average"]["all"]["mape"],
            "n_samples": c_h_test["historical_average"]["all"]["n"],
        })
        # Random Forest (Exp D)
        records.append({
            "horizon": h,
            "model": "random_forest_32",
            "model_type": "Applied ML (Bagging)",
            "features": "32_shared",
            "mae": d_exp_d["mae"],
            "rmse": d_exp_d["rmse"],
            "mape": d_exp_d["mape"],
            "n_samples": c_h_test["xgboost"]["all"]["n"],  # Same evaluation test partition
        })
        # XGBoost (32 shared features)
        records.append({
            "horizon": h,
            "model": "xgboost_32",
            "model_type": "Applied ML (Boosting)",
            "features": "32_shared",
            "mae": c_h_test["xgboost"]["all"]["mae"],
            "rmse": c_h_test["xgboost"]["all"]["rmse"],
            "mape": c_h_test["xgboost"]["all"]["mape"],
            "n_samples": c_h_test["xgboost"]["all"]["n"],
        })
        # XGBoost + HA (36 features)
        records.append({
            "horizon": h,
            "model": "xgboost_ha",
            "model_type": "Augmented Applied ML",
            "features": "32_shared + 4_ha",
            "mae": c_h_test["xgboost_ha"]["all"]["mae"],
            "rmse": c_h_test["xgboost_ha"]["all"]["rmse"],
            "mape": c_h_test["xgboost_ha"]["all"]["mape"],
            "n_samples": c_h_test["xgboost_ha"]["all"]["n"],
        })

    return pd.DataFrame(records)


def build_peak_offpeak_breakdown(c_data: dict[str, Any] | None = None) -> pd.DataFrame:
    """Extract temporal slice error metrics across all horizons for test set."""
    if c_data is None:
        c_data = load_person_c_results()

    horizons = ["5min", "15min", "30min", "60min"]
    periods = ["all", "peak", "off_peak", "morning_peak", "evening_peak"]
    models = ["persistence", "historical_average", "xgboost", "xgboost_ha"]
    records = []

    for h in horizons:
        for period in periods:
            rec: dict[str, Any] = {"horizon": h, "period": period}
            rec["rows"] = c_data["horizons"][h]["test"]["xgboost_ha"][period]["n"]
            for m in models:
                res = c_data["horizons"][h]["test"][m][period]
                rec[f"{m}_mae"] = res["mae"]
                rec[f"{m}_rmse"] = res["rmse"]
                rec[f"{m}_mape"] = res["mape"]
            records.append(rec)

    return pd.DataFrame(records)


def build_rf_ablation_summary(d_data: dict[str, Any] | None = None) -> pd.DataFrame:
    """Compile Person D's 4 feature ablation experiments (Exp A through D)."""
    if d_data is None:
        d_data = load_person_d_results()

    horizons = ["5min", "15min", "30min", "60min"]
    records = []
    exp_names = {
        "A": "Exp A (12 Lags)",
        "B": "Exp B (12 Lags + 7 Temporal)",
        "C": "Exp C (12 Lags + 8 Rolling + 3 Trend)",
        "D": "Exp D (Full 32 Features)",
    }

    for h in horizons:
        for code, label in exp_names.items():
            exp = d_data["experiments"][code][h]
            val_m = exp["val_metrics"]
            test_m = exp.get("test_metrics", {})
            records.append({
                "horizon": h,
                "experiment": code,
                "label": label,
                "n_features": exp["n_features"],
                "fit_time_s": exp["fit_time_seconds"],
                "val_mae": val_m["mae"],
                "val_rmse": val_m["rmse"],
                "val_mape": val_m["mape"],
                "val_gain_vs_pers": exp["val_improvement_mae"],
                "test_mae": test_m.get("mae"),
                "test_rmse": test_m.get("rmse"),
                "test_mape": test_m.get("mape"),
            })

    return pd.DataFrame(records)


def compute_horizon_degradation(df_test: pd.DataFrame) -> pd.DataFrame:
    """Calculate error escalation ratio from 5min to 60min: MAE(60m) / MAE(5m)."""
    piv = df_test.pivot(index="model", columns="horizon", values="mae")
    piv["degradation_ratio_60_to_5"] = round(piv["60min"] / piv["5min"], 3)
    piv["mae_increase_absolute"] = round(piv["60min"] - piv["5min"], 3)
    return piv.reset_index()
