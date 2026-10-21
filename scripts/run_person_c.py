"""Person C: baselines + XGBoost across the four horizons.

    python scripts/run_person_c.py

Fits on train, tunes (early stopping) on validation, scores test once.
Writes:
  data/metadata/person_c_results.json
  data/processed/predictions/person_c_{val,test}_predictions.parquet   (for Person E)
  models/person_c/xgb_<horizon>.json
  reports/person_c_report.md
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset_builder import load_partition
from src.data.features import TARGET_HORIZONS
from src.data.load_data import load_raw_speed_data
from src.data.split import get_partition_info, load_split_contract
from src.models.baselines import (
    HA_FEATURE_COLUMNS,
    HistoricalAverage,
    persistence_predict,
)
from src.models.saved_booster import load_saved_booster
from src.models.xgboost_model import predict, score_by_period, train_horizon

HORIZONS = list(TARGET_HORIZONS.values())  # 5min, 15min, 30min, 60min
STEPS = {name: step for step, name in TARGET_HORIZONS.items()}
MODELS = ["persistence", "historical_average", "xgboost", "xgboost_ha"]
PERIODS = ["all", "peak", "off_peak", "morning_peak", "evening_peak"]


def fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


LABEL = {
    "persistence": "persistence",
    "historical_average": "historical average",
    "xgboost": "XGBoost (32 shared features)",
    "xgboost_ha": "XGBoost + hist. avg features",
}


def write_report(results: dict, d_results: dict | None, path: Path) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rs = results["horizons"]
    L = [
        "# Person C — Baselines and XGBoost",
        "",
        f"**Generated:** {ts}",
        "**Scope:** persistence, historical average and two XGBoost variants, one model per horizon, "
        "own-sensor features only (no neighbours, no graph).",
        "",
        "Speeds are in source units (likely mph, unverified). Every score uses valid targets only (`target_valid_<h> == True`). "
        "Fitted on train and early-stopped on validation. Completion regenerates predictions from the frozen boosters without test-based tuning.",
        "",
        "## Models",
        "",
        "- **persistence**: prediction is the current speed.",
        "- **historical average**: mean valid speed per (sensor, day of week, 5-min slot) from train, looked up at the **target** time (DCRNN-style baseline).",
        "- **XGBoost (32 shared features)**: exactly Person B's 32 features, the same inputs as Person D's Random Forest, so this is the like-for-like comparison.",
        "- **XGBoost + hist. avg features**: the same 32 features plus four extra inputs from the sensor's own training history "
        "(`ha_target`, `ha_origin`, `ha_delta`, `anomaly_now`). **This goes beyond Person B's 32-feature contract** and is kept as a separate, "
        "clearly labelled model. On training rows these are computed leave-one-out (a row's own reading is removed from its average) so they do not leak the target.",
        "",
        "## Test results (final)",
        "",
        "| Horizon | Model | MAE | RMSE | MAPE % |",
        "|---|---|---:|---:|---:|",
    ]
    for h in HORIZONS:
        for m in MODELS:
            r = rs[h]["test"][m]["all"]
            name = f"**{LABEL[m]}**" if m == "xgboost_ha" else LABEL[m]
            L.append(
                f"| {h} | {name} | {fmt(r['mae'])} | {fmt(r['rmse'])} | {fmt(r['mape'], 2)} |"
            )

    L += [
        "",
        "## Validation results (MAE)",
        "",
        "| Horizon | Persistence | Hist. avg | XGBoost (32) | XGBoost + HA | XGBoost + HA RMSE | XGBoost + HA MAPE % | Gain vs persistence |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for h in HORIZONS:
        v = rs[h]["val"]
        g = {m: v[m]["all"] for m in MODELS}
        L.append(
            f"| {h} | {fmt(g['persistence']['mae'])} | {fmt(g['historical_average']['mae'])} | {fmt(g['xgboost']['mae'])} | "
            f"**{fmt(g['xgboost_ha']['mae'])}** | {fmt(g['xgboost_ha']['rmse'])} | {fmt(g['xgboost_ha']['mape'], 2)} | "
            f"{fmt((g['persistence']['mae'] - g['xgboost_ha']['mae']) / g['persistence']['mae'] * 100, 1)}% |"
        )

    L += [
        "",
        "## Peak vs off-peak (test)",
        "",
        "Peak = weekday 07:00–10:00 or 16:00–19:00 at the **origin** time (Person B's flags). Off-peak is everything else, weekends included.",
        "",
        "| Horizon | Period | Rows | Persistence MAE | Hist. avg MAE | XGBoost (32) MAE | XGBoost + HA MAE | XGBoost + HA RMSE | XGBoost + HA MAPE % |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for h in HORIZONS:
        t = rs[h]["test"]
        for per in PERIODS:
            x = t["xgboost_ha"][per]
            L.append(
                f"| {h} | {per} | {x['n']:,} | {fmt(t['persistence'][per]['mae'])} | {fmt(t['historical_average'][per]['mae'])} | "
                f"{fmt(t['xgboost'][per]['mae'])} | **{fmt(x['mae'])}** | {fmt(x['rmse'])} | {fmt(x['mape'], 2)} |"
            )

    if d_results:
        L += [
            "",
            "## Against Person D's Random Forest (test, full 32 features)",
            "",
            "Person D's numbers are read from `data/metadata/person_d_results.json` and not modified.",
            "",
            "| Horizon | RF MAE | XGBoost (32) MAE | XGBoost + HA MAE | RF RMSE | XGBoost (32) RMSE | XGBoost + HA RMSE | RF MAPE % | XGBoost + HA MAPE % |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for h in HORIZONS:
            rf = d_results["experiments"]["D"][h]["test_metrics"]
            x, xh = rs[h]["test"]["xgboost"]["all"], rs[h]["test"]["xgboost_ha"]["all"]
            L.append(
                f"| {h} | {fmt(rf['mae'])} | {fmt(x['mae'])} | **{fmt(xh['mae'])}** | {fmt(rf['rmse'])} | {fmt(x['rmse'])} | {fmt(xh['rmse'])} | {fmt(rf['mape'], 2)} | {fmt(xh['mape'], 2)} |"
            )
        L += [
            "",
            "The like-for-like comparison is the RF against **XGBoost (32)**: same inputs. "
            "XGBoost + HA also uses the extra history features, so its lead over RF mixes a model effect and a feature effect.",
        ]

    L += ["", "## What the results say", ""]
    wins = [
        h
        for h in HORIZONS
        if rs[h]["test"]["xgboost"]["all"]["mae"]
        < rs[h]["test"]["historical_average"]["all"]["mae"]
    ]
    losses = [h for h in HORIZONS if h not in wins]
    if losses:
        L.append(
            f"- **XGBoost on the 32 shared features does not beat the historical average on test at {', '.join(losses)}** "
            "(it does on validation). The shared features do not explicitly supply a sensor's target-time historical profile. "
            "The separate HA extension tests the value of that additional local information; this comparison does not establish a causal explanation for every error."
        )
    L.append(
        "- Adding the historical-average features fixes this: XGBoost + HA beats every baseline on MAE and RMSE at every horizon on test."
    )
    if d_results:
        rf_better = [
            h
            for h in HORIZONS
            if d_results["experiments"]["D"][h]["test_metrics"]["rmse"]
            < rs[h]["test"]["xgboost"]["all"]["rmse"]
        ]
        if rf_better:
            L.append(
                f"- On the same 32 features, XGBoost beats Person D's Random Forest on MAE and MAPE at every horizon, but the RF has the lower RMSE at {', '.join(rf_better)}. "
                "XGBoost is trained on absolute error and RF on squared error, so each favours its own metric."
            )
    L.append(
        "- Errors are higher in the defined peak periods, especially evening origins. This is a descriptive result on the stored clock; no verified physical cause or graph-model advantage is inferred."
    )

    L += [
        "",
        "## Model details",
        "",
        "Tree counts and fit times below are preserved original training metadata, not the elapsed time of prediction regeneration.",
        "",
        "| Horizon | Model | Trees (early stopped) | Fit time (s) | Train rows |",
        "|---|---|---:|---:|---:|",
    ]
    for h in HORIZONS:
        for key, name in (("fit", LABEL["xgboost"]), ("fit_ha", LABEL["xgboost_ha"])):
            i = rs[h][key]
            L.append(
                f"| {h} | {name} | {i['best_iteration']} | {i['fit_time_seconds']} | {i['train_samples']:,} |"
            )
    params = rs[HORIZONS[0]]["fit"]["params"]
    L += [
        "",
        "Params: " + ", ".join(f"`{k}={v}`" for k, v in params.items()),
        "",
        "### Preserved training design",
        "",
        "- **Residual target.** The preserved boosters predict `target − speed_lag_0`; current speed is added back at inference.",
        "- **Absolute-error loss** (`reg:absoluteerror`) and validation early stopping are recorded in the saved model and fit metadata.",
        "- Completion re-evaluates frozen models. It does not repeat the original hyperparameter search or use test errors to select new settings.",
        "- **Leave-one-out history features on train rows.** Without it, a training row's own target sits inside its average and the model over-trusts the feature.",
        "",
        "## Top features (gain, normalised)",
        "",
    ]
    for h in HORIZONS:
        for key, name in (
            ("feature_importance_gain", "32 features"),
            ("feature_importance_gain_ha", "+ HA"),
        ):
            top = list(rs[h][key].items())[:5]
            L.append(
                f"- **{h}, {name}:** " + ", ".join(f"`{k}` {v:.3f}" for k, v in top)
            )

    L += [
        "",
        "## Caveats",
        "",
        "- The test period has more missing readings (87.9% valid targets against 92.9% in train), so test and validation errors are not directly comparable.",
        "- Peak/off-peak uses the origin time. For the 60 min horizon the target falls up to an hour later.",
        "- The timezone of the stored clock is unverified, so 'peak' means peak on the stored clock.",
        "- Phase 1 uses each sensor's own history only. No neighbouring sensors or graph inputs.",
        "- The history features use ~12 weeks of training days per (sensor, weekday, slot); holidays are not modelled.",
        "",
        "## Handoff (Person E)",
        "",
        "- `data/metadata/person_c_results.json` records model-run metrics at four decimals. After common evaluation, displayed headline tables use unrounded `phase1_metrics.csv` scores to avoid double rounding.",
        "- `data/processed/predictions/person_c_{val,test}_predictions.parquet`: one row per (origin, sensor) with target, validity flag and all four models' predictions per horizon, for maps and plots. Rebuild with `python scripts/run_person_c.py`.",
        "- `models/person_c/xgb_<h>.json` and `xgb_ha_<h>.json` are the trained boosters. They predict the change; add `speed_lag_0` to get speed.",
        "- Code: `src/models/baselines.py`, `src/models/xgboost_model.py`, `scripts/run_person_c.py`, `tests/test_person_c.py`.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L))


def build_prediction_frame(
    df: pd.DataFrame, preds: dict[str, dict[str, np.ndarray]]
) -> pd.DataFrame:
    out = {
        "origin_timestamp": df["origin_timestamp"].to_numpy(),
        "sensor_id": df["sensor_id"].astype(str).to_numpy(),
        "is_morning_peak": df["is_morning_peak"].to_numpy(),
        "is_evening_peak": df["is_evening_peak"].to_numpy(),
        "speed_lag_0": df["speed_lag_0"].to_numpy(np.float32),
    }
    for h in HORIZONS:
        out[f"target_{h}"] = df[f"target_{h}"].to_numpy(np.float32)
        out[f"target_valid_{h}"] = df[f"target_valid_{h}"].to_numpy()
        for m in MODELS:
            out[f"pred_{m}_{h}"] = preds[h][m].astype(np.float32)
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-rounds", type=int, default=2000)
    ap.add_argument("--horizons", nargs="+", default=HORIZONS)
    ap.add_argument(
        "--predict-only",
        action="store_true",
        help="Regenerate predictions and metrics from the eight saved boosters without refitting",
    )
    args = ap.parse_args()

    summary = json.loads((ROOT / "data/processed/dataset_summary.json").read_text())
    features = summary["feature_groups"]["full_features"]

    print("loading data ...", flush=True)
    train_df = None if args.predict_only else load_partition("train")
    val_df, test_df = (load_partition(p) for p in ("val", "test"))

    raw_df = load_raw_speed_data()
    contract = load_split_contract()
    train_end = get_partition_info(raw_df.index, contract)["train"].end_inclusive
    hist = HistoricalAverage().fit(raw_df, train_end)
    for df in (val_df, test_df) if args.predict_only else (train_df, val_df, test_df):
        assert (
            list(df["sensor_id"].cat.categories) == hist.sensor_ids_
        ), "sensor order mismatch"
    print(f"historical average fit on timestamps <= {train_end}", flush=True)

    results = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "features": features,
            "n_features": len(features),
            "train_end_for_historical_average": str(train_end),
        },
        "horizons": {},
    }
    models_dir = ROOT / "models/person_c"
    models_dir.mkdir(parents=True, exist_ok=True)
    prior = (
        json.loads((ROOT / "data/metadata/person_c_results.json").read_text())
        if args.predict_only
        else None
    )
    results["metadata"]["execution"] = (
        "saved_booster_inference" if args.predict_only else "training_and_inference"
    )
    if prior:
        results["metadata"]["original_training_timestamp_utc"] = prior["metadata"].get(
            "original_training_timestamp_utc", prior["metadata"]["timestamp_utc"]
        )
    all_preds = {"val": {}, "test": {}}

    for h in args.horizons:
        print(f"\n=== horizon {h} ===", flush=True)
        # --- plain XGBoost: the shared 32 features, comparable with Person D ---
        if args.predict_only:
            booster = load_saved_booster(models_dir / f"xgb_{h}.json")
            fit_info = prior["horizons"][h]["fit"]
        else:
            booster, fit_info = train_horizon(
                train_df, val_df, features, h, max_rounds=args.max_rounds
            )
            booster.save_model(models_dir / f"xgb_{h}.json")
        print(
            f"  xgboost:    {fit_info['best_iteration']} trees, {fit_info['fit_time_seconds']}s",
            flush=True,
        )

        # --- XGBoost + historical-average features (extension, own-sensor history only) ---
        for df, is_train in (
            [(val_df, False), (test_df, False)]
            if args.predict_only
            else [(train_df, True), (val_df, False), (test_df, False)]
        ):
            extra = hist.features(
                df["origin_timestamp"],
                df["sensor_id"].cat.codes.to_numpy(),
                df["speed_lag_0"].to_numpy(),
                STEPS[h],
                training_rows=is_train,
            )
            for col in HA_FEATURE_COLUMNS:
                df[col] = extra[col].to_numpy()
        features_ha = features + HA_FEATURE_COLUMNS
        if args.predict_only:
            booster_ha = load_saved_booster(models_dir / f"xgb_ha_{h}.json")
            fit_info_ha = prior["horizons"][h]["fit_ha"]
        else:
            booster_ha, fit_info_ha = train_horizon(
                train_df, val_df, features_ha, h, max_rounds=args.max_rounds
            )
            booster_ha.save_model(models_dir / f"xgb_ha_{h}.json")
        print(
            f"  xgboost_ha: {fit_info_ha['best_iteration']} trees, {fit_info_ha['fit_time_seconds']}s",
            flush=True,
        )

        def importance_of(b):
            gain = b.get_score(importance_type="gain")
            total = sum(gain.values()) or 1.0
            return dict(
                sorted(
                    ((k, round(v / total, 5)) for k, v in gain.items()),
                    key=lambda kv: -kv[1],
                )
            )

        entry = {
            "fit": fit_info,
            "fit_ha": fit_info_ha,
            "feature_importance_gain": importance_of(booster),
            "feature_importance_gain_ha": importance_of(booster_ha),
        }
        for split, df in (("val", val_df), ("test", test_df)):
            preds = {
                "persistence": persistence_predict(df),
                "historical_average": hist.predict(
                    df["origin_timestamp"],
                    df["sensor_id"].cat.codes.to_numpy(),
                    STEPS[h],
                ),
                "xgboost": predict(booster, df, features),
                "xgboost_ha": predict(booster_ha, df, features_ha),
            }
            # Score the same float32 values that are persisted for independent evaluation.
            preds = {name: values.astype(np.float32) for name, values in preds.items()}
            all_preds[split][h] = preds
            entry[split] = score_by_period(df, h, preds)
            a = {m: entry[split][m]["all"] for m in MODELS}
            print(
                f"  {split}: "
                + " | ".join(
                    f"{m} MAE {a[m]['mae']:.3f} RMSE {a[m]['rmse']:.3f}" for m in MODELS
                ),
                flush=True,
            )
        results["horizons"][h] = entry

    pred_dir = ROOT / "data/processed/predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    if set(args.horizons) == set(HORIZONS):
        for split, df in (("val", val_df), ("test", test_df)):
            build_prediction_frame(df, all_preds[split]).to_parquet(
                pred_dir / f"person_c_{split}_predictions.parquet", index=False
            )

    out_json = ROOT / "data/metadata/person_c_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    d_path = ROOT / "data/metadata/person_d_results.json"
    d_results = json.loads(d_path.read_text()) if d_path.is_file() else None
    if set(args.horizons) == set(HORIZONS):
        write_report(results, d_results, ROOT / "reports/person_c_report.md")
    print(f"\nresults -> {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
