"""Score every Phase 1 and Phase 2 model on the same targets (Member 4).

Usage:
    python scripts/phase2_eval_all.py [--pred-root DIR] [--out DIR] [--figures-dir DIR]
                                      [--splits val test] [--repetitions 2000]

Contract: reports/phase2_plan.md sections 4.7, 4.8, 5.3, 5.4.

* Phase 1 predictions come from the Phase 1 parquets (targets, masks, origins).
* Phase 2 predictions ``<pred-root>/<model_id>/seed<S>_<split>.npz`` are checked
  against the Phase 1 origins and sensor order and attached by index. A file that
  fails a check stops the run; nothing is skipped silently.
* Seed handling: per-seed rows in metrics.csv; seed means in metrics_summary.csv;
  daily error sums averaged over seeds for the bootstrap (plan 5.4).
* Writes metrics.csv, metrics_summary.csv, bootstrap.csv, hypotheses.csv,
  sensor_metrics.csv, region_metrics.csv, events.csv, run_meta.csv,
  phase1_parity.json to ``--out`` and figures to ``--figures-dir``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from src.models.prediction_evaluation import geographic_regions, region_scores, temporal_masks
from src.phase2.common import HORIZON_NAMES, ROOT, load_graph_pickle
from src.phase2.evaluation.events import load_sensor_reference_speeds
from src.phase2.evaluation.hypotheses import (
    GRAPH_COMPARISONS,
    PHASE1_COMPARISONS,
    evaluate_hypotheses,
    run_comparisons,
)
from src.phase2.evaluation.loaders import (
    load_canonical_sensor_ids,
    load_phase1_predictions,
    load_phase2_npz,
)
from src.phase2.evaluation.scoring import compute_metrics_summary, score_dataframe

DEFAULT_PRED_ROOT = Path(os.environ.get("PHASE2_OUT", ROOT / "data" / "processed" / "phase2")) / "predictions"
DEFAULT_OUT_DIR = ROOT / "data" / "metadata" / "phase2"
DEFAULT_FIG_DIR = ROOT / "figures" / "phase2"
PHASE1_MODELS = {
    "persistence": "persistence",
    "historical_average": "historical_average",
    "random_forest_32": "random_forest",
    "xgboost_32": "xgboost",
    "xgboost_ha": "xgboost_ha",
}
PERIODS = ("all", "peak", "off_peak")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 evaluation runner")
    p.add_argument("--pred-root", type=Path, default=DEFAULT_PRED_ROOT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--figures-dir", type=Path, default=DEFAULT_FIG_DIR)
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.add_argument("--repetitions", type=int, default=2000)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def discover_phase2_predictions(pred_root: Path) -> list[dict]:
    """``<pred_root>/<model_id>/seed<S>_<split>.npz`` -> list of records, sorted."""
    found = []
    if not pred_root.is_dir():
        return found
    for model_dir in sorted(pred_root.iterdir()):
        if not model_dir.is_dir():
            continue
        for f in sorted(model_dir.glob("seed*_*.npz")):
            seed_part, split = f.stem.split("_", 1)
            try:
                seed = int(seed_part[4:])
            except ValueError:
                continue
            found.append({"model_id": model_dir.name, "seed": seed, "split": split, "path": f,
                          "col_prefix": f"{model_dir.name}__s{seed}"})
    return found


def read_run_meta(pred_root: Path) -> pd.DataFrame:
    rows = []
    if pred_root.is_dir():
        for f in sorted(pred_root.glob("*/seed*_meta.json")):
            m = json.loads(f.read_text())
            rows.append({k: m.get(k) for k in (
                "model_id", "seed", "params", "epochs_run", "best_epoch", "best_val_mae",
                "train_seconds", "seconds_per_epoch", "stop_reason", "device", "torch_version",
                "git_commit", "config_hash")})
            rows[-1]["graph"] = (m.get("config") or {}).get("graph")
            rows[-1]["model_params"] = json.dumps((m.get("config") or {}).get("params"))
    return pd.DataFrame(rows)


def attach_by_index(df: pd.DataFrame, pred: np.ndarray, origin_ns: np.ndarray,
                    sensor_ids: list[str], prefix: str) -> None:
    """Add ``pred_<prefix>_<h>`` columns in place by origin/sensor index (no merge copy)."""
    o_pos = pd.Index(origin_ns).get_indexer(df["origin_timestamp"].to_numpy().astype("datetime64[ns]").astype(np.int64))
    s_pos = pd.Index(sensor_ids).get_indexer(df["sensor_id"].astype(str).to_numpy())
    if (o_pos < 0).any() or (s_pos < 0).any():
        raise ValueError(f"{prefix}: predictions do not cover every Phase 1 row")
    for k, h in enumerate(HORIZON_NAMES):
        df[f"pred_{prefix}_{h}"] = pred[o_pos, k, s_pos].astype(np.float64)


def daily_by_period(df: pd.DataFrame, split: str, models_map: dict[str, str],
                    model_of: dict[str, str]) -> pd.DataFrame:
    """Per-day absolute-error sums for all/peak/off_peak, seed-averaged per model id."""
    masks = temporal_masks(df["origin_timestamp"])
    dates = pd.DatetimeIndex(df["origin_timestamp"]).normalize()
    days = pd.DatetimeIndex(sorted(dates.unique()))
    day_codes = days.get_indexer(dates)
    day_str = [str(d.date()) for d in days]
    rows = []
    for h in HORIZON_NAMES:
        valid = df[f"target_valid_{h}"].to_numpy(bool)
        y = df[f"target_{h}"].to_numpy(float)
        for col, prefix in models_map.items():
            err = np.abs(y - df[f"pred_{prefix}_{h}"].to_numpy(float))
            for period in PERIODS:
                m = valid & masks[period]
                n = np.bincount(day_codes[m], minlength=len(days))
                s = np.bincount(day_codes[m], weights=err[m], minlength=len(days))
                rows.append(pd.DataFrame({"split": split, "horizon": h, "model": model_of[col],
                                          "column": col, "period": period, "day": day_str,
                                          "n": n, "absolute_error_sum": s}))
    daily = pd.concat(rows, ignore_index=True)
    return (daily.groupby(["split", "horizon", "model", "period", "day"], as_index=False)
            .agg(n=("n", "first"), absolute_error_sum=("absolute_error_sum", "mean"), n_seeds=("column", "nunique")))


def event_metrics(df: pd.DataFrame, models_map: dict[str, str], model_of: dict[str, str],
                  seed_of: dict[str, object], ref: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Proxy-congestion detection and onset recall per model column (test split)."""
    thr = 0.6 * df["sensor_id"].astype(str).map(ref).to_numpy(float)
    origin_speed = df["speed_lag_0"].to_numpy(float)
    origin_free = np.isfinite(origin_speed) & (origin_speed > 0) & (origin_speed >= thr)
    rows, onset_err = [], []
    for h in HORIZON_NAMES:
        valid = df[f"target_valid_{h}"].to_numpy(bool)
        y = df[f"target_{h}"].to_numpy(float)
        y_c = valid & (y < thr)
        onset = origin_free & y_c
        for col, prefix in models_map.items():
            p = df[f"pred_{prefix}_{h}"].to_numpy(float)
            p_c = valid & (p < thr)
            tp, fp, fn = int((p_c & y_c).sum()), int((p_c & ~y_c).sum()), int((~p_c & y_c).sum())
            tp_on = int((p_c & onset).sum())
            rows.append({"model": model_of[col], "seed": seed_of[col], "horizon": h,
                         "n_valid_targets": int(valid.sum()), "actual_congested": int(y_c.sum()),
                         "predicted_congested": int(p_c.sum()),
                         "precision_congestion": tp / (tp + fp) if tp + fp else np.nan,
                         "recall_congestion": tp / (tp + fn) if tp + fn else np.nan,
                         "actual_onsets": int(onset.sum()),
                         "recall_onset": tp_on / onset.sum() if onset.sum() else np.nan,
                         "definition": "proxy: speed < 0.6 x sensor training 85th percentile; onset = not congested at origin, congested at target"})
            err = np.abs(y - p)
            onset_err.append({"model": model_of[col], "seed": seed_of[col], "horizon": h,
                              "mae_onset": float(err[onset].mean()) if onset.any() else np.nan,
                              "mae_other": float(err[valid & ~onset].mean())})
    return pd.DataFrame(rows), pd.DataFrame(onset_err)


# Phase 1's retrospective example (data/metadata/person_e_results.json -> example):
# sensor 717468, centre target time 2012-06-11 19:40 (stored clock).
CASE_SENSOR, CASE_START, CASE_END = "717468", "2012-06-11 14:00", "2012-06-11 23:00"


def write_case_study(df: pd.DataFrame, models_map: dict, model_of: dict, path: Path) -> None:
    sel = (df["sensor_id"].astype(str) == CASE_SENSOR) & df["origin_timestamp"].between(CASE_START, CASE_END)
    c = df.loc[sel, ["origin_timestamp", "sensor_id"] + [f"{k}_{h}" for h in HORIZON_NAMES
                                                          for k in ("target", "target_valid")]].copy()
    for h in HORIZON_NAMES:
        by_model: dict[str, list] = {}
        for col, prefix in models_map.items():
            by_model.setdefault(model_of[col], []).append(df.loc[sel, f"pred_{prefix}_{h}"].to_numpy(float))
        for model, arrs in by_model.items():
            c[f"pred_{model}_{h}"] = np.mean(arrs, axis=0)
    c.sort_values("origin_timestamp").to_csv(path, index=False)


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"Prediction root: {args.pred_root}\nOutputs: {args.out} | figures: {args.figures_dir}")

    sensors = load_canonical_sensor_ids()
    found = discover_phase2_predictions(args.pred_root)
    run_meta = read_run_meta(args.pred_root)
    print(f"Phase 2 prediction files found: {len(found)}")
    ref = load_sensor_reference_speeds()

    metrics_rows, daily_parts, sensor_parts = [], [], []
    events = onset = None
    test_frame_cols = None
    for split in args.splits:
        split = "val" if split in ("val", "validation") else "test"
        df = load_phase1_predictions(split=split)
        origins = pd.DatetimeIndex(sorted(df["origin_timestamp"].unique()))
        models_map = dict(PHASE1_MODELS)
        model_of = {m: m for m in PHASE1_MODELS}
        seed_of: dict[str, object] = {m: "" for m in PHASE1_MODELS}
        for rec in [r for r in found if r["split"] == split]:
            data = load_phase2_npz(rec["path"], split=split, canonical_origins=origins,
                                   canonical_sensors=sensors)
            attach_by_index(df, data["pred"], data["origin_ns"], data["sensor_ids"], rec["col_prefix"])
            models_map[rec["col_prefix"]] = rec["col_prefix"]
            model_of[rec["col_prefix"]] = rec["model_id"]
            seed_of[rec["col_prefix"]] = rec["seed"]
        print(f"[{split}] {len(df):,} rows, {len(origins):,} origins, {len(models_map)} model columns")

        overall, sens, _ = score_dataframe(df, split=split, models_map=models_map, training_free_speed=ref)
        overall["seed"] = overall["model"].map(seed_of)
        overall["model"] = overall["model"].map(model_of)
        metrics_rows.append(overall[["split", "horizon", "model", "seed", "period", "n", "mae", "rmse", "mape"]])
        sens["seed"] = sens["model"].map(seed_of)
        sens["model"] = sens["model"].map(model_of)
        sensor_parts.append(sens)
        daily_parts.append(daily_by_period(df, split, models_map, model_of))
        if split == "test":
            events, onset = event_metrics(df, models_map, model_of, seed_of, ref)
            write_case_study(df, models_map, model_of, args.out / "case_study_717468.csv")
        del df

    metrics = pd.concat(metrics_rows, ignore_index=True)
    metrics.to_csv(args.out / "metrics.csv", index=False)

    meta_stats = {}
    if not run_meta.empty:
        run_meta.to_csv(args.out / "run_meta.csv", index=False)
        for mid, g in run_meta.groupby("model_id"):
            for split in ("val", "test"):
                meta_stats[(mid, split)] = {"params": int(g["params"].iloc[0]),
                                            "train_seconds_mean": float(g["train_seconds"].mean())}
    summary = compute_metrics_summary(metrics, meta_stats=meta_stats)
    summary.to_csv(args.out / "metrics_summary.csv", index=False)
    print(f"Wrote metrics.csv ({len(metrics)} rows), metrics_summary.csv ({len(summary)} rows)")

    # Phase 1 parity (harness must reproduce Phase 1 within 1e-3)
    p1 = pd.read_csv(ROOT / "data" / "metadata" / "phase1_metrics.csv")
    mine = metrics[metrics["model"].isin(PHASE1_MODELS)].merge(
        p1, on=["split", "horizon", "model", "period"], suffixes=("", "_p1"))
    parity = {"rows_compared": int(len(mine)),
              "max_abs_mae_diff": float((mine["mae"] - mine["mae_p1"]).abs().max()),
              "max_abs_rmse_diff": float((mine["rmse"] - mine["rmse_p1"]).abs().max()),
              "n_counts_equal": bool((mine["n"] == mine["n_p1"]).all())}
    parity["passed"] = parity["rows_compared"] > 0 and parity["max_abs_mae_diff"] < 1e-3 and parity["n_counts_equal"]
    (args.out / "phase1_parity.json").write_text(json.dumps(parity, indent=2))
    print(f"Phase 1 parity: {parity}")
    if not parity["passed"]:
        raise SystemExit("Phase 1 numbers not reproduced; stopping.")

    # Sensor and region metrics (seed-averaged sums)
    sens_all = pd.concat(sensor_parts, ignore_index=True)
    sum_cols = ["n", "absolute_error_sum", "squared_error_sum", "percentage_error_sum", "actual_sum", "predicted_sum",
                "actual_congested_count", "predicted_congested_count"]
    sens_avg = sens_all.groupby(["split", "horizon", "model", "period", "sensor_id"], as_index=False)[sum_cols].mean()
    sens_avg["mae"] = sens_avg["absolute_error_sum"] / sens_avg["n"]
    sens_avg["rmse"] = np.sqrt(sens_avg["squared_error_sum"] / sens_avg["n"])
    sens_avg["mape"] = sens_avg["percentage_error_sum"] / sens_avg["n"]
    meta = pd.read_csv(ROOT / "data" / "metadata" / "sensors.csv", dtype={"sensor_id": str})
    ids, _, adj = load_graph_pickle(ROOT / "data" / "raw" / "adj_mx.pkl")
    a = (np.asarray(adj) > 0)
    np.fill_diagonal(a, False)
    deg = pd.DataFrame({"sensor_id": [str(s) for s in ids],
                        "out_degree": a.sum(1), "in_degree": a.sum(0),
                        "graph_degree": (a | a.T).sum(1)})
    regions = geographic_regions(meta)[["sensor_id", "region", "latitude", "longitude"]]
    sens_out = sens_avg.merge(deg, on="sensor_id", how="left").merge(regions, on="sensor_id", how="left")
    sens_out.to_csv(args.out / "sensor_metrics.csv", index=False)
    region_scores(sens_avg, meta).to_csv(args.out / "region_metrics.csv", index=False)

    # Events (seed-averaged rates)
    if events is not None:
        ev = events.groupby(["model", "horizon"], as_index=False).agg(
            n_seeds=("seed", "count"), n_valid_targets=("n_valid_targets", "first"),
            actual_congested=("actual_congested", "first"), predicted_congested=("predicted_congested", "mean"),
            precision_congestion=("precision_congestion", "mean"), recall_congestion=("recall_congestion", "mean"),
            actual_onsets=("actual_onsets", "first"), recall_onset=("recall_onset", "mean"),
            definition=("definition", "first"))
        ev.to_csv(args.out / "events.csv", index=False)
        onset_avg = onset.groupby(["model", "horizon"], as_index=False)[["mae_onset", "mae_other"]].mean()
        onset_avg.to_csv(args.out / "onset_errors.csv", index=False)

    # Bootstrap and hypotheses (test split)
    daily = pd.concat(daily_parts, ignore_index=True)
    test_daily = daily[daily["split"] == "test"]
    boot = run_comparisons(test_daily, GRAPH_COMPARISONS + PHASE1_COMPARISONS)
    boot.to_csv(args.out / "bootstrap.csv", index=False)
    print(f"Wrote bootstrap.csv ({len(boot)} intervals)")

    p1b = pd.read_csv(ROOT / "data" / "metadata" / "paired_bootstrap.csv")
    chk = p1b.merge(boot[boot["period"] == "all"], on=["baseline", "candidate", "horizon"], suffixes=("_p1", ""))
    if len(chk) != len(p1b) or (chk["mae_difference"] - chk["mae_difference_p1"]).abs().max() > 1e-6 \
            or (chk["ci95_low"] - chk["ci95_low_p1"]).abs().max() > 1e-6:
        raise SystemExit("Phase 1 bootstrap intervals not reproduced; stopping.")
    parity["phase1_bootstrap_reproduced"] = True
    (args.out / "phase1_parity.json").write_text(json.dumps(parity, indent=2))

    t60 = sens_out[(sens_out["split"] == "test") & (sens_out["horizon"] == "60min") & (sens_out["period"] == "all")]
    sensor_gain = None
    if {"gat", "gat_identity"} <= set(t60["model"]):
        g = t60.pivot(index="sensor_id", columns="model", values="mae")
        sensor_gain = pd.DataFrame({"gain": g["gat_identity"] - g["gat"]}).join(
            deg.set_index("sensor_id")["graph_degree"]).dropna()
    onset_gain = None
    if events is not None and {"gat", "gat_identity"} <= set(onset_avg["model"]):
        o = onset_avg.set_index(["horizon", "model"])
        onset_gain = pd.DataFrame([{"horizon": h,
                                    "gain_onset": o.loc[(h, "gat_identity"), "mae_onset"] - o.loc[(h, "gat"), "mae_onset"],
                                    "gain_other": o.loc[(h, "gat_identity"), "mae_other"] - o.loc[(h, "gat"), "mae_other"]}
                                   for h in ("15min", "30min", "60min")])
    hyp = evaluate_hypotheses(test_daily, sensor_gain=sensor_gain, onset_gain=onset_gain)
    hyp.to_csv(args.out / "hypotheses.csv", index=False)
    print("Hypotheses:")
    for hid, g in hyp.groupby("id", sort=False):
        print(f"  {hid}: {g['overall_verdict'].iloc[0]}")

    # Figures
    from src.phase2.viz.report_figures import render_report_figures
    figs = render_report_figures(metrics, summary, boot, sens_out, events_df=pd.read_csv(args.out / "events.csv") if events is not None else None,
                                 run_meta=run_meta, out_dir=args.figures_dir,
                                 attention_dir=args.pred_root.parent / "attention",
                                 case_path=args.out / "case_study_717468.csv")
    for f in figs:
        print(f"  figure: {f.name}")
    print("Evaluation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
