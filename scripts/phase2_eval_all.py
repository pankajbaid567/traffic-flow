"""Master evaluation and scoring CLI for Phase 2 (Member 4).

Usage:
    python scripts/phase2_eval_all.py [--pred-root DIR] [--out DIR] [--splits val test]

Contract: reports/phase2_plan.md section 4.7, 4.8, 8.4.
- Discovers Phase 1 models and any Phase 2 .npz predictions in pred-root.
- Evaluates pooled MAE/RMSE/MAPE across all horizons and peak/off-peak periods.
- Runs paired circular block bootstrap and evaluates formal hypotheses H0-H5.
- Computes congestion onset precision/recall and stratified slices.
- Renders publication-grade 300 DPI figures.
- Gracefully handles partial or absent Phase 2 models without crashing.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from src.phase2.common import HORIZON_NAMES, ROOT
from src.phase2.evaluation.bootstrap import (
    execute_all_bootstrap_comparisons,
    evaluate_hypotheses,
)
from src.phase2.evaluation.events import (
    compute_congestion_event_metrics,
    compute_stratified_slices,
    load_sensor_reference_speeds,
)
from src.phase2.evaluation.loaders import (
    EXPECTED_ORIGIN_COUNTS,
    attach_phase2_predictions,
    load_canonical_origins,
    load_canonical_sensor_ids,
    load_phase1_predictions,
    load_phase2_npz,
)
from src.phase2.evaluation.scoring import (
    compute_metrics_summary,
    format_metrics_table,
    score_dataframe,
)
from src.phase2.viz.plots import render_all_figures

DEFAULT_PRED_ROOT = Path(os.environ.get("PHASE2_OUT", ROOT / "data" / "processed" / "phase2")) / "predictions"
DEFAULT_OUT_DIR = ROOT / "data" / "metadata" / "phase2"
DEFAULT_FIG_DIR = ROOT / "figures" / "phase2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 Master Evaluation Runner")
    parser.add_argument(
        "--pred-root",
        type=Path,
        default=DEFAULT_PRED_ROOT,
        help=f"Directory containing Phase 2 model subdirectories with .npz files (default: {DEFAULT_PRED_ROOT})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory for CSV metadata (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIG_DIR,
        help=f"Output directory for figures (default: {DEFAULT_FIG_DIR})",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test"],
        help="Partitions to score (default: val test)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed console tables",
    )
    return parser.parse_args()


def discover_phase2_predictions(pred_root: Path) -> list[dict[str, Any]]:
    """Scan pred_root for <model_id>/seed<S>_<split>.npz files."""
    found = []
    if not pred_root.is_dir():
        return found

    for model_dir in pred_root.iterdir():
        if not model_dir.is_dir():
            continue
        model_id = model_dir.name
        for npz_file in model_dir.glob("seed*_*.*"):
            if not npz_file.name.endswith(".npz"):
                continue
            name_parts = npz_file.stem.split("_")
            if len(name_parts) >= 2 and name_parts[0].startswith("seed"):
                try:
                    seed = int(name_parts[0][4:])
                    split = name_parts[1]
                    found.append({
                        "model_id": model_id,
                        "seed": seed,
                        "split": split,
                        "path": npz_file,
                        "col_prefix": f"{model_id}_seed{seed}",
                    })
                except ValueError:
                    continue
    return found


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Phase 2 Evaluation Pipeline ===")
    print(f"Prediction root: {args.pred_root}")
    print(f"Metadata output: {args.out}")
    print(f"Figures output:  {args.figures_dir}")
    print(f"Target splits:   {args.splits}")

    canonical_sensors = load_canonical_sensor_ids()
    discovered_phase2 = discover_phase2_predictions(args.pred_root)
    print(f"Discovered Phase 2 models on disk: {len(discovered_phase2)} .npz files")

    all_metrics_rows = []
    daily_stats_list = []
    sensor_scores_list = []
    scored_models_display = set()

    # Pre-load reference speeds for congestion proxies
    try:
        ref_speeds = load_sensor_reference_speeds()
    except Exception as e:
        print(f"Warning: could not load training reference speeds: {e}")
        ref_speeds = {}

    for split in args.splits:
        norm_split = "val" if split in ("val", "validation") else "test"
        print(f"\n--- Processing split: {split} ({norm_split}) ---")

        # 1. Load Phase 1 predictions frame
        try:
            df = load_phase1_predictions(split=norm_split)
            print(f"Loaded Phase 1 {norm_split} partition: {len(df):,} rows")
            phase1_available = True
        except FileNotFoundError as e:
            print(f"Notice: Phase 1 predictions absent ({e}). Skipping Phase 1 models.")
            phase1_available = False
            continue

        # Canonical models map for Phase 1
        models_map = {
            "persistence": "persistence",
            "historical_average": "historical_average",
            "random_forest_32": "random_forest",
            "xgboost_32": "xgboost",
            "xgboost_ha": "xgboost_ha",
        }
        for m in models_map:
            scored_models_display.add(f"Phase 1: {m}")

        # 2. Attach any discovered Phase 2 predictions
        split_p2 = [p for p in discovered_phase2 if p["split"] == norm_split]
        model_meta_map = {}
        for p in split_p2:
            model_id = p["model_id"]
            seed = p["seed"]
            col_prefix = p["col_prefix"]
            try:
                p2_data = load_phase2_npz(
                    p["path"],
                    split=norm_split,
                    canonical_sensors=canonical_sensors,
                )
                df = attach_phase2_predictions(df, p2_data, col_prefix=col_prefix)
                models_map[col_prefix] = col_prefix
                model_meta_map[col_prefix] = {"model_id": model_id, "seed": seed}
                scored_models_display.add(f"Phase 2: {model_id} (seed {seed})")
                print(f"  Attached Phase 2: {model_id} (seed {seed})")
            except Exception as e:
                print(f"  Warning: failed to attach {p['path'].name}: {e}")

        # 3. Score aligned frame
        print(f"Scoring {len(models_map)} models on {norm_split}...")
        overall, sensors, daily = score_dataframe(
            df=df,
            split=norm_split,
            models_map=models_map,
            training_free_speed=ref_speeds,
        )

        formatted_overall = format_metrics_table(overall, model_metadata_map=model_meta_map)
        all_metrics_rows.append(formatted_overall)
        daily_stats_list.append(daily)
        sensor_scores_list.append(sensors)

    if not all_metrics_rows:
        print("\nNo splits could be evaluated (data missing). Exiting cleanly.")
        return 0

    # 4. Save metrics.csv and compute metrics_summary.csv
    combined_metrics = pd.concat(all_metrics_rows, ignore_index=True)
    metrics_path = args.out / "metrics.csv"
    combined_metrics.to_csv(metrics_path, index=False)
    print(f"\nWrote: {metrics_path} ({len(combined_metrics)} rows)")

    metrics_summary = compute_metrics_summary(combined_metrics)
    summary_path = args.out / "metrics_summary.csv"
    metrics_summary.to_csv(summary_path, index=False)
    print(f"Wrote: {summary_path} ({len(metrics_summary)} rows)")

    # 5. Bootstrap and hypothesis testing (on test split)
    combined_daily = pd.concat(daily_stats_list, ignore_index=True)
    test_daily = combined_daily[combined_daily["split"] == "test"]
    if not test_daily.empty:
        print("\nExecuting paired circular block bootstrap...")
        bootstrap_df = execute_all_bootstrap_comparisons(test_daily)
        boot_path = args.out / "bootstrap.csv"
        bootstrap_df.to_csv(boot_path, index=False)
        print(f"Wrote: {boot_path} ({len(bootstrap_df)} comparisons)")

        hypotheses_df = evaluate_hypotheses(bootstrap_df)
        hyp_path = args.out / "hypotheses.csv"
        hypotheses_df.to_csv(hyp_path, index=False)
        print(f"Wrote: {hyp_path} (H0-H5 evaluated)")
    else:
        bootstrap_df = pd.DataFrame()
        hypotheses_df = evaluate_hypotheses(bootstrap_df)
        hyp_path = args.out / "hypotheses.csv"
        hypotheses_df.to_csv(hyp_path, index=False)
        print(f"Test split not scored; wrote placeholder {hyp_path}")

    # 6. Congestion events
    combined_sensors = pd.concat(sensor_scores_list, ignore_index=True)
    if "test" in args.splits and phase1_available:
        try:
            test_df = load_phase1_predictions(split="test")
            events_df = compute_congestion_event_metrics(
                df=test_df,
                models=["persistence", "xgboost_ha"],
                reference_speeds=ref_speeds,
            )
            events_path = args.out / "events.csv"
            events_df.to_csv(events_path, index=False)
            print(f"Wrote: {events_path} ({len(events_df)} rows)")
        except Exception as e:
            print(f"Warning: could not compute events: {e}")

    # 7. Slices (sensor_metrics and region_metrics)
    if not combined_sensors.empty:
        try:
            sensor_metrics, region_metrics = compute_stratified_slices(
                df=None, sensor_scores_df=combined_sensors
            )
            sm_path = args.out / "sensor_metrics.csv"
            rm_path = args.out / "region_metrics.csv"
            sensor_metrics.to_csv(sm_path, index=False)
            region_metrics.to_csv(rm_path, index=False)
            print(f"Wrote: {sm_path} & {rm_path}")
        except Exception as e:
            print(f"Warning: could not write slices: {e}")

    # 8. Render figures
    print("\nRendering publication figures...")
    try:
        figs = render_all_figures(
            metrics_df=combined_metrics,
            metrics_summary_df=metrics_summary,
            sensor_scores_df=combined_sensors,
        )
        for name, path in figs.items():
            print(f"  Rendered: {path.name}")
    except Exception as e:
        print(f"Warning: figure rendering encountered error: {e}")

    # 9. Summary console output
    print("\n=== Evaluation Complete ===")
    print(f"Scored models: {sorted(list(scored_models_display))}")
    if not hypotheses_df.empty:
        print("\nHypothesis Verdict Summary:")
        for _, row in hypotheses_df.iterrows():
            print(f"  {row['id']}: {row['verdict']:<15} ({row['statement']})")

    return 0


if __name__ == "__main__":
    sys.exit(main())

