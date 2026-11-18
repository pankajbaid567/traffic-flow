"""Unit tests for Phase 2 Evaluation Harness, Statistics and Visualizations (Member 4).

Contract: reports/phase2_plan.md section 4.7, 4.8, 8.4.
Acceptance criteria:
- Phase 1 reproduction gate: rescoring reproduces phase1_metrics.csv within 1e-3.
- Corrupted inputs (shapes, origins, non-finite values, sensor order) raise ValueError.
- patch_models context manager strictly restores original MODELS mapping.
- End-to-end runner scores stand-in Phase 2 files (real origins) and reproduces Phase 1.
- Verdicts follow the plan's rule; period rows are really period-filtered;
  a one-term contrast equals Phase 1's paired_daily_bootstrap exactly.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.models import prediction_evaluation
from src.phase2.common import HORIZON_NAMES, ROOT
from src.phase2.evaluation.hypotheses import (
    bootstrap_contrast,
    evaluate_hypotheses,
    run_comparisons,
)
from src.phase2.evaluation.loaders import (
    attach_phase2_predictions,
    load_canonical_sensor_ids,
    load_phase1_predictions,
    load_phase2_npz,
)
from src.phase2.evaluation.scoring import (
    compute_metrics_summary,
    format_metrics_table,
    patch_models,
    score_dataframe,
    verify_phase1_reproduction,
)
from src.phase2.viz.plots import render_all_figures

DATA_DIR = ROOT / "data"
PRED_DIR = DATA_DIR / "processed" / "predictions"
PHASE1_METRICS_PATH = DATA_DIR / "metadata" / "phase1_metrics.csv"
PHASE1_BOOTSTRAP_PATH = DATA_DIR / "metadata" / "paired_bootstrap.csv"


class TestPatchModelsContextManager(unittest.TestCase):
    """Test that patch_models restores prediction_evaluation.MODELS."""

    def test_patch_models_restores_original_mapping(self):
        original = prediction_evaluation.MODELS.copy()
        temp_models = {"test_model": "test_prefix"}

        with patch_models(temp_models):
            self.assertEqual(prediction_evaluation.MODELS, temp_models)

        self.assertEqual(prediction_evaluation.MODELS, original)

    def test_patch_models_restores_on_exception(self):
        original = prediction_evaluation.MODELS.copy()
        temp_models = {"broken_model": "broken_prefix"}

        with self.assertRaises(RuntimeError):
            with patch_models(temp_models):
                self.assertEqual(prediction_evaluation.MODELS, temp_models)
                raise RuntimeError("Intentional error inside context")

        self.assertEqual(prediction_evaluation.MODELS, original)


class TestPhase2LoadersAndCorruptions(unittest.TestCase):
    """Test input validation, hard checks and rejection of corrupted predictions."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.n_origins = 100
        self.n_sensors = 207
        self.origins = pd.date_range("2012-06-05 08:00", periods=self.n_origins, freq="5min")
        self.sensors = load_canonical_sensor_ids()
        self.valid_pred = np.random.uniform(40.0, 65.0, size=(self.n_origins, 4, self.n_sensors)).astype(np.float32)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write_npz(self, filename: str, pred, origin_ns, sensor_ids) -> Path:
        path = Path(self.temp_dir) / filename
        np.savez(path, pred=pred, origin_ns=origin_ns, sensor_ids=sensor_ids)
        return path

    def test_load_valid_npz(self):
        path = self._write_npz(
            "valid.npz",
            self.valid_pred,
            self.origins.asi8,
            self.sensors,
        )
        data = load_phase2_npz(
            path,
            split="any",
            canonical_sensors=self.sensors,
            canonical_origins=self.origins,
        )
        self.assertEqual(data["pred"].shape, (self.n_origins, 4, self.n_sensors))
        self.assertEqual(len(data["sensor_ids"]), 207)

    def test_rejects_non_finite_predictions(self):
        corrupt = self.valid_pred.copy()
        corrupt[10, 2, 50] = np.nan
        path = self._write_npz("nan.npz", corrupt, self.origins.asi8, self.sensors)
        with self.assertRaises(ValueError):
            load_phase2_npz(path, split="any")

        corrupt[10, 2, 50] = np.inf
        path_inf = self._write_npz("inf.npz", corrupt, self.origins.asi8, self.sensors)
        with self.assertRaises(ValueError):
            load_phase2_npz(path_inf, split="any")

    def test_rejects_wrong_origin_count(self):
        path = self._write_npz("wrong_len.npz", self.valid_pred, self.origins.asi8, self.sensors)
        # Expected test split origin count is 6844
        with self.assertRaises(ValueError):
            load_phase2_npz(path, split="test")

    def test_rejects_mismatched_sensor_order(self):
        scrambled_sensors = list(reversed(self.sensors))
        path = self._write_npz("scrambled.npz", self.valid_pred, self.origins.asi8, scrambled_sensors)
        with self.assertRaises(ValueError):
            load_phase2_npz(path, split="any", canonical_sensors=self.sensors)

    def test_attach_phase2_predictions(self):
        # Create minimal aligned frame
        records = []
        for orig in self.origins[:5]:
            for s in self.sensors[:10]:
                row = {"origin_timestamp": orig, "sensor_id": s}
                for h in HORIZON_NAMES:
                    row[f"target_{h}"] = 55.0
                    row[f"target_valid_{h}"] = True
                records.append(row)
        base_df = pd.DataFrame(records)

        p2_data = {
            "pred": self.valid_pred[:5, :, :10],
            "origin_dt": self.origins[:5],
            "origin_ns": self.origins[:5].asi8,
            "sensor_ids": self.sensors[:10],
        }
        attached = attach_phase2_predictions(base_df, p2_data, model_col_prefix="mock_gat")
        for h in HORIZON_NAMES:
            self.assertIn(f"pred_mock_gat_{h}", attached.columns)
            self.assertFalse(attached[f"pred_mock_gat_{h}"].isna().any())


class TestPhase1ReproductionGate(unittest.TestCase):
    """Phase 1 reproduction gate: verify scoring reproduces phase1_metrics.csv within 1e-3."""

    def test_phase1_metrics_reproduction_within_tolerance(self):
        val_parquet = PRED_DIR / "person_c_val_predictions.parquet"
        if not val_parquet.is_file():
            self.skipTest(
                f"Phase 1 prediction parquets absent in {PRED_DIR}. "
                "Download phase1_predictions.zip into data/processed/predictions/ to run reproduction gate."
            )

        splits_scored = []
        for s in ["val", "test"]:
            df = load_phase1_predictions(split=s)
            p1_models = {
                "persistence": "persistence",
                "historical_average": "historical_average",
                "random_forest_32": "random_forest",
                "xgboost_32": "xgboost",
                "xgboost_ha": "xgboost_ha",
            }
            overall, _, _ = score_dataframe(df, split=s, models_map=p1_models)
            splits_scored.append(format_metrics_table(overall))

        combined = pd.concat(splits_scored, ignore_index=True)
        max_diffs = verify_phase1_reproduction(combined, PHASE1_METRICS_PATH, tol=1e-3)
        print(f"\n[Phase 1 Reproduction Gate PASSED] Max deviations: {max_diffs}")
        self.assertLessEqual(max_diffs["mae"], 1e-3)
        self.assertLessEqual(max_diffs["rmse"], 1e-3)
        self.assertLessEqual(max_diffs["mape"], 1e-3)


def _synthetic_daily(gains: dict, days: int = 24, seed: int = 0) -> pd.DataFrame:
    """Daily sums where model m has MAE 3.0 - gains[m][(horizon, period)] (+ day noise)."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(days):
        day = str((pd.Timestamp("2012-06-04") + pd.Timedelta(days=d)).date())
        for h in HORIZON_NAMES:
            for period in ("all", "peak", "off_peak"):
                n = 50_000 if period == "all" else (10_000 if period == "peak" else 40_000)
                noise = rng.normal(0, 0.02)
                for m, g in gains.items():
                    mae = 3.0 - g.get((h, period), 0.0) + noise
                    rows.append(dict(split="test", horizon=h, model=m, period=period, day=day,
                                     n=n, absolute_error_sum=mae * n))
    return pd.DataFrame(rows)


class TestHypothesesEvaluation(unittest.TestCase):
    """Verdicts are computed by code from period-filtered, paired day sums."""

    def test_missing_models_are_not_evaluated(self):
        h = evaluate_hypotheses(_synthetic_daily({"xgboost_ha": {}}))
        self.assertEqual(list(dict.fromkeys(h["id"])), ["H1", "H2", "H3", "H4", "H5", "H0"])
        self.assertTrue((h["overall_verdict"] == "not_evaluated").all())

    def test_verdicts_follow_the_rule(self):
        big = {(hh, p): 0.3 for hh in HORIZON_NAMES for p in ("all", "peak", "off_peak")}
        gains = {
            "gru_local": {}, "gat_identity": {}, "gat_shuffled": {}, "xgboost_ha": {}, "dcrnn": big,
            # gat: large gain, larger in peak, growing with horizon
            "gat": {(hh, p): (0.1 + 0.1 * k) * (2.0 if p == "peak" else 1.0) / (1.0 if p != "all" else 1.0)
                    for k, hh in enumerate(HORIZON_NAMES) for p in ("all", "peak", "off_peak")},
        }
        h = evaluate_hypotheses(_synthetic_daily(gains)).groupby("id")["overall_verdict"].first()
        self.assertEqual(h["H1"], "supported")
        self.assertEqual(h["H2"], "supported")
        self.assertEqual(h["H3"], "supported")
        self.assertEqual(h["H4"], "supported")
        self.assertEqual(h["H0"], "contradicted")

    def test_no_gain_is_inconclusive(self):
        gains = {m: {} for m in ("gru_local", "gat", "gat_identity", "gat_shuffled", "dcrnn", "xgboost_ha")}
        h = evaluate_hypotheses(_synthetic_daily(gains)).groupby("id")["overall_verdict"].first()
        self.assertEqual(h["H1"], "inconclusive")
        self.assertEqual(h["H0"], "supported")

    def test_period_rows_are_really_filtered(self):
        gains = {"gat_identity": {}, "gat": {(hh, "peak"): 0.5 for hh in HORIZON_NAMES}}
        boot = run_comparisons(_synthetic_daily(gains), [("gat_identity", "gat", ("all", "peak", "off_peak"))])
        by = boot.set_index(["horizon", "period"])["mae_difference"]
        self.assertAlmostEqual(by[("60min", "peak")], 0.5, places=6)
        self.assertAlmostEqual(by[("60min", "off_peak")], 0.0, places=6)
        self.assertAlmostEqual(by[("60min", "all")], 0.0, places=6)

    def test_single_term_contrast_equals_phase1_bootstrap(self):
        d = _synthetic_daily({"persistence": {}, "xgboost_ha": {(hh, "all"): 0.4 for hh in HORIZON_NAMES}}, seed=3)
        mine = bootstrap_contrast(d, [(1.0, "persistence", "xgboost_ha", "30min", "all")])
        ref = prediction_evaluation.paired_daily_bootstrap(
            d[d["period"] == "all"], "persistence", "xgboost_ha", "30min")
        for k in ("mae_difference", "ci95_low", "ci95_high"):
            self.assertAlmostEqual(mine[k], ref[k], places=12)


class TestEndToEndEvaluationRunner(unittest.TestCase):
    """Run scripts/phase2_eval_all.py on stand-in Phase 2 files with the real origins."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.pred_dir = Path(self.temp_dir) / "predictions"
        self.out_dir = Path(self.temp_dir) / "metadata"
        self.fig_dir = Path(self.temp_dir) / "figures"
        self.pred_dir.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_runner_scores_phase2_files_and_writes_outputs(self):
        if not (PRED_DIR / "person_c_val_predictions.parquet").is_file():
            self.skipTest("Phase 1 predictions absent; skipping end-to-end runner test.")
        sensors = load_canonical_sensor_ids()
        rng = np.random.default_rng(0)
        for split in ("val", "test"):
            df = pd.read_parquet(PRED_DIR / f"person_c_{split}_predictions.parquet",
                                 columns=["origin_timestamp", "sensor_id"] + [f"pred_xgboost_ha_{h}" for h in HORIZON_NAMES])
            origins = np.sort(df["origin_timestamp"].unique())
            oi = pd.Index(origins).get_indexer(df["origin_timestamp"])
            si = pd.Index(sensors).get_indexer(df["sensor_id"].astype(str))
            base = np.zeros((len(origins), 4, 207), np.float32)
            for k, h in enumerate(HORIZON_NAMES):
                base[oi, k, si] = df[f"pred_xgboost_ha_{h}"].to_numpy()
            for model, sd in (("gat", 0.5), ("gru_local", 1.0)):
                (self.pred_dir / model).mkdir(exist_ok=True)
                for s in (42, 43):
                    np.savez(self.pred_dir / model / f"seed{s}_{split}.npz",
                             pred=base + rng.normal(0, sd, base.shape).astype(np.float32),
                             origin_ns=origins.astype("datetime64[ns]").astype(np.int64),
                             sensor_ids=np.array(sensors))
        import subprocess
        import sys
        res = subprocess.run([sys.executable, str(ROOT / "scripts" / "phase2_eval_all.py"),
                              "--pred-root", str(self.pred_dir), "--out", str(self.out_dir),
                              "--figures-dir", str(self.fig_dir)], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stdout[-2000:] + res.stderr[-2000:])
        summary = pd.read_csv(self.out_dir / "metrics_summary.csv")
        self.assertTrue({"gat", "gru_local"} <= set(summary["model"]))
        self.assertEqual(int(summary[summary["model"] == "gat"]["n_seeds"].max()), 2)
        hyp = pd.read_csv(self.out_dir / "hypotheses.csv")
        self.assertEqual(hyp[hyp["id"] == "H1"]["overall_verdict"].iloc[0], "supported")
        parity = pd.read_json(self.out_dir / "phase1_parity.json", typ="series")
        self.assertTrue(parity["passed"])
        self.assertTrue(parity["phase1_bootstrap_reproduced"])
        self.assertTrue((self.fig_dir / "fig_horizon_mae.png").is_file())
        self.assertTrue((self.fig_dir / "fig_graph_ablation.png").is_file())


if __name__ == "__main__":
    unittest.main()

