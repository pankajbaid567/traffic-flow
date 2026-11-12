"""Unit tests for Phase 2 Evaluation Harness, Statistics and Visualizations (Member 4).

Contract: reports/phase2_plan.md section 4.7, 4.8, 8.4.
Acceptance criteria:
- Phase 1 reproduction gate: rescoring reproduces phase1_metrics.csv within 1e-3.
- Corrupted inputs (shapes, origins, non-finite values, sensor order) raise ValueError.
- patch_models context manager strictly restores original MODELS mapping.
- End-to-end evaluation runner executes on synthetic mock models in temp dir.
- Hypotheses table contains all formal hypotheses H0-H5.
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
from src.phase2.evaluation.bootstrap import (
    evaluate_hypotheses,
    execute_all_bootstrap_comparisons,
    verify_phase1_bootstrap_reproduction,
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


class TestHypothesesEvaluation(unittest.TestCase):
    """Test formal hypothesis evaluation logic and verdict rules."""

    def test_hypotheses_verdicts_with_empty_and_synthetic_data(self):
        # Empty bootstrap returns all H0-H5 with 'not_evaluated' or 'descriptive_only'
        empty_bootstrap = pd.DataFrame()
        h_df = evaluate_hypotheses(empty_bootstrap)
        self.assertEqual(len(h_df), 6)
        expected_ids = ["H1", "H2", "H3", "H4", "H5", "H0"]
        self.assertEqual(list(h_df["id"]), expected_ids)

        # Synthetic bootstrap where gat beats gru_local (H1 supported)
        synth_boot = pd.DataFrame([
            {
                "split": "test", "baseline": "gru_local", "candidate": "gat",
                "horizon": "30min", "period": "all", "days": 24, "block_days": 2,
                "repetitions": 2000, "nonempty_repetitions": 2000, "seed": 42,
                "mae_difference": 0.25, "ci95_low": 0.15, "ci95_high": 0.35,
            },
            {
                "split": "test", "baseline": "gru_local", "candidate": "gat",
                "horizon": "60min", "period": "all", "days": 24, "block_days": 2,
                "repetitions": 2000, "nonempty_repetitions": 2000, "seed": 42,
                "mae_difference": 0.40, "ci95_low": 0.20, "ci95_high": 0.60,
            },
        ])
        h_res = evaluate_hypotheses(synth_boot)
        h1_row = h_res[h_res["id"] == "H1"].iloc[0]
        self.assertEqual(h1_row["verdict"], "supported")


class TestEndToEndEvaluationRunner(unittest.TestCase):
    """Test phase2_eval_all execution on synthetic mock predictions in temp directory."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.pred_dir = Path(self.temp_dir) / "predictions"
        self.out_dir = Path(self.temp_dir) / "metadata"
        self.fig_dir = Path(self.temp_dir) / "figures"
        self.pred_dir.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_runner_on_synthetic_fake_models(self):
        val_parquet = PRED_DIR / "person_c_val_predictions.parquet"
        if not val_parquet.is_file():
            self.skipTest("Phase 1 predictions absent; skipping end-to-end runner test.")

        # Create synthetic fake Phase 2 models in temp pred_dir
        canon_sensors = load_canonical_sensor_ids()
        for model_id in ["gat", "gru_local"]:
            model_sub = self.pred_dir / model_id
            model_sub.mkdir()
            for s in [42, 43]:
                for split, n_orig in [("val", 3416), ("test", 6844)]:
                    fake_pred = np.random.uniform(45.0, 60.0, size=(n_orig, 4, 207)).astype(np.float32)
                    origin_ns = np.arange(n_orig, dtype=np.int64) * 300_000_000_000
                    np.savez(
                        model_sub / f"seed{s}_{split}.npz",
                        pred=fake_pred,
                        origin_ns=origin_ns,
                        sensor_ids=canon_sensors,
                    )

        import subprocess
        cmd = [
            str(ROOT / ".phase2-venv" / "bin" / "python"),
            str(ROOT / "scripts" / "phase2_eval_all.py"),
            "--pred-root", str(self.pred_dir),
            "--out", str(self.out_dir),
            "--figures-dir", str(self.fig_dir),
            "--splits", "val", "test",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Runner failed:\n{res.stderr}")
        self.assertTrue((self.out_dir / "metrics.csv").is_file())
        self.assertTrue((self.out_dir / "metrics_summary.csv").is_file())
        self.assertTrue((self.out_dir / "hypotheses.csv").is_file())
        self.assertTrue((self.fig_dir / "fig_horizon_mae.png").is_file())


if __name__ == "__main__":
    unittest.main()

