"""Unit tests for Person D deliverables (Random Forest and feature ablations)."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models.metrics import compute_metrics, masked_mae, masked_mape, masked_rmse
from src.models.random_forest import build_model, load_feature_groups, train_and_evaluate


class TestPersonD(unittest.TestCase):
    def test_feature_groups_structure(self):
        groups = load_feature_groups()
        self.assertIn("lag_only", groups)
        self.assertIn("lag_temporal", groups)
        self.assertIn("lag_temporal_rolling", groups)
        self.assertIn("full_features", groups)

        self.assertEqual(len(groups["lag_only"]), 12)
        self.assertEqual(len(groups["lag_temporal"]), 19)
        self.assertEqual(len(groups["lag_temporal_rolling"]), 27)
        self.assertEqual(len(groups["full_features"]), 32)

    def test_metrics_accuracy(self):
        y_true = np.array([50.0, 60.0, 0.0, 70.0])
        y_pred = np.array([52.0, 58.0, 10.0, 70.0])

        # Null val = 0.0 masks the 3rd element
        mae = masked_mae(y_true, y_pred, null_val=0.0)
        self.assertAlmostEqual(mae, (2.0 + 2.0 + 0.0) / 3.0)

        rmse = masked_rmse(y_true, y_pred, null_val=0.0)
        self.assertAlmostEqual(rmse, np.sqrt((4.0 + 4.0 + 0.0) / 3.0))

        mape = masked_mape(y_true, y_pred, null_val=0.0)
        expected_mape = ((2.0 / 50.0 + 2.0 / 60.0 + 0.0 / 70.0) / 3.0) * 100.0
        self.assertAlmostEqual(mape, expected_mape)

        metrics = compute_metrics(y_true, y_pred, null_val=0.0)
        self.assertEqual(list(metrics.keys()), ["mae", "rmse", "mape"])

    def test_rf_model_builder(self):
        model = build_model(n_estimators=10, max_depth=5, max_samples=0.5, random_state=42)
        self.assertEqual(model.n_estimators, 10)
        self.assertEqual(model.max_depth, 5)
        self.assertEqual(model.max_samples, 0.5)
        self.assertEqual(model.random_state, 42)

    def test_train_and_evaluate_synthetic(self):
        n = 100
        rng = np.random.default_rng(42)
        feats = ["speed_lag_0", "speed_lag_1"]
        train_df = pd.DataFrame({
            "speed_lag_0": rng.uniform(40, 70, n),
            "speed_lag_1": rng.uniform(40, 70, n),
            "target_valid_15min": [True] * (n - 5) + [False] * 5,
        })
        train_df["target_15min"] = train_df["speed_lag_0"] * 0.9 + rng.normal(0, 1, n)

        val_df = pd.DataFrame({
            "speed_lag_0": rng.uniform(40, 70, 50),
            "speed_lag_1": rng.uniform(40, 70, 50),
            "target_valid_15min": [True] * 50,
        })
        val_df["target_15min"] = val_df["speed_lag_0"] * 0.9 + rng.normal(0, 1, 50)

        res, model = train_and_evaluate(
            train_df=train_df,
            val_df=val_df,
            test_df=None,
            features=feats,
            horizon="15min",
            n_estimators=5,
            max_depth=3,
            max_samples=0.5,
            evaluate_test=False,
        )

        self.assertEqual(res["horizon"], "15min")
        self.assertEqual(res["n_features"], 2)
        self.assertEqual(res["train_samples"], 95)
        self.assertIn("mae", res["val_metrics"])
        self.assertIn("feature_importances", res)
        self.assertEqual(len(res["feature_importances"]), 2)


if __name__ == "__main__":
    unittest.main()
