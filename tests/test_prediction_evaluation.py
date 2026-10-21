"""Regression checks for independent evaluation and offline model inference."""

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from src.models.metrics import compute_metrics, masked_mae
from src.models.prediction_evaluation import (
    align_predictions,
    score_frame,
    region_scores,
    paired_daily_bootstrap,
    HORIZONS,
    MODELS,
)
from src.models.saved_booster import SavedBooster


def frames():
    times = pd.to_datetime(
        ["2012-06-04 07:00", "2012-06-04 07:00", "2012-06-04 17:00", "2012-06-04 17:00"]
    )
    c = pd.DataFrame(
        {
            "origin_timestamp": times,
            "sensor_id": ["a", "b", "a", "b"],
            "is_morning_peak": [1, 1, 0, 0],
            "is_evening_peak": [0, 0, 1, 1],
        }
    )
    d = c.copy()
    for h in HORIZONS:
        c[f"target_{h}"] = [10.0, 20.0, 0.0, 40.0]
        c[f"target_valid_{h}"] = [True, True, False, True]
        d[f"target_{h}"] = c[f"target_{h}"]
        d[f"target_valid_{h}"] = c[f"target_valid_{h}"]
        for model in ["persistence", "historical_average", "xgboost", "xgboost_ha"]:
            c[f"pred_{model}_{h}"] = [11.0, 18.0, 999.0, 43.0]
        d[f"pred_random_forest_{h}"] = [11.0, 18.0, 999.0, 43.0]
    return c, d


class IndependentEvaluationTests(unittest.TestCase):
    def test_corrupted_alignment_and_flags_raise(self):
        c, d = frames()
        bad = d.iloc[::-1].reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "alignment"):
            align_predictions(c, bad)
        c.loc[2, "target_valid_5min"] = True
        d.loc[2, "target_valid_5min"] = True
        with self.assertRaisesRegex(ValueError, "validity"):
            align_predictions(c, d)

    def test_invalid_predictions_raise_instead_of_disappearing(self):
        c, d = frames()
        c.loc[0, "pred_xgboost_5min"] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            align_predictions(c, d)
        with self.assertRaisesRegex(ValueError, "finite"):
            compute_metrics(np.array([10.0]), np.array([np.nan]))

    def test_sensor_and_region_metrics_reconstruct_overall(self):
        c, d = frames()
        merged = align_predictions(c, d)
        metrics, sensors, daily = score_frame(merged, "test", {"a": 50, "b": 60})
        allrow = metrics[
            (metrics.horizon == "5min")
            & (metrics.model == "random_forest_32")
            & (metrics.period == "all")
        ].iloc[0]
        self.assertEqual(allrow.n, 3)
        self.assertAlmostEqual(allrow.mae, 2.0)
        s = sensors[
            (sensors.horizon == "5min")
            & (sensors.model == "random_forest_32")
            & (sensors.period == "all")
        ]
        self.assertEqual(int(s.n.sum()), 3)
        self.assertAlmostEqual(s.absolute_error_sum.sum() / s.n.sum(), allrow.mae)
        geo = pd.DataFrame(
            {
                "sensor_id": ["a", "b"],
                "latitude": [34.05, 34.05],
                "longitude": [-118.45, -118.45],
            }
        )
        reg = region_scores(sensors, geo)
        r = reg[
            (reg.horizon == "5min")
            & (reg.model == "random_forest_32")
            & (reg.period == "all")
        ].iloc[0]
        self.assertAlmostEqual(r.mae, allrow.mae)
        self.assertEqual(r.n, allrow.n)

    def test_paired_day_blocks_preserve_known_effect(self):
        rows = []
        for day in range(4):
            for model, error in [("base", 3), ("candidate", 1)]:
                rows.append(
                    dict(
                        split="test",
                        horizon="15min",
                        model=model,
                        day=f"2012-06-0{day+1}",
                        n=10,
                        absolute_error_sum=error * 10,
                    )
                )
        result = paired_daily_bootstrap(
            pd.DataFrame(rows), "base", "candidate", "15min", repetitions=100
        )
        for k in ["mae_difference", "ci95_low", "ci95_high"]:
            self.assertAlmostEqual(result[k], 2.0)

    def test_nonfinite_targets_excluded_and_empty_is_not_perfect(self):
        metrics = compute_metrics(
            np.array([10.0, 0.0, -1.0, np.inf, np.nan]),
            np.array([12.0, 99.0, 99.0, 99.0, 99.0]),
        )
        self.assertEqual(metrics["mae"], 2.0)
        self.assertTrue(np.isnan(masked_mae(np.array([0.0]), np.array([10.0]))))


class SavedBoosterTests(unittest.TestCase):
    def test_numeric_threshold_missing_direction_and_best_iteration(self):
        tree = {
            "left_children": [1, -1, -1],
            "right_children": [2, -1, -1],
            "split_indices": [0, 0, 0],
            "split_conditions": [10.0, 1.0, -2.0],
            "default_left": [1, 0, 0],
            "split_type": [0, 0, 0],
            "categories": [],
            "loss_changes": [1.0, 0.0, 0.0],
        }
        extra = {
            "left_children": [-1],
            "right_children": [-1],
            "split_indices": [0],
            "split_conditions": [999.0],
            "default_left": [0],
            "split_type": [0],
            "categories": [],
            "loss_changes": [0.0],
        }
        document = {
            "learner": {
                "learner_model_param": {
                    "num_target": "1",
                    "num_class": "0",
                    "base_score": "[0.5]",
                },
                "attributes": {"best_iteration": "0"},
                "feature_names": ["f"],
                "objective": {"name": "reg:absoluteerror"},
                "gradient_booster": {
                    "name": "gbtree",
                    "model": {
                        "trees": [tree, extra],
                        "tree_info": [0, 0],
                        "iteration_indptr": [0, 1, 2],
                    },
                },
            }
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tree.json"
            path.write_text(json.dumps(document))
            booster = SavedBooster(path)
            df = pd.DataFrame(
                {"f": [9.0, 10.0, np.nan], "speed_lag_0": [50.0, 50.0, 50.0]}
            )
            np.testing.assert_allclose(
                booster.predict_frame(df, ["f"]), [51.5, 48.5, 51.5]
            )
            with self.assertRaisesRegex(ValueError, "feature order"):
                booster.predict_frame(df, ["other"])


if __name__ == "__main__":
    unittest.main()
