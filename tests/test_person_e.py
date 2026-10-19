"""Unit tests for Person E evaluation metrics, benchmark logic, and spatial mapping."""
from __future__ import annotations

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.models.metrics import (
    masked_mae,
    masked_rmse,
    masked_mape,
    compute_metrics,
    compute_relative_gain,
    error_distribution_quantiles,
)
from src.models.evaluation import (
    load_person_c_results,
    load_person_d_results,
    build_unified_test_benchmark,
    build_peak_offpeak_breakdown,
    build_rf_ablation_summary,
    compute_horizon_degradation,
)
from src.visualization.spatial_mapping import (
    load_sensor_geo_metadata,
    assign_freeway_corridors,
    compute_corridor_statistics,
)


class TestPersonEMetrics(unittest.TestCase):
    """Test suite for Person E evaluation metrics adhering to benchmark rules."""

    def test_masked_mae_excludes_zero(self):
        y_true = np.array([0.0, 50.0, 60.0, 0.0])
        y_pred = np.array([20.0, 55.0, 58.0, 10.0])
        mae = masked_mae(y_true, y_pred, null_val=0.0)
        self.assertAlmostEqual(mae, 3.5, places=4)

    def test_masked_rmse_excludes_zero(self):
        y_true = np.array([0.0, 50.0, 60.0, 0.0])
        y_pred = np.array([99.0, 54.0, 57.0, 99.0])
        rmse = masked_rmse(y_true, y_pred, null_val=0.0)
        self.assertAlmostEqual(rmse, np.sqrt(12.5), places=4)

    def test_masked_mape_calculation(self):
        y_true = np.array([0.0, 50.0, 100.0])
        y_pred = np.array([10.0, 55.0, 90.0])
        mape = masked_mape(y_true, y_pred, null_val=0.0)
        self.assertAlmostEqual(mape, 10.0, places=4)

    def test_compute_metrics_dictionary(self):
        y_true = np.array([40.0, 60.0])
        y_pred = np.array([42.0, 57.0])
        res = compute_metrics(y_true, y_pred)
        self.assertIn("mae", res)
        self.assertIn("rmse", res)
        self.assertIn("mape", res)
        self.assertEqual(res["mae"], 2.5)

    def test_compute_relative_gain(self):
        gain = compute_relative_gain(5.0, 4.0)
        self.assertEqual(gain, 20.0)
        degradation = compute_relative_gain(4.0, 5.0)
        self.assertEqual(degradation, -25.0)

    def test_error_distribution_quantiles(self):
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y_pred = np.array([11.0, 22.0, 33.0, 44.0, 55.0])
        quantiles = error_distribution_quantiles(y_true, y_pred, quantiles=(0.5, 0.9))
        self.assertEqual(quantiles["p50"], 3.0)
        self.assertAlmostEqual(quantiles["p90"], 4.6, places=1)

    def test_empty_mask_handling(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([10.0, 20.0])
        self.assertEqual(masked_mae(y_true, y_pred), 0.0)
        self.assertEqual(masked_rmse(y_true, y_pred), 0.0)
        self.assertEqual(masked_mape(y_true, y_pred), 0.0)


class TestPersonEEvaluationEngine(unittest.TestCase):
    """Test suite for benchmark loading and synthesis."""

    def test_load_person_c_and_d_metadata(self):
        c_data = load_person_c_results()
        d_data = load_person_d_results()
        self.assertIn("horizons", c_data)
        self.assertIn("experiments", d_data)

    def test_build_unified_test_benchmark(self):
        df_bench = build_unified_test_benchmark()
        self.assertEqual(len(df_bench), 20)  # 4 horizons x 5 models
        self.assertIn("mae", df_bench.columns)
        self.assertIn("rmse", df_bench.columns)
        self.assertIn("mape", df_bench.columns)

        # Check 5min XGBoost (32) MAE matches 2.362
        xgb_5m = df_bench[(df_bench["horizon"] == "5min") & (df_bench["model"] == "xgboost_32")].iloc[0]
        self.assertAlmostEqual(xgb_5m["mae"], 2.362, places=3)

    def test_peak_offpeak_breakdown_structure(self):
        df_slices = build_peak_offpeak_breakdown()
        self.assertGreater(len(df_slices), 0)
        self.assertIn("persistence_mae", df_slices.columns)
        self.assertIn("xgboost_ha_mae", df_slices.columns)

    def test_rf_ablation_summary_structure(self):
        df_abl = build_rf_ablation_summary()
        self.assertEqual(len(df_abl), 16)  # 4 horizons x 4 tiers
        self.assertIn("val_mae", df_abl.columns)
        self.assertIn("test_mae", df_abl.columns)

    def test_compute_horizon_degradation(self):
        df_bench = build_unified_test_benchmark()
        df_deg = compute_horizon_degradation(df_bench)
        self.assertIn("degradation_ratio_60_to_5", df_deg.columns)
        # Persistence degradation ratio should be > 1.8
        pers_deg = df_deg[df_deg["model"] == "persistence"].iloc[0]["degradation_ratio_60_to_5"]
        self.assertGreater(pers_deg, 1.8)


class TestPersonESpatialMapping(unittest.TestCase):
    """Test suite for spatial mapping and corridor aggregation."""

    def test_assign_freeway_corridors(self):
        sample_df = pd.DataFrame([
            {"sensor_id": "1", "latitude": 34.05, "longitude": -118.45},  # Westside -> I-405
            {"sensor_id": "2", "latitude": 34.22, "longitude": -118.25},  # North -> I-210
            {"sensor_id": "3", "latitude": 34.04, "longitude": -118.35},  # Central -> I-10
        ])
        res = assign_freeway_corridors(sample_df)
        self.assertEqual(res.loc[0, "corridor"], "I-405 (San Diego Fwy)")
        self.assertEqual(res.loc[1, "corridor"], "I-210 (Foothill Fwy)")
        self.assertEqual(res.loc[2, "corridor"], "I-10 (Santa Monica Fwy)")

    def test_load_sensor_geo_metadata_complete_network(self):
        df_sensors = load_sensor_geo_metadata()
        self.assertEqual(len(df_sensors), 207)
        self.assertIn("corridor", df_sensors.columns)
        self.assertIn("morning_peak_speed", df_sensors.columns)
        self.assertIn("evening_peak_speed", df_sensors.columns)

    def test_compute_corridor_statistics(self):
        df_sensors = load_sensor_geo_metadata()
        stats = compute_corridor_statistics(df_sensors)
        self.assertGreaterEqual(len(stats), 4)
        self.assertIn("morning_speed_drop_pct", stats.columns)
        self.assertIn("evening_speed_drop_pct", stats.columns)


if __name__ == "__main__":
    unittest.main()
