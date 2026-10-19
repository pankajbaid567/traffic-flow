"""Unit tests for Person E evaluation metrics, benchmark logic, and spatial mapping."""
from __future__ import annotations

import unittest
import numpy as np

from src.models.metrics import (
    masked_mae,
    masked_rmse,
    masked_mape,
    compute_metrics,
    compute_relative_gain,
    error_distribution_quantiles,
)


class TestPersonEMetrics(unittest.TestCase):
    """Test suite for Person E evaluation metrics adhering to benchmark rules."""

    def test_masked_mae_excludes_zero(self):
        y_true = np.array([0.0, 50.0, 60.0, 0.0])
        y_pred = np.array([20.0, 55.0, 58.0, 10.0])
        # Only indices 1 and 2 count: |50-55|=5, |60-58|=2 -> mean 3.5
        mae = masked_mae(y_true, y_pred, null_val=0.0)
        self.assertAlmostEqual(mae, 3.5, places=4)

    def test_masked_rmse_excludes_zero(self):
        y_true = np.array([0.0, 50.0, 60.0, 0.0])
        y_pred = np.array([99.0, 54.0, 57.0, 99.0])
        # |50-54|^2 = 16, |60-57|^2 = 9 -> mean 12.5 -> sqrt = 3.5355
        rmse = masked_rmse(y_true, y_pred, null_val=0.0)
        self.assertAlmostEqual(rmse, np.sqrt(12.5), places=4)

    def test_masked_mape_calculation(self):
        y_true = np.array([0.0, 50.0, 100.0])
        y_pred = np.array([10.0, 55.0, 90.0])
        # |50-55|/50 = 0.10, |100-90|/100 = 0.10 -> mean = 10.0%
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
        y_pred = np.array([11.0, 22.0, 33.0, 44.0, 55.0])  # errors: 1, 2, 3, 4, 5
        quantiles = error_distribution_quantiles(y_true, y_pred, quantiles=(0.5, 0.9))
        self.assertEqual(quantiles["p50"], 3.0)
        self.assertAlmostEqual(quantiles["p90"], 4.6, places=1)

    def test_empty_mask_handling(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([10.0, 20.0])
        self.assertEqual(masked_mae(y_true, y_pred), 0.0)
        self.assertEqual(masked_rmse(y_true, y_pred), 0.0)
        self.assertEqual(masked_mape(y_true, y_pred), 0.0)


if __name__ == "__main__":
    unittest.main()
