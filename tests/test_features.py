"""Unit tests for Person B feature engineering functions."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.data.features import (
    ALL_FEATURE_COLUMNS,
    LAG_COLUMNS,
    MISSINGNESS_COLUMNS,
    ROLLING_COLUMNS,
    TARGET_COLUMNS,
    TARGET_VALID_COLUMNS,
    TEMPORAL_COLUMNS,
    TREND_COLUMNS,
    compute_lag_features,
    compute_missingness_features,
    compute_rolling_features,
    compute_temporal_features,
    compute_trend_features,
    extract_targets,
)


class TestFeatures(unittest.TestCase):
    def setUp(self):
        # 100 timestamps with 5-minute sampling
        self.times = pd.date_range("2020-01-01 00:00:00", periods=100, freq="5min")
        self.sensors = ["sensor_1", "sensor_2"]
        # Monotonically increasing test values
        data = np.column_stack([
            np.arange(100, dtype=float),
            np.arange(100, dtype=float) * 2.0,
        ])
        self.df = pd.DataFrame(data, index=self.times, columns=self.sensors)

    def test_lags_causality_and_alignment(self):
        lags = compute_lag_features(self.df, lags=range(12))
        self.assertEqual(len(lags), 12)
        for k in range(12):
            col = f"speed_lag_{k}"
            self.assertIn(col, lags)
            # Row t should equal df at t - k
            val_at_15 = lags[col].iloc[15, 0]
            expected_val = self.df.iloc[15 - k, 0]
            self.assertEqual(val_at_15, expected_val)

        # Before lag k, value is NaN
        self.assertTrue(np.isnan(lags["speed_lag_5"].iloc[4, 0]))
        self.assertFalse(np.isnan(lags["speed_lag_5"].iloc[5, 0]))

    def test_rolling_features_backward_looking(self):
        rolling = compute_rolling_features(self.df)
        self.assertIn("rolling_mean_15min", rolling)
        self.assertIn("rolling_mean_30min", rolling)
        self.assertIn("rolling_mean_60min", rolling)
        self.assertIn("rolling_std_15min", rolling)
        self.assertIn("rolling_std_30min", rolling)
        self.assertIn("rolling_std_60min", rolling)
        self.assertIn("rolling_min_30min", rolling)
        self.assertIn("rolling_max_30min", rolling)

        # 15min = 3 steps (t-2, t-1, t)
        # At index 10, sensor 0 has values 8, 9, 10 -> mean = 9.0
        self.assertAlmostEqual(rolling["rolling_mean_15min"].iloc[10, 0], 9.0)

        # 30min = 6 steps (t-5 ... t)
        # At index 10: 5, 6, 7, 8, 9, 10 -> min=5.0, max=10.0, mean=7.5
        self.assertAlmostEqual(rolling["rolling_min_30min"].iloc[10, 0], 5.0)
        self.assertAlmostEqual(rolling["rolling_max_30min"].iloc[10, 0], 10.0)
        self.assertAlmostEqual(rolling["rolling_mean_30min"].iloc[10, 0], 7.5)

        # Causality verification: changing future values at t+1 must NOT change rolling feature at t
        df_modified = self.df.copy()
        df_modified.iloc[11, 0] = 9999.0
        rolling_mod = compute_rolling_features(df_modified)
        self.assertEqual(rolling["rolling_mean_15min"].iloc[10, 0], rolling_mod["rolling_mean_15min"].iloc[10, 0])

    def test_trends(self):
        trends = compute_trend_features(self.df)
        # sensor 0 increments by 1.0 each step
        self.assertAlmostEqual(trends["speed_change_5min"].iloc[10, 0], 1.0)
        self.assertAlmostEqual(trends["speed_change_15min"].iloc[10, 0], 3.0)
        self.assertAlmostEqual(trends["speed_change_30min"].iloc[10, 0], 6.0)

    def test_temporal_features(self):
        temporal = compute_temporal_features(self.times)
        self.assertEqual(len(temporal), 100)
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
            self.assertTrue((temporal[col] >= -1.0001).all())
            self.assertTrue((temporal[col] <= 1.0001).all())

        # Check weekend indicator on a known weekend date
        # 2020-01-01 was Wednesday (dow=2) -> is_weekend = 0
        self.assertEqual(temporal["is_weekend"].iloc[0], 0)
        sat_times = pd.date_range("2020-01-04 00:00:00", periods=5, freq="5min")
        sat_temp = compute_temporal_features(sat_times)
        self.assertTrue((sat_temp["is_weekend"] == 1).all())

        # Check peak indicators
        morning_times = pd.date_range("2020-01-01 08:00:00", periods=5, freq="5min")
        m_temp = compute_temporal_features(morning_times)
        self.assertTrue((m_temp["is_morning_peak"] == 1).all())
        self.assertTrue((m_temp["is_evening_peak"] == 0).all())

        evening_times = pd.date_range("2020-01-01 17:00:00", periods=5, freq="5min")
        e_temp = compute_temporal_features(evening_times)
        self.assertTrue((e_temp["is_morning_peak"] == 0).all())
        self.assertTrue((e_temp["is_evening_peak"] == 1).all())

    def test_missingness_features(self):
        mask = pd.DataFrame(False, index=self.times, columns=self.sensors)
        mask.iloc[5, 0] = True
        mask.iloc[6, 0] = True
        miss = compute_missingness_features(mask, history_window=12)
        self.assertEqual(miss["speed_missing_flag"].iloc[5, 0], 1)
        self.assertEqual(miss["speed_missing_flag"].iloc[7, 0], 0)
        # At index 7, window [0..7] contains 2 missing values
        self.assertEqual(miss["missing_count_60min"].iloc[7, 0], 2)

    def test_targets_and_validity(self):
        raw = self.df.copy()
        raw.iloc[15, 0] = 0.0  # exact zero at index 15
        vals, valids = extract_targets(raw)

        # target_5min at index 14 is raw at index 15 (which is 0.0)
        self.assertEqual(vals["target_5min"].iloc[14, 0], 0.0)
        # Under benchmark contract, exact zero target is NOT valid
        self.assertFalse(valids["target_valid_5min"].iloc[14, 0])

        # target_5min at index 13 is raw at index 14 (which is positive)
        self.assertTrue(valids["target_valid_5min"].iloc[13, 0])


if __name__ == "__main__":
    unittest.main()
