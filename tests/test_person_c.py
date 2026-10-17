"""Unit tests for Person C deliverables (baselines, XGBoost, period scoring)."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models.baselines import HA_FEATURE_COLUMNS, HistoricalAverage, persistence_predict
from src.models.xgboost_model import predict, score_by_period, train_horizon, period_masks


def make_raw(days: int = 14, sensors: int = 2) -> pd.DataFrame:
    """Speed depends only on weekday and slot, so the historical average is exact."""
    idx = pd.date_range("2012-03-05", periods=days * 288, freq="5min")  # starts on a Monday
    slot = (idx.hour * 60 + idx.minute) // 5
    base = 40.0 + 0.1 * slot.to_numpy() + 5.0 * idx.dayofweek.to_numpy()
    data = {str(i): base + i for i in range(sensors)}
    return pd.DataFrame(data, index=idx)


class TestHistoricalAverage(unittest.TestCase):
    def test_lookup_uses_target_time_and_skips_zeros(self):
        raw = make_raw()
        raw.iloc[5, 0] = 0.0  # a zero reading must not pull the mean down
        ha = HistoricalAverage().fit(raw, raw.index[-1])
        origin = pd.Series([raw.index[100]])
        pred = ha.predict(origin, np.array([0]), horizon_steps=12)
        expected = raw.iloc[100 + 12, 0]  # the pattern repeats weekly, so the slot value is exact
        self.assertAlmostEqual(float(pred[0]), expected)

        # slot of row 5 is only affected through its own cell, which still averages valid readings
        at5 = ha.predict(pd.Series([raw.index[5 - 1]]), np.array([0]), horizon_steps=1)
        self.assertAlmostEqual(float(at5[0]), make_raw().iloc[5, 0])

    def test_fit_ignores_data_after_train_end(self):
        raw = make_raw()
        raw.iloc[7 * 288:, :] = 999.0  # second week looks wildly different
        ha = HistoricalAverage().fit(raw, raw.index[7 * 288 - 1])
        pred = ha.predict(pd.Series([raw.index[0]]), np.array([0]), horizon_steps=1)
        self.assertLess(float(pred[0]), 200.0)

    def test_leave_one_out_removes_own_reading(self):
        raw = make_raw()
        raw.iloc[10, 0] = raw.iloc[10, 0] + 50.0  # outlier in week 1
        ha = HistoricalAverage().fit(raw, raw.index[-1])
        ts = pd.DatetimeIndex([raw.index[10]])
        full = ha.lookup(ts, np.array([0]))[0]
        loo = ha.lookup(ts, np.array([0]), exclude_self=True)[0]
        # with the outlier removed, the other week's clean reading is all that is left
        self.assertAlmostEqual(loo, make_raw().iloc[10 + 7 * 288, 0])
        self.assertGreater(full, loo)

    def test_exclude_self_outside_train_raises(self):
        raw = make_raw()
        ha = HistoricalAverage().fit(raw, raw.index[7 * 288 - 1])
        with self.assertRaises(ValueError):
            ha.lookup(pd.DatetimeIndex([raw.index[-1]]), np.array([0]), exclude_self=True)

    def test_feature_columns(self):
        raw = make_raw()
        ha = HistoricalAverage().fit(raw, raw.index[-1])
        f = ha.features(pd.Series(raw.index[:5]), np.zeros(5, dtype=int), np.full(5, 50.0), 3, training_rows=True)
        self.assertEqual(list(f.columns), HA_FEATURE_COLUMNS)
        np.testing.assert_allclose(f["ha_delta"], f["ha_target"] - f["ha_origin"], atol=1e-4)


def make_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lag0 = rng.uniform(20, 70, n)
    df = pd.DataFrame({
        "speed_lag_0": lag0,
        "speed_lag_1": lag0 + rng.normal(0, 2, n),
        "is_morning_peak": (np.arange(n) % 10 < 2).astype(np.int8),
        "is_evening_peak": (np.arange(n) % 10 == 5).astype(np.int8),
    })
    df["target_15min"] = lag0 * 0.9 + 5 + rng.normal(0, 1, n)
    df["target_valid_15min"] = True
    df.loc[:4, "target_valid_15min"] = False
    df.loc[:4, "target_15min"] = 0.0
    return df


class TestXGBoost(unittest.TestCase):
    def test_train_predict_and_scoring(self):
        train, val = make_frame(400, 1), make_frame(150, 2)
        feats = ["speed_lag_0", "speed_lag_1"]
        booster, info = train_horizon(train, val, feats, "15min", params={"max_depth": 3}, max_rounds=60, n_jobs=1)
        self.assertEqual(info["train_samples"], 395)  # five invalid targets dropped

        preds = predict(booster, val, feats)
        self.assertEqual(len(preds), len(val))  # predicts every row, validity applied at scoring

        scored = score_by_period(val, "15min", {"xgboost": preds, "persistence": persistence_predict(val)})
        self.assertEqual(set(scored["xgboost"]), {"all", "peak", "off_peak", "morning_peak", "evening_peak"})
        self.assertLess(scored["xgboost"]["all"]["mae"], scored["persistence"]["all"]["mae"])
        self.assertEqual(scored["xgboost"]["all"]["n"], 145)

    def test_period_masks_partition(self):
        df = make_frame(100, 3)
        m = period_masks(df)
        self.assertTrue(np.array_equal(m["peak"], ~m["off_peak"]))
        self.assertTrue(np.array_equal(m["peak"], m["morning_peak"] | m["evening_peak"]))

    def test_zero_target_rows_do_not_count(self):
        df = make_frame(50, 4)
        p = df["speed_lag_0"].to_numpy()
        res = score_by_period(df, "15min", {"m": p})
        self.assertEqual(res["m"]["all"]["n"], 45)


if __name__ == "__main__":
    unittest.main()
