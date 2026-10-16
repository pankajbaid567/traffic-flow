"""Integration and leakage prevention tests for Person B dataset pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

from src.data.dataset_builder import FEATURE_GROUPS, load_partition
from src.data.load_data import (
    apply_causal_imputation,
    check_sensor_order,
    create_missingness_mask,
    fit_training_sensor_medians,
)
from src.data.split import (
    get_eligible_sample_origins,
    get_partition_info,
    load_split_contract,
    summarize_splits,
)

PROJECT = Path(__file__).resolve().parents[1]


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.contract = load_split_contract(PROJECT / "data/metadata/split_contract.json")
        # Generate synthetic 1000-step grid for fast unit verification
        self.times = pd.date_range("2012-03-01 00:00:00", periods=1000, freq="5min")
        self.sensors = ["sensor_A", "sensor_B"]
        np.random.seed(42)
        vals = 40.0 + 10.0 * np.random.randn(1000, 2)
        self.df = pd.DataFrame(vals, index=self.times, columns=self.sensors)

    def test_split_contract_bounds(self):
        self.assertIn("partitions", self.contract)
        train_info = self.contract["partitions"]["train"]
        val_info = self.contract["partitions"]["validation"]
        test_info = self.contract["partitions"]["test"]

        # Ensure timestamps are chronological and continuous
        train_end = pd.to_datetime(train_info["end_inclusive"])
        val_start = pd.to_datetime(val_info["start_inclusive"])
        val_end = pd.to_datetime(val_info["end_inclusive"])
        test_start = pd.to_datetime(test_info["start_inclusive"])

        self.assertEqual(val_start - train_end, pd.Timedelta(minutes=5))
        self.assertEqual(test_start - val_end, pd.Timedelta(minutes=5))

    def test_boundary_rejection_rule(self):
        # Construct synthetic contract with known partitions
        synth_contract = {
            "partitions": {
                "train": {
                    "start_inclusive": "2012-03-01 00:00:00",
                    "end_inclusive": "2012-03-01 08:15:00",  # 100 steps (0 to 99)
                    "grid_timestamps": 100,
                },
                "validation": {
                    "start_inclusive": "2012-03-01 08:20:00",
                    "end_inclusive": "2012-03-01 12:25:00",  # 50 steps (100 to 149)
                    "grid_timestamps": 50,
                },
            }
        }
        train_origins, _ = get_eligible_sample_origins(
            self.times,
            "train",
            history_steps=12,
            horizon_steps=[1, 3, 6, 12],
            contract=synth_contract,
        )
        # For train: start is 0, end is 99. Earliest origin = 11, latest origin = 99 - 12 = 87
        self.assertEqual(train_origins[0], 11)
        self.assertEqual(train_origins[-1], 87)

        # For validation: start is 100, end is 149.
        # Earliest origin = 100 - 1 = 99. Latest origin = 149 - 12 = 137
        val_origins, _ = get_eligible_sample_origins(
            self.times,
            "validation",
            history_steps=12,
            horizon_steps=[1, 3, 6, 12],
            contract=synth_contract,
        )
        self.assertEqual(val_origins[0], 99)
        self.assertEqual(val_origins[-1], 137)

        # For val origin 99: target t+1 is index 100 (inside val!), target t+12 is 111 (inside val!)
        # History is 99 - 11 = 88 to 99 (observed in train partition)
        self.assertTrue((val_origins + 1 >= 100).all())
        self.assertTrue((val_origins + 12 <= 149).all())

    def test_causal_imputation_no_test_leakage(self):
        # Insert zeros into train and test
        df_leak = self.df.copy()
        df_leak.iloc[0:5, 0] = 0.0  # initial zero
        df_leak.iloc[100:105, 0] = 0.0  # gap
        mask = create_missingness_mask(df_leak)

        # Fit training medians using only up to index 200
        train_end_ts = self.times[200]
        medians_baseline = fit_training_sensor_medians(df_leak, mask, train_end_ts)

        # Alter future test data (after index 200) drastically
        df_leak.iloc[500:600, 0] = 999999.0
        mask_after = create_missingness_mask(df_leak)
        medians_after = fit_training_sensor_medians(df_leak, mask_after, train_end_ts)

        # Training medians must be completely identical regardless of future values
        pd.testing.assert_series_equal(medians_baseline, medians_after)

    def test_imputation_only_uses_past(self):
        df = self.df.copy()
        df.iloc[0:3, 0] = 0.0      # leading gap -> training median
        df.iloc[50:55, 0] = 0.0    # gap in the middle -> last value before it
        mask = create_missingness_mask(df)
        medians = fit_training_sensor_medians(df, mask, self.times[200])
        filled = apply_causal_imputation(df, mask, medians)

        self.assertTrue((filled.iloc[0:3, 0] == medians["sensor_A"]).all())
        self.assertTrue((filled.iloc[50:55, 0] == df.iloc[49, 0]).all())

        # changing a later value must not change anything filled before it
        df2 = df.copy()
        df2.iloc[55, 0] = 123.0
        filled2 = apply_causal_imputation(df2, create_missingness_mask(df2), medians)
        pd.testing.assert_frame_equal(filled.iloc[:55], filled2.iloc[:55])

    def test_sensor_order_check(self):
        sensors = pd.DataFrame({"sensor_id": ["sensor_A", "sensor_B"]})
        check_sensor_order(self.df, sensors)
        with self.assertRaises(ValueError):
            check_sensor_order(self.df, sensors.iloc[::-1])

    def test_features_yaml_matches_code(self):
        config = yaml.safe_load((PROJECT / "configs/features.yaml").read_text())
        groups = config["ablation_groups"]
        self.assertEqual(groups["lag_only"], FEATURE_GROUPS["lag_only"])
        self.assertEqual(groups["temporal"], FEATURE_GROUPS["temporal"])
        self.assertEqual(groups["rolling"], FEATURE_GROUPS["rolling"])
        self.assertEqual(groups["trend"], FEATURE_GROUPS["trend"])
        self.assertEqual(groups["missingness"], FEATURE_GROUPS["missingness"])
        self.assertEqual(config["horizon_steps"], self.contract["horizon_steps"])
        self.assertEqual(config["history_steps"], self.contract["history_steps"])

    def test_processed_parquets_exist_and_loadable(self):
        summary_path = PROJECT / "data/processed/dataset_summary.json"
        if not summary_path.exists():
            self.skipTest("dataset_builder has not generated processed parquets yet.")

        summary = json.loads(summary_path.read_text())
        self.assertIn("partitions", summary)

        for part in ["train", "validation", "test"]:
            df_part = load_partition(part, processed_dir=PROJECT / "data/processed")
            self.assertFalse(df_part.empty)
            self.assertIn("origin_timestamp", df_part.columns)
            self.assertIn("sensor_id", df_part.columns)
            self.assertIn("speed_lag_0", df_part.columns)
            self.assertIn("target_5min", df_part.columns)
            self.assertIn("target_valid_5min", df_part.columns)


if __name__ == "__main__":
    unittest.main()
