"""Small synthetic fixtures verify descriptive arithmetic, not dataset facts."""

import json
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.data.descriptive import render_descriptive, summarize_descriptive
from src.data.quality import analyze_frame


class DescriptiveTests(unittest.TestCase):
    def inspect(self, frame):
        analysis = analyze_frame(frame, config={"expected_sensors": frame.shape[1]})
        return summarize_descriptive(frame, analysis)

    def test_profiles_pool_sensor_cells_instead_of_averaging_unequal_means(self):
        frame = pd.DataFrame([[10, 10], [100, 0], [40, 40]], columns=["A", "B"],
                             index=pd.to_datetime(["2020-01-06 07:00", "2020-01-06 07:05", "2020-01-11 07:00"]))
        result = self.inspect(frame)
        hourly = result["tables"]["hourly_profile"].set_index("hour")
        self.assertEqual(hourly.loc[7, "valid_count"], 5)
        self.assertEqual(hourly.loc[7, "mean_speed"], 40)
        self.assertNotEqual(hourly.loc[7, "mean_speed"], (10 + 100 + 40) / 3)
        weekday = result["tables"]["weekday_weekend_profile"].set_index("day_type")
        self.assertEqual(weekday.loc["Weekday", "mean_speed"], 40)
        self.assertEqual(weekday.loc["Weekday", "excluded_fraction"], .25)
        period = result["tables"]["traffic_period_profile"].set_index("period")
        self.assertEqual(period.loc["Weekday morning peak", "mean_speed"], 40)
        self.assertEqual(period.loc["Weekend", "valid_count"], 2)

    def test_network_zero_blocks_break_at_time_gaps_and_report_contribution(self):
        frame = pd.DataFrame([[0, 0], [0, 0], [0, 0], [0, 10]], columns=["A", "B"],
                             index=pd.to_datetime(["2020-01-01 00:00", "2020-01-01 00:05",
                                                   "2020-01-01 00:15", "2020-01-01 00:20"]))
        result = self.inspect(frame)
        self.assertEqual(result["tables"]["all_network_zero_blocks"]["length"].tolist(), [2, 1])
        evidence = result["summary"]["network_zero_evidence"]
        self.assertEqual(evidence["contiguous_blocks"], 2)
        self.assertEqual(evidence["longest_block"]["steps"], 2)
        self.assertAlmostEqual(evidence["share_of_zero_cells_in_all_zero_rows"], 6 / 7)

    def test_peak_windows_are_half_open_weekday_only(self):
        times = ["2020-01-06 06:55", "2020-01-06 07:00", "2020-01-06 09:55", "2020-01-06 10:00",
                 "2020-01-06 15:55", "2020-01-06 16:00", "2020-01-06 18:55", "2020-01-06 19:00",
                 "2020-01-11 07:00", "2020-01-12 16:00"]
        frame = pd.DataFrame({"A": np.arange(len(times)) + 1.0}, index=pd.to_datetime(times))
        periods = self.inspect(frame)["tables"]["traffic_period_profile"].set_index("period")
        self.assertEqual(periods["timestamp_count"].to_dict(), {
            "Weekday morning peak": 2, "Weekday evening peak": 2, "Weekday off-peak": 4, "Weekend": 2})

    def test_positive_near_zero_raw_values_and_analysis_are_preserved(self):
        frame = pd.DataFrame({"A": [0.01, 0, np.nan, -1, np.inf]},
                             index=pd.date_range("2020-01-01", periods=5, freq="5min"))
        before = frame.copy(deep=True)
        analysis = analyze_frame(frame, config={"expected_sensors": 1})
        valid_before = analysis["valid_speed"].copy(deep=True)
        result = summarize_descriptive(frame, analysis)
        assert_frame_equal(frame, before)
        assert_frame_equal(analysis["valid_speed"], valid_before)
        self.assertEqual(result["summary"]["overall_valid_speed"]["n"], 1)
        self.assertEqual(result["summary"]["overall_valid_speed"]["mean"], .01)
        self.assertEqual(result["summary"]["availability"]["excluded_sensor_cells"], 4)
        json.dumps(result["summary"], allow_nan=False)
        text = render_descriptive(**result)
        self.assertIn("Positive near-zero speeds remain observed", text)
        self.assertIn("unverified source units", text)
        self.assertIn("may be nonrandom", text)

    def test_all_invalid_and_no_zero_data_remain_json_serializable(self):
        for values in ([np.nan, -1], [10, 20]):
            frame = pd.DataFrame({"A": values}, index=pd.date_range("2020-01-01", periods=2, freq="5min"))
            result = self.inspect(frame)
            json.dumps(result["summary"], allow_nan=False)
            self.assertIsNone(result["summary"]["network_zero_evidence"]["longest_block"])
            self.assertIn("0** consecutive", render_descriptive(**result))

    def test_duplicate_timestamps_do_not_join_network_zero_blocks(self):
        frame = pd.DataFrame({"A": [0, 0, 0, 0]},
                             index=pd.to_datetime(["2020-01-01 00:00", "2020-01-01 00:05",
                                                   "2020-01-01 00:05", "2020-01-01 00:10"]))
        self.assertEqual(self.inspect(frame)["tables"]["all_network_zero_blocks"]["length"].tolist(), [1] * 4)

    def test_evidence_tables_rank_sensor_rates_and_run_lengths(self):
        frame = pd.DataFrame({"A": [0, 0, 10, 0, 10], "B": [0, 0, 0, 0, 10],
                              "C": [10, 10, 10, 10, 0], "D": [10, 10, 10, 10, 10]},
                             index=pd.date_range("2020-01-01", periods=5, freq="5min"))
        result = self.inspect(frame)
        self.assertEqual(result["tables"]["top_zero_sensors"]["sensor_id"].tolist(), ["B", "A", "C"])
        self.assertEqual(result["tables"]["longest_zero_runs"]["length"].tolist(), [4, 2, 1])


if __name__ == "__main__":
    unittest.main()
