"""Synthetic fixtures exercise quality rules; they are not METR-LA evidence."""

import json
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.data.quality import analyze_frame


class QualityTests(unittest.TestCase):
    def inspect(self, frame, **kwargs):
        config = {"expected_sensors": frame.shape[1]}
        config.update(kwargs.pop("config", {}))
        return analyze_frame(frame, config=config, **kwargs)

    def frame(self, values, columns=None):
        array = np.asarray(values, dtype=float)
        if array.ndim == 1:
            array = array[:, None]
        return pd.DataFrame(array, index=pd.date_range("2020-01-01", periods=len(array), freq="5min"),
                            columns=columns or [str(i) for i in range(array.shape[1])])

    def test_masks_invalid_readings_without_changing_raw_data_or_order(self):
        frame = self.frame([[50, 0], [np.nan, 0.01], [np.inf, -2], [60, -np.inf]], ["B", "A"])
        before = frame.copy(deep=True)
        result = self.inspect(frame)
        assert_frame_equal(frame, before)
        self.assertEqual(result["sensors"]["sensor_id"].tolist(), ["B", "A"])
        self.assertEqual(result["valid_speed"].columns.tolist(), ["B", "A"])
        self.assertEqual(result["valid_speed"].iloc[1, 1], 0.01)
        self.assertEqual(result["summary"]["value_counts"], {
            "missing_nan": 1, "infinite": 2, "zero": 1, "negative_finite": 1,
            "valid_positive_finite": 3, "invalid_total": 5,
        })
        self.assertEqual(result["valid_speed"].isna().sum().sum(), 5)
        json.dumps(result["summary"], allow_nan=False)

    def test_zero_runs_include_short_runs_and_flag_long_runs(self):
        frame = self.frame([0] * 12 + [30] + [0, 0] + [40])
        result = self.inspect(frame)
        runs = result["zero_runs"]
        self.assertEqual(runs["length"].tolist(), [12, 2])
        self.assertEqual(runs["possible_outage"].tolist(), [True, False])
        sensor = result["sensors"].iloc[0]
        self.assertEqual(sensor["zero_run_count"], 2)
        self.assertEqual(sensor["max_zero_run"], 12)
        self.assertEqual(result["summary"]["zero_evidence"]["possible_outage_runs"], 1)
        self.assertEqual(result["summary"]["zero_evidence"]["all_zero_timestamp_rows"], 14)

    def test_zero_and_flatline_runs_break_at_missing_timestamps(self):
        frame = self.frame([[0, 50]] * 14)
        frame.index = frame.index[:7].append(frame.index[7:] + pd.Timedelta(minutes=5))
        result = self.inspect(frame)
        self.assertEqual(result["zero_runs"]["length"].tolist(), [7, 7])
        self.assertTrue(result["flatline_runs"].empty)
        self.assertEqual(result["sensors"].iloc[1]["max_flatline_run"], 7)
        self.assertEqual(result["summary"]["timestamp_validation"]["missing_timestamp_count"], 1)
        self.assertEqual(result["timestamp_gaps"].iloc[0]["expected_missing_slots"], 1)
        self.assertEqual(result["summary"]["dimensions"]["missing_timestamp_cells"], 2)

    def test_positive_constant_run_is_flagged_but_retained(self):
        frame = self.frame([42] * 12 + [43, 42])
        result = self.inspect(frame)
        self.assertEqual(result["flatline_runs"]["length"].tolist(), [12])
        self.assertEqual(result["sensors"].iloc[0]["flatline_run_count"], 1)
        assert_frame_equal(result["valid_speed"], frame)

    def test_changes_require_consecutive_valid_observations(self):
        frame = self.frame([10, 40, 0, 80, 10, 60])
        frame.index = frame.index[:5].append(frame.index[5:] + pd.Timedelta(minutes=5))
        result = self.inspect(frame)
        self.assertEqual(result["discontinuities"]["speed_change"].tolist(), [30.0, -70.0])

    def test_duplicate_timestamp_breaks_changes_and_runs(self):
        frame = self.frame([0, 0, 0, 0])
        frame.index = pd.DatetimeIndex(["2020-01-01 00:00", "2020-01-01 00:05",
                                       "2020-01-01 00:05", "2020-01-01 00:10"])
        result = self.inspect(frame)
        self.assertEqual(result["zero_runs"]["length"].tolist(), [1, 1, 1, 1])
        self.assertEqual(result["summary"]["timestamp_validation"]["duplicate_rows"], 1)
        self.assertEqual(result["summary"]["validation_status"], "fail")

    def test_unordered_data_is_reported_and_analysis_view_preserves_order(self):
        frame = self.frame([10, 40, 70]).iloc[[2, 0, 1]]
        result = self.inspect(frame)
        self.assertFalse(result["summary"]["timestamp_validation"]["chronologically_ordered"])
        self.assertEqual(result["discontinuities"]["speed_change"].tolist(), [30.0, 30.0])
        self.assertTrue(result["valid_speed"].index.equals(frame.index))
        self.assertEqual(result["summary"]["validation_status"], "fail")

    def test_offgrid_timestamps_do_not_get_silently_resampled(self):
        frame = self.frame([10, 20, 30])
        frame.index = pd.DatetimeIndex(["2020-01-01 00:00", "2020-01-01 00:06", "2020-01-01 00:10"])
        result = self.inspect(frame)
        timestamps = result["summary"]["timestamp_validation"]
        self.assertEqual(timestamps["off_grid_timestamp_count"], 1)
        self.assertEqual(timestamps["missing_timestamp_count"], 1)
        self.assertEqual(len(result["valid_speed"]), 3)
        self.assertTrue(result["discontinuities"].empty)

    def test_empty_and_all_invalid_sensors_have_explicit_evidence(self):
        frame = self.frame([[0, np.nan], [0, np.inf], [0, -2]])
        result = self.inspect(frame)
        self.assertEqual(result["summary"]["sensor_quality"]["no_valid_observations"], 2)
        self.assertEqual(result["summary"]["sensor_quality"]["all_zero_sensors"], 1)
        self.assertEqual(result["summary"]["sensor_quality"]["raw_constant_sensors"], 1)
        self.assertTrue(result["sensors"]["valid_mean"].isna().all())
        self.assertEqual(result["summary"]["zero_evidence"]["all_invalid_timestamp_rows"], 3)
        json.dumps(result["summary"], allow_nan=False)

    def test_metadata_is_joined_by_id_in_data_order(self):
        frame = self.frame([[10, 30], [20, 40]], ["B", "A"])
        locations = pd.DataFrame({"sensor_id": ["A", "B", "C"], "latitude": [1, 2, 3],
                                  "longitude": [-1, -2, -3]})
        result = self.inspect(frame, locations=locations)
        self.assertEqual(result["sensors"]["latitude"].tolist(), [2, 1])
        self.assertEqual(result["sensors"]["longitude"].tolist(), [-2, -1])
        self.assertEqual(result["summary"]["metadata_validation"]["extra_sensor_ids"], ["C"])

    def test_duplicate_metadata_is_not_ambiguously_joined(self):
        frame = self.frame([10, 20], ["A"])
        locations = pd.DataFrame({"sensor_id": ["A", "A"], "latitude": [1, 2]})
        result = self.inspect(frame, locations=locations)
        self.assertEqual(result["summary"]["metadata_validation"]["duplicate_sensor_ids"], 1)
        self.assertNotIn("latitude", result["sensors"])

    def test_graph_coverage_order_and_matrix_shape(self):
        frame = self.frame([[10, 30], [20, 40]], ["B", "A"])
        for ids, adjacency in [(["A", "B"], np.eye(2)), (["B", "C"], np.eye(2)),
                               (["B", "A"], np.eye(3)), (["B", "A"], np.full((2, 2), np.inf))]:
            with self.subTest(ids=ids, shape=adjacency.shape):
                result = self.inspect(frame, graph_sensor_ids=ids, adjacency=adjacency)
                self.assertEqual(result["summary"]["validation_status"], "fail")
                self.assertFalse(result["summary"]["graph_validation"]["used_as_features"])
        correct = self.inspect(frame, graph_sensor_ids=["B", "A"], adjacency=np.eye(2))
        self.assertEqual(correct["summary"]["validation_status"], "pass")

    def test_sensor_count_is_verified_against_configuration(self):
        frame = self.frame([10, 30])
        result = analyze_frame(frame)
        self.assertEqual(result["summary"]["validation_status"], "fail")
        self.assertIn("Expected 207 sensors", result["summary"]["issues"][0])

    def test_invalid_input_types_are_rejected(self):
        cases = [pd.DataFrame(), pd.DataFrame({"A": [1]}),
                 pd.DataFrame({"A": ["text"]}, index=pd.date_range("2020", periods=1)),
                 pd.DataFrame({"A": [True]}, index=pd.date_range("2020", periods=1)),
                 pd.DataFrame({"A": [1 + 2j]}, index=pd.date_range("2020", periods=1)),
                 pd.DataFrame([[1, 2]], columns=[1, "1"], index=pd.date_range("2020", periods=1)),
                 pd.DataFrame({"A": [1]}, index=pd.DatetimeIndex([pd.NaT]))]
        for frame in cases:
            with self.subTest(frame=repr(frame)), self.assertRaises(ValueError):
                analyze_frame(frame)

    def test_low_variance_threshold_uses_valid_positive_values(self):
        frame = self.frame([[50, 10], [0, 30], [50.1, 20], [np.nan, 40]])
        result = self.inspect(frame)
        self.assertTrue(result["sensors"].iloc[0]["is_low_variance"])
        self.assertFalse(result["sensors"].iloc[1]["is_low_variance"])
        self.assertAlmostEqual(result["sensors"].iloc[0]["valid_mean"], 50.05)

    def test_timezone_is_preserved_without_inference(self):
        frame = self.frame([10, 30])
        frame.index = frame.index.tz_localize("UTC")
        result = self.inspect(frame)
        self.assertEqual(result["summary"]["timestamp_validation"]["timezone"], "UTC")
        self.assertEqual(str(result["valid_speed"].index.tz), "UTC")


if __name__ == "__main__":
    unittest.main()
