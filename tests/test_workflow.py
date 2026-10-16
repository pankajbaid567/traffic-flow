"""Workflow checks use temporary outputs and explicitly synthetic test speeds."""

from __future__ import annotations

import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from scripts.acquire_data import copy_immutable, is_hdf5, sha256
from scripts import acquire_data
from src.data.quality import analyze_frame
from src.data.run_person_a import (
    analyze_context,
    load_context,
    load_graph_inventory,
    markdown_table,
    split_contract,
    verify_manifest,
    write_analysis_outputs,
    write_blocked_outputs,
)


PROJECT = Path(__file__).resolve().parents[1]
GRAPH_SOURCE = PROJECT / "data/raw"


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="person-a-workflow-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def graph_fixture(self):
        """Copy the existing upstream graph, never synthetic project metadata."""
        files = {}
        for filename in ("adj_mx.pkl", "graph_sensor_ids.txt", "graph_sensor_locations.csv"):
            source = GRAPH_SOURCE / filename
            target = self.root / "data/raw" / filename
            copy_immutable(source, target)
            files[str(target.relative_to(self.root))] = {
                "bytes": target.stat().st_size, "sha256": sha256(target),
                "source_path": str(source),
            }
        manifest = {"files": files, "speed_data_present": False}
        (self.root / "data/raw/manifest.json").write_text(json.dumps(manifest))
        return manifest

    def analysis_fixture(self, *, structural_failure=False):
        """This small generated frame is solely an integration-test fixture."""
        index = pd.date_range("2020-01-01", periods=100, freq="5min")
        values = np.column_stack((40 + 10 * np.sin(np.arange(100)),
                                  45 + 10 * np.cos(np.arange(100))))
        values[:12, 0] = 0
        values[15, 1] = np.nan
        values[25, 1] = 0.01
        frame = pd.DataFrame(values, index=index, columns=["test-A", "test-B"])
        if structural_failure:
            frame = frame.drop(index[30])
        config = {
            "frequency": "5min", "history_steps": 12, "horizon_steps": [1, 3, 6, 12],
            "split": {"train_fraction": 0.70, "validation_fraction": 0.10, "test_fraction": 0.20},
        }
        analysis = analyze_frame(frame, config={"expected_sensors": 2})
        context = {"df": frame, "config": config,
                   "manifest": {"files": {"data/raw/metr-la.h5": {"sha256": "a" * 64}}}}
        return context, analysis

    def test_split_uses_complete_grid_with_fixed_boundaries(self):
        grid = pd.date_range("2020-01-01", periods=100, freq="5min")
        observed = grid.delete([20, 21, 80])
        config = {"frequency": "5min", "history_steps": 12, "horizon_steps": [1, 3, 6, 12],
                  "split": {"train_fraction": 0.70, "validation_fraction": 0.10, "test_fraction": 0.20}}
        contract = split_contract(observed, config)
        partitions = contract["partitions"]
        self.assertEqual([partitions[name]["grid_timestamps"] for name in ("train", "validation", "test")],
                         [70, 10, 20])
        self.assertEqual(partitions["train"]["end_inclusive"], str(grid[69]))
        self.assertEqual(partitions["validation"]["start_inclusive"], str(grid[70]))
        self.assertEqual(partitions["test"]["start_inclusive"], str(grid[80]))
        self.assertEqual(contract["history_steps"], 12)
        self.assertEqual(contract["horizon_steps"], [1, 3, 6, 12])
        self.assertIn("All targets", contract["rule"])

    def test_manifest_detects_changed_and_absent_raw_files(self):
        manifest = self.graph_fixture()
        self.assertEqual(verify_manifest(self.root), manifest)
        target = self.root / "data/raw/graph_sensor_ids.txt"
        original = target.read_bytes()
        target.write_bytes(original + b"changed")
        with self.assertRaisesRegex(ValueError, "checksum changed"):
            verify_manifest(self.root)
        target.unlink()
        with self.assertRaisesRegex(ValueError, "absent"):
            verify_manifest(self.root)

    def test_raw_copy_is_idempotent_and_refuses_different_bytes(self):
        source = self.root / "source.bin"
        target = self.root / "raw/canonical.bin"
        source.write_bytes(b"original raw bytes")
        copy_immutable(source, target)
        copy_immutable(source, target)
        digest = sha256(target)
        source.write_bytes(b"different source bytes")
        with self.assertRaises(FileExistsError):
            copy_immutable(source, target)
        self.assertEqual(sha256(target), digest)
        self.assertEqual(target.read_bytes(), b"original raw bytes")

    def test_hdf5_signature_rejects_html_without_claiming_table_validity(self):
        path = self.root / "download"
        path.write_bytes(b"<!doctype html><title>Drive</title>")
        self.assertFalse(is_hdf5(path))
        path.write_bytes(b"\x89HDF\r\n\x1a\nheader-only-test-fixture")
        self.assertTrue(is_hdf5(path))
        # This tests only header classification, not a real HDF5 table reader.

    def test_graph_inventory_matches_upstream_ids_mapping_and_locations(self):
        self.graph_fixture()
        locations, ids, adjacency, checks = load_graph_inventory(self.root)
        self.assertEqual(len(ids), 207)
        self.assertEqual(adjacency.shape, (207, 207))
        self.assertEqual(set(ids), set(locations["sensor_id"]))
        self.assertTrue(checks["id_mapping_matches_order"])
        self.assertTrue(checks["sensor_text_order_matches"])
        self.assertEqual(checks["speed_alignment_status"], "pending_speed_file")
        path = self.root / "data/raw/graph_sensor_ids.txt"
        path.write_text(",".join(reversed(ids)))
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            load_graph_inventory(self.root)

    def test_missing_speed_blocks_before_hdf_reader_and_emits_no_measured_counts(self):
        self.graph_fixture()
        with self.assertRaisesRegex(FileNotFoundError, "speed data absent") as error:
            load_context(self.root)
        write_blocked_outputs(str(error.exception), self.root)
        metadata = self.root / "data/metadata"
        summary = json.loads((metadata / "quality_summary.json").read_text())
        self.assertEqual(summary["validation_status"], "blocked")
        self.assertFalse(summary["speed_statistics_computed"])
        self.assertNotIn("value_counts", summary)
        inventory = pd.read_csv(metadata / "sensors.csv", dtype={"sensor_id": str})
        self.assertEqual(len(inventory), 207)
        self.assertTrue(inventory["speed_alignment_status"].eq("pending_speed_validation").all())
        self.assertNotIn("valid_mean", inventory)
        self.assertEqual(len(pd.read_csv(metadata / "missingness_summary.csv")), 0)
        self.assertFalse((metadata / "split_contract.json").exists())
        self.assertFalse((self.root / "figures").exists())
        report = (self.root / "reports/data_quality_report.md").read_text()
        self.assertIn("not measured", report)
        self.assertIn("BLOCKED", report)

    def test_blocked_rerun_archives_previous_measured_outputs(self):
        self.graph_fixture()
        context, analysis = self.analysis_fixture()
        write_analysis_outputs(context, analysis, self.root, plots=False)
        figure = self.root / "figures/previous_test_figure.png"
        figure.parent.mkdir()
        figure.write_bytes(b"test-only prior figure artifact")
        write_blocked_outputs("test-only missing source", self.root)
        metadata = self.root / "data/metadata"
        summary = json.loads((metadata / "quality_summary.json").read_text())
        archive = self.root / summary["previous_outputs_archived"]
        self.assertTrue((archive / "data/metadata/split_contract.json").exists())
        self.assertTrue((archive / "data/metadata/hourly_mean_by_sensor.csv").exists())
        self.assertTrue((archive / "figures/previous_test_figure.png").exists())
        self.assertTrue((archive / "reports/data_quality_report.md").exists())
        self.assertFalse((metadata / "split_contract.json").exists())
        self.assertFalse((metadata / "hourly_mean_by_sensor.csv").exists())
        self.assertFalse(figure.exists())
        self.assertEqual(len(pd.read_csv(metadata / "missingness_summary.csv")), 0)
        self.assertNotIn("valid_mean", pd.read_csv(metadata / "sensors.csv"))

    def test_analysis_context_accepts_full_project_config_without_relaxing_checks(self):
        context, _ = self.analysis_fixture()
        context.update({"locations": None, "graph_sensor_ids": None, "adjacency": None})
        context["config"].update({"dataset": "METR-LA", "expected_sensors": 2,
                                  "imputation": "none_person_a", "speed_units": "unverified_source_units"})
        analysis = analyze_context(context)
        self.assertEqual(analysis["summary"]["dimensions"]["sensors"], 2)
        self.assertEqual(analysis["summary"]["validation_status"], "warning")

    def test_analysis_outputs_have_csv_evidence_serialized_summary_and_report(self):
        context, analysis = self.analysis_fixture()
        paths = write_analysis_outputs(context, analysis, self.root, plots=False)
        self.assertEqual(paths, {})
        metadata = self.root / "data/metadata"
        for key in ("sensors", "missingness_summary", "missingness_over_time", "timestamp_gaps",
                    "zero_runs", "flatline_runs", "discontinuities"):
            with self.subTest(table=key):
                self.assertTrue((metadata / f"{key}.csv").is_file())
        sensors = pd.read_csv(metadata / "sensors.csv")
        self.assertEqual(sensors["sensor_id"].tolist(), ["test-A", "test-B"])
        missingness = pd.read_csv(metadata / "missingness_summary.csv")
        self.assertEqual(missingness["zero_count"].sum(), 12)
        self.assertEqual(missingness["missing_count"].sum(), 1)
        summary = json.loads((metadata / "quality_summary.json").read_text())
        self.assertEqual(summary["speed_sha256"], "a" * 64)
        self.assertEqual(summary["value_counts"]["zero"], 12)
        json.dumps(summary, allow_nan=False)
        quantiles = pd.read_csv(metadata / "sensor_speed_quantiles.csv")
        self.assertGreater(quantiles["0.0"].min(), 0)
        contract = json.loads((metadata / "split_contract.json").read_text())
        self.assertEqual(contract["partitions"]["test"]["grid_timestamps"], 20)
        report = (self.root / "reports/data_quality_report.md").read_text()
        self.assertIn("Validation status: WARNING", report)
        self.assertIn("| sensor_id | position | start_timestamp", report)
        self.assertIn("test-A", report)
        self.assertIn("No sensor is automatically removed", report)
        self.assertFalse((self.root / "figures").exists())

    def test_structural_failure_writes_diagnostics_without_split_contract(self):
        context, analysis = self.analysis_fixture(structural_failure=True)
        self.assertEqual(analysis["summary"]["validation_status"], "fail")
        write_analysis_outputs(context, analysis, self.root, plots=False)
        metadata = self.root / "data/metadata"
        self.assertTrue((metadata / "timestamp_gaps.csv").exists())
        self.assertFalse((metadata / "split_contract.json").exists())
        report = (self.root / "reports/data_quality_report.md").read_text()
        self.assertIn("Validation status: FAIL", report)

    def test_failed_rerun_invalidates_prior_split_boundaries(self):
        context, analysis = self.analysis_fixture()
        write_analysis_outputs(context, analysis, self.root, plots=False)
        self.assertTrue((self.root / "data/metadata/split_contract.json").exists())
        frame = context["df"].drop(context["df"].index[30])
        context["df"] = frame
        failed = analyze_frame(frame, config={"expected_sensors": 2})
        write_analysis_outputs(context, failed, self.root, plots=False)
        self.assertFalse((self.root / "data/metadata/split_contract.json").exists())

    def test_acquisition_rejects_existing_html_and_changed_hdf5_provenance(self):
        # Reconstruct upstream paths using this project's copied source assets;
        # the shared project must not require the original sibling checkout.
        source = self.root / "upstream"
        relative_files = {
            "data/sensor_graph/adj_mx.pkl": "adj_mx.pkl",
            "data/sensor_graph/graph_sensor_ids.txt": "graph_sensor_ids.txt",
            "data/sensor_graph/graph_sensor_locations.csv": "graph_sensor_locations.csv",
            "README.md": "provenance/DCRNN_README.md",
            "LICENSE": "provenance/DCRNN_LICENSE",
            "lib/metrics.py": "provenance/DCRNN_metrics.py.txt",
            "scripts/eval_baseline_methods.py": "provenance/DCRNN_eval_baseline_methods.py.txt",
        }
        for relative, copied_asset in relative_files.items():
            copy_immutable(GRAPH_SOURCE / copied_asset, source / relative)
        raw = self.root / "data/raw"
        raw.mkdir(parents=True)
        speed = raw / "metr-la.h5"
        speed.write_bytes(b"<html>download confirmation</html>")
        arguments = ["acquire_data.py", "--dcrnn", str(source)]
        with mock.patch.object(acquire_data, "ROOT", self.root), \
             mock.patch("sys.argv", arguments), \
             mock.patch.object(acquire_data.subprocess, "check_output", return_value=acquire_data.COMMIT), \
             redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "not HDF5"):
                acquire_data.main()
            speed.write_bytes(acquire_data.HDF5_MAGIC + b"test-only-header-not-a-real-hdf5-table")
            manifest = {"files": {"data/raw/metr-la.h5": {"sha256": "wrong-digest"}}}
            manifest_path = raw / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "checksum changed"):
                acquire_data.main()
            self.assertEqual(json.loads(manifest_path.read_text()), manifest)

    def test_markdown_table_escapes_delimiters_and_caps_evidence(self):
        frame = pd.DataFrame({"case": ["A|B\nC"] + [str(i) for i in range(20)]})
        table = markdown_table(frame, limit=2)
        self.assertIn(r"A\|B C", table)
        self.assertEqual(len(table.splitlines()), 4)
        self.assertEqual(markdown_table(frame.iloc[:0]), "None detected.")


if __name__ == "__main__":
    unittest.main()
