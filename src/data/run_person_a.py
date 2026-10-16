"""Reproducible Person A inventory, quality checks, metadata, report and EDA."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(root: Path = ROOT) -> dict:
    path = root / "data/raw/manifest.json"
    if not path.exists():
        raise FileNotFoundError("Run scripts/acquire_data.py first to record raw provenance.")
    manifest = json.loads(path.read_text())
    for relative, expected in manifest["files"].items():
        file = root / relative
        if not file.is_file() or checksum(file) != expected["sha256"]:
            raise ValueError(f"Raw file absent or checksum changed: {relative}")
    return manifest


def load_graph_inventory(root: Path = ROOT) -> tuple[pd.DataFrame, list[str], np.ndarray, dict]:
    """Only read trusted upstream pickle copied by acquire_data.py."""
    raw = root / "data/raw"
    with (raw / "adj_mx.pkl").open("rb") as stream:
        ids, id_to_index, adjacency = pickle.load(stream, encoding="latin1")
    ids = [str(value) for value in ids]
    mapping = {str(key): int(value) for key, value in id_to_index.items()}
    text_ids = (raw / "graph_sensor_ids.txt").read_text().strip().replace("\n", ",").split(",")
    text_ids = [value.strip() for value in text_ids if value.strip()]
    locations = pd.read_csv(raw / "graph_sensor_locations.csv", dtype={"sensor_id": str})
    adjacency = np.asarray(adjacency)
    checks = {
        "sensor_count": len(ids),
        "unique_sensor_ids": len(set(ids)) == len(ids),
        "adjacency_shape": list(adjacency.shape),
        "adjacency_finite": bool(np.isfinite(adjacency).all()),
        "adjacency_nonnegative": bool((adjacency >= 0).all()),
        "id_mapping_matches_order": mapping == {sensor: i for i, sensor in enumerate(ids)},
        "sensor_text_order_matches": text_ids == ids,
        "locations_have_unique_ids": not bool(locations["sensor_id"].duplicated().any()),
        "locations_cover_graph": set(locations["sensor_id"]) == set(ids),
        "locations_finite": bool(np.isfinite(locations[["latitude", "longitude"]]).all().all()),
        "speed_alignment_status": "pending_speed_file",
    }
    required = ["unique_sensor_ids", "adjacency_finite", "adjacency_nonnegative", "id_mapping_matches_order", "sensor_text_order_matches", "locations_have_unique_ids", "locations_cover_graph", "locations_finite"]
    if adjacency.shape != (len(ids), len(ids)) or not all(checks[key] for key in required):
        raise ValueError(f"Upstream graph metadata is inconsistent: {checks}")
    return locations, ids, adjacency, checks


def load_context(root: Path = ROOT) -> dict:
    manifest = verify_manifest(root)
    locations, ids, adjacency, graph = load_graph_inventory(root)
    path = root / "data/raw/metr-la.h5"
    if not path.exists():
        raise FileNotFoundError(f"METR-LA speed data absent: {path}. See reports/data_sources.md.")
    if "data/raw/metr-la.h5" not in manifest["files"]:
        raise ValueError("Speed data has no checksum/provenance; run scripts/acquire_data.py first.")
    try:
        with pd.HDFStore(path, mode="r") as store:
            keys = store.keys()
            if len(keys) != 1:
                raise ValueError(f"Expected one canonical speed table; HDF5 keys: {keys}")
            df = store[keys[0]]
    except ImportError as error:
        raise ImportError("PyTables is needed for METR-LA HDF5; install requirements.txt.") from error
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas speed DataFrame in the canonical HDF5 file.")
    config = yaml.safe_load((root / "configs/data_quality.yaml").read_text())
    return {"df": df, "locations": locations, "graph_sensor_ids": ids,
            "adjacency": adjacency, "config": config, "manifest": manifest,
            "graph_inventory": graph}


def json_default(value):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta, Path)):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)}")


def analyze_context(context: dict) -> dict:
    from .quality import DEFAULT_CONFIG, analyze_frame
    return analyze_frame(context["df"], locations=context["locations"],
                         graph_sensor_ids=context["graph_sensor_ids"],
                         adjacency=context["adjacency"],
                         config={key: value for key, value in context["config"].items() if key in DEFAULT_CONFIG})


def markdown_table(frame: pd.DataFrame, limit: int = 12) -> str:
    frame = frame.head(limit)
    if frame.empty:
        return "None detected."
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(cell, frame.columns)) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines.extend("| " + " | ".join(map(cell, row)) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def write_blocked_outputs(reason: str, root: Path = ROOT) -> None:
    metadata = root / "data/metadata"
    reports = root / "reports"
    metadata.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    previous = metadata / "quality_summary.json"
    archived = None
    if previous.exists():
        prior_status = json.loads(previous.read_text()).get("validation_status")
        if prior_status != "blocked":
            archived = reports / "archived_outputs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            for folder, patterns in [(metadata, ["*.csv", "*.json", "README.md"]),
                                     (root / "figures", ["*.png", "representative_sensor_selection.csv"]),
                                     (reports, ["data_quality_report.md"])]:
                for pattern in patterns:
                    for path in folder.glob(pattern):
                        destination = archived / path.relative_to(root)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(path), destination)
    (metadata / "split_contract.json").unlink(missing_ok=True)
    graph, manifest = {}, {}
    try:
        manifest = verify_manifest(root)
        locations, ids, _, graph = load_graph_inventory(root)
        sensors = pd.DataFrame({"sensor_id": ids, "position": range(len(ids))}).merge(
            locations[["sensor_id", "latitude", "longitude"]], on="sensor_id", how="left", validate="one_to_one")
        sensors["speed_alignment_status"] = "pending_speed_validation"
        sensors.to_csv(metadata / "sensors.csv", index=False)
        missing_columns = ["sensor_id", "position", "observation_count", "valid_count", "missing_count",
                           "infinite_count", "nonfinite_count", "zero_count", "negative_count", "invalid_count"]
        missing_columns += [f"{kind}_fraction" for kind in ["missing", "infinite", "nonfinite", "zero", "negative", "invalid"]]
        pd.DataFrame(columns=missing_columns).to_csv(metadata / "missingness_summary.csv", index=False)
    except Exception as error:
        graph = {"inventory_error": str(error)}
    summary = {"validation_status": "blocked", "reason": reason,
               "generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "graph_inventory": graph, "speed_statistics_computed": False,
               "previous_outputs_archived": str(archived.relative_to(root)) if archived else None}
    (metadata / "quality_summary.json").write_text(json.dumps(summary, indent=2, default=json_default) + "\n")
    (metadata / "README.md").write_text("# Metadata status\n\nSpeed analysis is blocked. `sensors.csv` contains the verified graph/location inventory only; speed-column alignment is pending. `missingness_summary.csv` is schema-only with zero result rows. Blank counts do not mean zero missingness. No speed statistics, split boundaries or EDA figures have been computed.\n")
    manifest_rows = pd.DataFrame([{"file": key, "bytes": value["bytes"], "sha256": value["sha256"]} for key, value in manifest.get("files", {}).items()])
    report = f"""# METR-LA data quality report — Person A, Phase 1

**Status: BLOCKED — speed dataset not validated.**

{reason}

## Completed inventory

The separate project folder contains copies of the upstream graph, sensor IDs, coordinates and source evidence. Original DCRNN files are unchanged. Checksums are in `data/raw/manifest.json`.

```json
{json.dumps(graph, indent=2, default=json_default)}
```

These graph sensor counts are not evidence that the speed table has matching columns. Speed-file dimensions, dates, sampling gaps, missingness, zeros, sensor variance and outages remain **not measured**.

{markdown_table(manifest_rows)}

## Data-quality rule

Preserve raw measurements byte-for-byte. Following the [upstream benchmark](https://github.com/liyaguang/DCRNN/blob/602afd9d767d3aa1c9b3eac51710d6aeee12c227/scripts/eval_baseline_methods.py#L89-L106), flag exact zero as missing/suspect, mask NaN/nonfinite/negative values, and retain positive near-zero speeds. A zero flag does not establish its physical cause. No imputation is performed by Person A. No flow or occupancy variables are fabricated.

Run-length, low-variance and discontinuity thresholds are predeclared diagnostics in `configs/data_quality.yaml`; they are not proof of outages and do not authorize sensor removal. Units and timezone are unverified; preserve stored values and clocks.

## Pending acceptance criteria

- Acquire `data/raw/metr-la.h5` and record its checksum.
- Validate actual sensor columns, dimensions, chronological index, elapsed cadence, start/end, duplicate and missing timestamps.
- Compute missing/zero/invalid counts, constant and low-variance sensors, zero runs, flatlines and large jumps.
- Populate measured missingness metadata, all EDA tables and ten figures.
- Execute both notebooks against the actual speed dataset and inspect outputs.

## Resume

See `README.md` for acquisition and run commands, `reports/data_sources.md` for the official source, and `reports/person_b_handoff.md` for the leakage-safe contract. The notebooks are prepared but unexecuted; they contain no invented results. Person A is not complete, and the team's full Phase 1 is not complete.
"""
    (reports / "data_quality_report.md").write_text(report)


def split_contract(index: pd.DatetimeIndex, config: dict) -> dict:
    grid = pd.date_range(index.min(), index.max(), freq=config["frequency"])
    n = len(grid)
    train_end = int(np.floor(n * config["split"]["train_fraction"]))
    val_end = train_end + int(np.floor(n * config["split"]["validation_fraction"]))
    partitions = {}
    for name, start, end in [("train", 0, train_end), ("validation", train_end, val_end), ("test", val_end, n)]:
        partitions[name] = {"start_inclusive": str(grid[start]) if start < end else None,
                            "end_inclusive": str(grid[end - 1]) if start < end else None,
                            "grid_timestamps": end - start}
    return {"basis": "complete_5min_timestamp_grid", "partitions": partitions,
            "history_steps": config["history_steps"], "horizon_steps": config["horizon_steps"],
            "rule": "All targets for a sample must belong to one partition. History may use earlier observed timestamps only. Do not shuffle. Missing grid rows require explicit masks; row offsets on gapped data are not elapsed horizons."}


def write_analysis_outputs(context: dict, analysis: dict, root: Path = ROOT, plots: bool = True) -> dict:
    metadata = root / "data/metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(parents=True, exist_ok=True)
    names = ["sensors", "missingness_summary", "missingness_over_time", "timestamp_gaps", "zero_runs", "flatline_runs", "discontinuities"]
    for name in names:
        analysis[name].to_csv(metadata / f"{name}.csv", index=name == "missingness_over_time")
    summary = analysis["summary"]
    summary["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["speed_sha256"] = context["manifest"]["files"]["data/raw/metr-la.h5"]["sha256"]
    (metadata / "quality_summary.json").write_text(json.dumps(summary, indent=2, default=json_default) + "\n")
    valid = analysis["valid_speed"]
    from .descriptive import summarize_descriptive, render_descriptive
    descriptive = summarize_descriptive(context["df"], analysis)
    (metadata / "eda_summary.json").write_text(json.dumps(descriptive["summary"], indent=2, default=json_default) + "\n")
    for name, table in descriptive["tables"].items():
        table.to_csv(metadata / f"{name}.csv", index=False)
    pd.DataFrame({"mean_speed": valid.mean(axis=1), "valid_sensors": valid.notna().sum(axis=1)}).to_csv(metadata / "network_speed_over_time.csv", index_label="timestamp")
    valid.groupby(valid.index.hour).mean().to_csv(metadata / "hourly_mean_by_sensor.csv", index_label="stored_clock_hour")
    valid.groupby(valid.index.dayofweek).mean().to_csv(metadata / "weekday_mean_by_sensor.csv", index_label="day_of_week_monday_0")
    valid.quantile([0, .05, .25, .5, .75, .95, 1]).T.to_csv(metadata / "sensor_speed_quantiles.csv", index_label="sensor_id")
    figure_paths = {}
    if summary.get("validation_status") != "fail":
        contract = split_contract(context["df"].index, context["config"])
        (metadata / "split_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    else:
        (metadata / "split_contract.json").unlink(missing_ok=True)
    plot_index_valid = context["df"].index.is_monotonic_increasing and context["df"].index.is_unique
    if plots and plot_index_valid:
        os.environ.setdefault("MPLCONFIGDIR", str(root / ".mplconfig"))
        os.environ.setdefault("MPLBACKEND", "Agg")
        from .eda import generate_eda
        figure_paths = generate_eda(context["df"], analysis, root / "figures")
    (metadata / "README.md").write_text("# Metadata\n\nComputed from immutable raw speed data using `python -m src.data.run_person_a`. `sensors.csv` follows original speed-column order. Zero/invalid flags affect descriptive valid-speed summaries only; raw data is untouched. Run/threshold flags are descriptive, not proven outages. See the quality report and handoff for definitions.\n")
    sections = ["# METR-LA data quality report — Person A, Phase 1", f"**Validation status: {summary['validation_status'].upper()}**",
                render_descriptive(descriptive["summary"], descriptive["tables"]),
                "## Measured dataset validation", "All numeric findings below are computed from the checksum-recorded HDF5 file. Source values and timestamp timezone are preserved; speed units are unverified."]
    for key, value in summary.items():
        sections += [f"### {key.replace('_', ' ').capitalize()}", "```json\n" + json.dumps(value, indent=2, default=json_default) + "\n```"]
    sections += ["## Zero and missing-value handling", "The [upstream baseline](https://github.com/liyaguang/DCRNN/blob/602afd9d767d3aa1c9b3eac51710d6aeee12c227/scripts/eval_baseline_methods.py#L89-L106) masks zero targets and excludes zero observations from historical averages. We preserve raw bytes, treat exact zero as benchmark missing/suspect, mask NaN/nonfinite/negative values and retain positive near-zero observations. The data alone cannot establish whether any particular zero is stopped traffic, an outage or corruption.",
                 "Contiguous zero runs, simultaneous sensor failures, flatlines and large jumps provide diagnostic evidence. Run thresholds are predeclared in config. No sensor is automatically removed, and no full-period statistic is used to fit imputation or model selection. Imputation belongs to Person B and must use only past observations and parameters fitted on training timestamps. Invalid targets must remain masked rather than imputed.",
                 "## Diagnostic evidence tables (first 12 rows; full CSVs in metadata)"]
    for key in ["timestamp_gaps", "zero_runs", "flatline_runs", "discontinuities"]:
        sections += [f"### {key.replace('_', ' ').capitalize()}", markdown_table(analysis[key])]
    sections += ["## EDA figures", "Descriptive full-period EDA is not an estimator fitted for forecasting. Missing/suspect speed values are excluded; observed-speed patterns can be biased by nonrandom missingness. Hours use the stored clock; peak definitions are weekdays 07:00–10:00 and 16:00–19:00, half-open intervals."]
    sections.extend(f"- [{key.replace('_', ' ')}](../figures/{path})" for key, path in figure_paths.items())
    if not figure_paths:
        sections.append("Figures were not generated in this run (disabled or blocked by a nonunique/unordered timestamp index).")
    sections += ["## Handoff and limitations", "See [Person B handoff](person_b_handoff.md). Structural failures block the canonical handoff. Warning-level diagnostics require awareness, not automatic sensor deletion. `data/metadata/split_contract.json` defines timestamp boundaries only; Person B owns feature samples. No models, performance findings, graph features, SUMO or control were implemented."]
    (root / "reports/data_quality_report.md").write_text("\n\n".join(sections) + "\n")
    return figure_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plots", action="store_true", help="Validate and write tables without figures")
    args = parser.parse_args()
    try:
        context = load_context()
    except (FileNotFoundError, ImportError, ValueError, TypeError, OSError) as error:
        write_blocked_outputs(str(error))
        print(f"BLOCKED: {error}", file=sys.stderr)
        return 2
    try:
        analysis = analyze_context(context)
    except (ValueError, TypeError) as error:
        write_blocked_outputs(str(error))
        print(f"BLOCKED: {error}", file=sys.stderr)
        return 2
    paths = write_analysis_outputs(context, analysis, plots=not args.no_plots)
    print(f"Validation: {analysis['summary']['validation_status']}; {len(paths)} EDA figures; report: reports/data_quality_report.md")
    return 1 if analysis["summary"]["validation_status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
