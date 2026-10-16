"""Copy canonical source files without changing existing raw bytes.

The official dataset link is in reports/data_sources.md. --url accepts a direct
public HDF5 download URL; a Google Drive folder/HTML page is not a data file.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "602afd9d767d3aa1c9b3eac51710d6aeee12c227"
UPSTREAM = f"https://github.com/liyaguang/DCRNN/blob/{COMMIT}"
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_immutable(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(source) != sha256(target):
            raise FileExistsError(f"Refusing to replace different canonical raw file: {target}")
        return
    shutil.copy2(source, target)


def is_hdf5(path: Path) -> bool:
    with path.open("rb") as stream:
        return stream.read(8) == HDF5_MAGIC


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dcrnn", type=Path, default=ROOT.parent / "DCRNN")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--metr-la", type=Path, help="Existing original metr-la.h5")
    group.add_argument("--url", help="Direct public HDF5 download URL")
    args = parser.parse_args()
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = raw / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": {}}
    source_root = args.dcrnn.resolve()
    try:
        commit = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest["dcrnn_commit"] = commit
    manifest["official_dataset_folder"] = "https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX"

    def record(source: Path, dest: Path, source_url: str | None) -> None:
        key = str(dest.relative_to(ROOT))
        prior = manifest["files"].get(key, {})
        manifest["files"][key] = {
            "source_path": str(source.resolve()), "source_url": source_url,
            "sha256": sha256(dest), "bytes": dest.stat().st_size,
            "recorded_at_utc": prior.get("recorded_at_utc", datetime.now(timezone.utc).isoformat()),
        }

    copies = {
        "data/sensor_graph/adj_mx.pkl": "adj_mx.pkl",
        "data/sensor_graph/graph_sensor_locations.csv": "graph_sensor_locations.csv",
        "data/sensor_graph/graph_sensor_ids.txt": "graph_sensor_ids.txt",
        "README.md": "provenance/DCRNN_README.md",
        "LICENSE": "provenance/DCRNN_LICENSE",
        "lib/metrics.py": "provenance/DCRNN_metrics.py.txt",
        "scripts/eval_baseline_methods.py": "provenance/DCRNN_eval_baseline_methods.py.txt",
    }
    for relative, target in copies.items():
        source = source_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"Required DCRNN source absent: {source}")
        dest = raw / target
        copy_immutable(source, dest)
        record(source, dest, f"https://github.com/liyaguang/DCRNN/blob/{commit or COMMIT}/{relative}")

    speed_source = args.metr_la
    if speed_source is None and (source_root / "data/metr-la.h5").is_file():
        speed_source = source_root / "data/metr-la.h5"
    if args.url and not (raw / "metr-la.h5").exists():
        temporary = raw / "metr-la.h5.download"
        try:
            request = urllib.request.Request(args.url, headers={"User-Agent": "METR-LA-Person-A/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
                shutil.copyfileobj(response, out)
            if not is_hdf5(temporary):
                raise ValueError("Download is not HDF5 (possibly an HTML/Drive confirmation page).")
            temporary.rename(raw / "metr-la.h5")
            record(raw / "metr-la.h5", raw / "metr-la.h5", args.url)
        except Exception as error:
            manifest["download_error"] = str(error)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            temporary.unlink(missing_ok=True)
            raise
    if speed_source is not None:
        if not is_hdf5(speed_source):
            raise ValueError(f"Not an original HDF5 file: {speed_source}")
        copy_immutable(speed_source, raw / "metr-la.h5")
        record(speed_source, raw / "metr-la.h5", None)
        manifest["files"]["data/raw/metr-la.h5"]["reference_distribution_url"] = manifest["official_dataset_folder"]
    elif (raw / "metr-la.h5").exists():
        if not is_hdf5(raw / "metr-la.h5"):
            raise ValueError("Existing raw/metr-la.h5 is not HDF5; acquisition is not complete.")
        prior_speed = manifest["files"].get("data/raw/metr-la.h5")
        if prior_speed and sha256(raw / "metr-la.h5") != prior_speed["sha256"]:
            raise ValueError("Canonical speed-file checksum changed; refusing to replace its provenance.")
        if not prior_speed:
            record(raw / "metr-la.h5", raw / "metr-la.h5", None)
    manifest["speed_data_present"] = (raw / "metr-la.h5").is_file()
    if manifest["speed_data_present"]:
        manifest.pop("download_error", None)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for key, value in manifest["files"].items():
        print(f"{key}: {value['bytes']:,} bytes, SHA-256 {value['sha256']}")
    if not manifest["speed_data_present"]:
        print("BLOCKED: metr-la.h5 is absent. Graph files copied; no speed data was synthesized.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
