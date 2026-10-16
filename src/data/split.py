"""Train/val/test split handling (Person B).

The split itself is Person A's, stored in data/metadata/split_contract.json
(70/10/20 of the full 5-minute grid). This module only turns those
timestamp boundaries into the list of usable prediction origins.

Rules for an origin t (index into the grid):
  - needs 12 steps of history, t-11 ... t  (can come from an earlier split,
    that data was already observed at time t)
  - all targets t+1, t+3, t+6, t+12 must be inside the same split
  - anything that would cross the split boundary is dropped
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PartitionInfo:
    name: str
    start_inclusive: pd.Timestamp
    end_inclusive: pd.Timestamp
    grid_timestamps: int
    start_idx: int
    end_idx: int  # inclusive


def load_split_contract(path: Path | str = ROOT / "data/metadata/split_contract.json") -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"split contract not found: {path}")
    return json.loads(path.read_text())


def get_partition_info(
    timestamps: pd.DatetimeIndex,
    contract: dict[str, Any] | None = None,
) -> dict[str, PartitionInfo]:
    """Find the grid index of each split's start/end and check the counts match."""
    if contract is None:
        contract = load_split_contract()

    parts = {}
    for name, p in contract["partitions"].items():
        start = pd.Timestamp(p["start_inclusive"])
        end = pd.Timestamp(p["end_inclusive"])
        if start not in timestamps or end not in timestamps:
            raise KeyError(f"{name} boundaries {start} / {end} not on the grid")

        start_idx = timestamps.get_loc(start)
        end_idx = timestamps.get_loc(end)
        n = end_idx - start_idx + 1
        if n != int(p["grid_timestamps"]):
            raise ValueError(f"{name}: contract says {p['grid_timestamps']} timestamps, grid has {n}")

        parts[name] = PartitionInfo(name, start, end, n, start_idx, end_idx)
    return parts


def get_eligible_sample_origins(
    timestamps: pd.DatetimeIndex,
    partition_name: str,
    history_steps: int = 12,
    horizon_steps: list[int] | None = None,
    contract: dict[str, Any] | None = None,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Return (grid indices, timestamps) of every usable origin for one split.

    Note the first val/test origin is the last timestamp of the previous
    split: its first target (t+1) is already inside the new split.
    """
    horizon_steps = horizon_steps or [1, 3, 6, 12]
    parts = get_partition_info(timestamps, contract=contract)
    if partition_name not in parts:
        raise ValueError(f"unknown partition {partition_name!r}, have {list(parts)}")
    part = parts[partition_name]

    first = max(part.start_idx - min(horizon_steps), history_steps - 1)
    last = part.end_idx - max(horizon_steps)
    if first > last:
        return np.array([], dtype=int), pd.DatetimeIndex([])

    idx = np.arange(first, last + 1)
    return idx, timestamps[idx]


def summarize_splits(
    timestamps: pd.DatetimeIndex,
    history_steps: int = 12,
    horizon_steps: list[int] | None = None,
    contract: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Small table of timestamps / origins per split, handy in notebooks."""
    rows = []
    for name, part in get_partition_info(timestamps, contract=contract).items():
        idx, times = get_eligible_sample_origins(
            timestamps, name, history_steps, horizon_steps, contract
        )
        rows.append({
            "partition": name,
            "start_timestamp": str(part.start_inclusive),
            "end_timestamp": str(part.end_inclusive),
            "grid_timestamps": part.grid_timestamps,
            "eligible_origins": len(idx),
            "first_origin": str(times[0]) if len(times) else None,
            "last_origin": str(times[-1]) if len(times) else None,
        })
    return pd.DataFrame(rows)
