"""Phase 2 data pipeline: METR-LA windows for deep models (Member 1).

Built on the frozen Phase 1 loaders so masking, filling, partitions and origins
are identical to Phase 1:

* a reading is *valid* if it is finite and > 0; zeros/NaN/inf/negatives are
  unavailable (``create_missingness_mask``), never "zero traffic";
* inputs use the causal fill (forward fill, then the sensor's **training**
  median for leading gaps); targets come from the raw array and are never filled;
* the scaler is one global mean/std over valid **training** cells only;
* origins come from ``get_eligible_sample_origins`` (23,967 / 3,416 / 6,844 on
  the real data), with all four targets inside the origin's own partition.

The whole ``(T, N, C=7)`` float32 feature tensor (about 200 MB on the real data)
lives on the accelerator. A batch is a single index-gather, so there are no
DataLoader workers and nothing is materialised per window.

Channels (``C = 7``): 0 z-scored filled speed, 1 missing flag, 2-3 hour
sin/cos, 4-5 day-of-week sin/cos, 6 is_weekend. Time features use Phase 1's
``compute_temporal_features`` on the stored clock (no timezone conversion).

Batch dict keys: ``x [B,12,N,7]``, ``y [B,4,N]`` (source speed units, 0 where
invalid), ``y_valid [B,4,N]`` (bool), ``origin_idx [B]`` (grid index of the
origin), ``pos [B]`` (position inside the dataset).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch

from src.data.features import compute_temporal_features
from src.data.load_data import (
    apply_causal_imputation,
    check_sensor_order,
    create_missingness_mask,
    fit_training_sensor_medians,
    load_raw_speed_data,
    load_sensor_metadata,
)
from src.data.split import get_eligible_sample_origins, get_partition_info, load_split_contract
from src.models.prediction_evaluation import temporal_masks
from src.phase2.common import (
    CHANNEL_NAMES,
    HISTORY_STEPS,
    HORIZON_NAMES,
    HORIZON_STEPS,
    NUM_CHANNELS,
    ROOT,
    resolve_device,
)

SPLIT_TO_PARTITION = {"train": "train", "val": "validation", "test": "test"}


@dataclass(frozen=True)
class StdScaler:
    """Global z-score fitted on valid training cells only."""

    mean: float
    std: float

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse(self, z):
        return z * self.std + self.mean


class Phase2Data:
    """Shared arrays for one run: features, targets, masks, scaler, partitions."""

    def __init__(
        self,
        raw_df: pd.DataFrame,
        contract: dict[str, Any],
        device: str | torch.device = "cpu",
        subset: float = 1.0,
    ) -> None:
        if not 0.0 < subset <= 1.0:
            raise ValueError(f"subset must be in (0, 1], got {subset}")
        history = int(contract["history_steps"])
        horizons = tuple(int(h) for h in contract["horizon_steps"])
        if history != HISTORY_STEPS or horizons != HORIZON_STEPS:
            raise ValueError(
                f"Phase 2 expects history={HISTORY_STEPS}, horizons={HORIZON_STEPS}; "
                f"contract has history={history}, horizons={horizons}"
            )
        self.device = resolve_device(device) if isinstance(device, str) else device
        self.subset = float(subset)
        self.contract = contract
        self.timestamps = pd.DatetimeIndex(raw_df.index)
        self.sensor_ids = [str(c) for c in raw_df.columns]
        self.num_nodes = len(self.sensor_ids)
        self.partitions = get_partition_info(self.timestamps, contract=contract)

        # --- masks and causal fill, exactly as Phase 1 (Person B) ---
        mask_df = create_missingness_mask(raw_df)
        train_end = self.partitions["train"].end_inclusive
        medians = fit_training_sensor_medians(raw_df, mask_df, train_end)
        filled_df = apply_causal_imputation(raw_df, mask_df, medians)

        raw64 = raw_df.to_numpy(dtype=np.float64)
        unavailable = mask_df.to_numpy(dtype=bool)
        train_end_idx = self.partitions["train"].end_idx
        train_values = raw64[: train_end_idx + 1][~unavailable[: train_end_idx + 1]]
        if train_values.size == 0:
            raise ValueError("no valid training readings to fit the scaler")
        self.scaler = StdScaler(mean=float(train_values.mean()), std=float(train_values.std()))
        if not self.scaler.std > 0:
            raise ValueError("training speed has zero variance")

        # kept for tests/inspection (numpy, small)
        self.raw = raw64.astype(np.float32)
        self.unavailable = unavailable
        self.filled = filled_df.to_numpy(dtype=np.float64).astype(np.float32)
        self.fill_medians = medians.to_numpy(dtype=np.float64)

        # --- feature tensor (T, N, C) ---
        n_steps = len(self.timestamps)
        time_df = compute_temporal_features(self.timestamps)
        time_block = time_df[["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]].to_numpy(
            dtype=np.float32
        )
        feats = np.empty((n_steps, self.num_nodes, NUM_CHANNELS), dtype=np.float32)
        feats[:, :, 0] = self.scaler.transform(filled_df.to_numpy(dtype=np.float64)).astype(np.float32)
        feats[:, :, 1] = unavailable.astype(np.float32)
        feats[:, :, 2:] = time_block[:, None, :]
        if not np.isfinite(feats).all():
            raise ValueError("non-finite values in the feature tensor")

        # --- targets: raw values, validity kept separately, never filled ---
        valid = np.isfinite(raw64) & (raw64 > 0.0)
        targets = np.where(valid, self.raw, 0.0).astype(np.float32)

        self.features = torch.from_numpy(feats).to(self.device)
        self.targets = torch.from_numpy(targets).to(self.device)
        self.target_valid = torch.from_numpy(valid).to(self.device)
        self.hist_offsets = torch.arange(-(HISTORY_STEPS - 1), 1, device=self.device)
        self.horizon_offsets = torch.tensor(HORIZON_STEPS, device=self.device)

        # --- origins per split ---
        self.origins: dict[str, np.ndarray] = {}
        for split, part in SPLIT_TO_PARTITION.items():
            origins, _ = get_eligible_sample_origins(
                self.timestamps, part, history_steps=history, horizon_steps=list(horizons), contract=contract
            )
            origins = np.asarray(origins, dtype=np.int64)
            if self.subset < 1.0:
                keep = max(1, int(round(self.subset * len(origins))))
                origins = origins[np.unique(np.linspace(0, len(origins) - 1, keep).astype(np.int64))]
            self.origins[split] = origins

    # ------------------------------------------------------------------ factories
    @classmethod
    def from_files(
        cls,
        raw_path: str | Path = ROOT / "data/raw/metr-la.h5",
        contract_path: str | Path = ROOT / "data/metadata/split_contract.json",
        sensors_path: str | Path = ROOT / "data/metadata/sensors.csv",
        device: str | torch.device = "cpu",
        subset: float = 1.0,
    ) -> "Phase2Data":
        raw_df = load_raw_speed_data(raw_path)
        check_sensor_order(raw_df, load_sensor_metadata(sensors_path))
        return cls(raw_df, load_split_contract(contract_path), device=device, subset=subset)

    # ------------------------------------------------------------------ views
    def dataset(self, split: str) -> "WindowDataset":
        if split not in self.origins:
            raise KeyError(f"unknown split {split!r}; use one of {list(self.origins)}")
        return WindowDataset(self, split)

    def datasets(self) -> dict[str, "WindowDataset"]:
        return {split: self.dataset(split) for split in self.origins}


class WindowDataset:
    """Eligible forecast origins of one split, gathered on demand."""

    def __init__(self, data: Phase2Data, split: str) -> None:
        self.data = data
        self.split = split
        self.origins = data.origins[split]
        self.origins_t = torch.from_numpy(self.origins).to(data.device)
        self.timestamps = data.timestamps[self.origins]
        self.scaler = data.scaler
        self.sensor_ids = data.sensor_ids
        self.num_nodes = data.num_nodes
        self.device = data.device

    def __len__(self) -> int:
        return len(self.origins)

    @property
    def origin_ns(self) -> np.ndarray:
        """Origin timestamps as int64 nanoseconds (stored clock, no timezone)."""
        return self.timestamps.to_numpy().astype("datetime64[ns]").astype(np.int64)

    def get_batch(self, pos) -> dict[str, torch.Tensor]:
        pos_t = torch.as_tensor(pos, dtype=torch.long, device=self.device).reshape(-1)
        origin = self.origins_t[pos_t]
        hist = origin[:, None] + self.data.hist_offsets  # [B, 12]
        tgt = origin[:, None] + self.data.horizon_offsets  # [B, 4]
        return {
            "x": self.data.features[hist],  # [B, 12, N, C]
            "y": self.data.targets[tgt],  # [B, 4, N]
            "y_valid": self.data.target_valid[tgt],  # [B, 4, N]
            "origin_idx": origin,
            "pos": pos_t,
        }

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError(i)
        return {k: v[0] for k, v in self.get_batch([i]).items()}

    def batches(
        self,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
        epoch: int = 0,
        drop_last: bool = False,
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield batches in order, or in a seed/epoch-determined shuffled order.

        The shuffle depends only on ``(seed, epoch)``, so a resumed run sees the same
        order as an uninterrupted one.
        """
        n = len(self)
        if shuffle:
            gen = torch.Generator().manual_seed(int(seed) * 1_000_003 + int(epoch))
            order = torch.randperm(n, generator=gen)
        else:
            order = torch.arange(n)
        for start in range(0, n, batch_size):
            chunk = order[start : start + batch_size]
            if drop_last and len(chunk) < batch_size:
                break
            yield self.get_batch(chunk)

    def period_masks(self) -> dict[str, np.ndarray]:
        """Phase 1 slices by forecast-origin time: all, peak, off_peak, morning_peak, evening_peak."""
        return temporal_masks(self.timestamps)

    def targets_numpy(self, chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(y, valid)`` as ``[n_origins, 4, N]`` numpy arrays (source speed units)."""
        ys, vs = [], []
        for start in range(0, len(self), chunk):
            batch = self.get_batch(np.arange(start, min(start + chunk, len(self))))
            ys.append(batch["y"].cpu().numpy())
            vs.append(batch["y_valid"].cpu().numpy())
        return np.concatenate(ys), np.concatenate(vs)

    def last_input_speed(self) -> np.ndarray:
        """Filled speed at each origin, ``[n_origins, N]`` (source units); Phase 1 ``speed_lag_0``."""
        return self.data.filled[self.origins]


_CACHE: dict[tuple, Phase2Data] = {}


def load_data(cfg: dict[str, Any] | None = None) -> Phase2Data:
    """Build (or reuse within the process) the shared data object.

    ``cfg`` keys (all optional): ``device``, ``subset``, ``raw_path``,
    ``contract_path``, ``sensors_path``.
    """
    cfg = dict(cfg or {})
    device = resolve_device(cfg.get("device", "cpu"))
    subset = float(cfg.get("subset", 1.0))
    paths = tuple(
        str(cfg.get(k, ROOT / rel))
        for k, rel in [
            ("raw_path", "data/raw/metr-la.h5"),
            ("contract_path", "data/metadata/split_contract.json"),
            ("sensors_path", "data/metadata/sensors.csv"),
        ]
    )
    key = (paths, str(device), subset)
    if key not in _CACHE:
        _CACHE[key] = Phase2Data.from_files(*paths, device=device, subset=subset)
    return _CACHE[key]


def build_datasets(cfg: dict[str, Any] | None = None) -> dict[str, WindowDataset]:
    """Plan stub API: ``{"train": ..., "val": ..., "test": ...}``."""
    return load_data(cfg).datasets()


__all__ = [
    "CHANNEL_NAMES",
    "HORIZON_NAMES",
    "HORIZON_STEPS",
    "Phase2Data",
    "StdScaler",
    "WindowDataset",
    "build_datasets",
    "load_data",
]
