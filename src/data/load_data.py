"""Loading the METR-LA speed table and handling missing readings (Person B).

Zeros in METR-LA mean "no reading", not a stopped car, so they are treated
as missing together with NaN / inf / negative values (same rule as Person A
and the upstream DCRNN baselines).

Inputs are filled causally: forward fill only, and anything still empty at
the very start of a sensor's series gets that sensor's training median.
Targets are never filled, see features.extract_targets.
"""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def load_raw_speed_data(
    path: Path | str = ROOT / "data/raw/metr-la.h5",
    freq: str = "5min",
) -> pd.DataFrame:
    """Read the speed table and put it on a full 5-minute grid.

    Duplicate timestamps raise an error (Person A contract says don't average
    or drop them silently). If any 5-minute slots are absent they are added
    back as all-NaN rows, which the missing mask then picks up.
    The supplied file has no gaps, so normally this is a no-op.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"speed file not found: {path}")

    with pd.HDFStore(path, mode="r") as store:
        keys = store.keys()
        if len(keys) != 1:
            raise ValueError(f"expected one table in {path.name}, found {keys}")
        df = store[keys[0]]

    df.index = pd.DatetimeIndex(df.index)
    df.columns = [str(c).strip() for c in df.columns]

    if df.index.has_duplicates:
        raise ValueError("duplicate timestamps in speed data")
    if df.columns.has_duplicates:
        raise ValueError("duplicate sensor ids in speed data")
    df = df.sort_index()

    grid = pd.date_range(df.index[0], df.index[-1], freq=freq)
    n_absent = len(grid) - len(df)
    if n_absent:
        df = df.reindex(grid)
    df.attrs["absent_timestamps"] = int(n_absent)
    return df


def load_sensor_metadata(path: Path | str = ROOT / "data/metadata/sensors.csv") -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"sensor metadata not found: {path}")
    return pd.read_csv(path, dtype={"sensor_id": str})


def load_graph_metadata(
    path: Path | str = ROOT / "data/raw/adj_mx.pkl",
) -> tuple[list[str], dict[str, int], np.ndarray]:
    """Graph pickle. Not used as a model input in Phase 1, only kept for Phase 2."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"adjacency file not found: {path}")
    with path.open("rb") as f:
        ids, id_to_index, adj = pickle.load(f, encoding="latin1")
    return [str(i) for i in ids], {str(k): int(v) for k, v in id_to_index.items()}, np.asarray(adj)


def check_sensor_order(df: pd.DataFrame, sensors: pd.DataFrame) -> None:
    """Make sure speed columns are in the same order as Person A's sensors.csv."""
    expected = sensors["sensor_id"].astype(str).tolist()
    if list(df.columns) != expected:
        raise ValueError("speed columns do not match sensors.csv order")


def create_missingness_mask(
    df: pd.DataFrame,
    zero_as_missing: bool = True,
    negative_as_missing: bool = True,
) -> pd.DataFrame:
    """True where a reading can't be used (NaN/inf, and by default 0 or < 0)."""
    values = df.to_numpy()
    mask = ~np.isfinite(values)
    with np.errstate(invalid="ignore"):
        if zero_as_missing:
            mask |= values == 0.0
        if negative_as_missing:
            mask |= values < 0.0
    return pd.DataFrame(mask, index=df.index, columns=df.columns)


def fit_training_sensor_medians(
    df: pd.DataFrame,
    mask: pd.DataFrame,
    train_end_inclusive: pd.Timestamp | str,
) -> pd.Series:
    """Median valid speed per sensor, using training timestamps only.

    A sensor with no valid training reading at all falls back to the
    overall training median (doesn't happen for METR-LA, but just in case).
    """
    train_end = pd.Timestamp(train_end_inclusive)
    in_train = df.index <= train_end
    valid_train = df.loc[in_train].where(~mask.loc[in_train])

    medians = valid_train.median()
    return medians.fillna(float(np.nanmedian(valid_train.to_numpy())))


def apply_causal_imputation(
    df: pd.DataFrame,
    mask: pd.DataFrame,
    fallback_medians: pd.Series,
) -> pd.DataFrame:
    """Fill missing inputs with the last valid reading seen so far.

    Only ffill, never bfill or interpolation, so a value at time t only
    depends on readings at or before t. Leading gaps (no earlier reading)
    get the training median. Use this for features only, not targets.
    """
    filled = df.where(~mask).ffill()
    return filled.fillna(fallback_medians)
