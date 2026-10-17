"""Baseline forecasters for Phase 1 (Person C).

Two baselines every learned model has to beat:

  persistence        prediction = speed now (speed_lag_0, forward-filled input)
  historical average mean speed of (sensor, day of week, 5-min time slot) in the
                     training period, looked up at the TARGET time

The historical average follows the DCRNN baseline: zeros are "no reading" so they
are skipped, and it is fit on training timestamps only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.load_data import create_missingness_mask

HA_FEATURE_COLUMNS = ["ha_target", "ha_origin", "ha_delta", "anomaly_now"]
N_SLOTS = 288  # 5-minute slots per day
STEP_MINUTES = 5


def persistence_predict(df: pd.DataFrame) -> np.ndarray:
    return df["speed_lag_0"].to_numpy(dtype=np.float64)


def _slot_index(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Flat (day-of-week, time-slot) index in [0, 7 * 288)."""
    slot = (timestamps.hour * 60 + timestamps.minute) // STEP_MINUTES
    return (timestamps.dayofweek * N_SLOTS + slot).to_numpy()


class HistoricalAverage:
    """Mean valid speed per (sensor, day of week, time of day), fit on train only."""

    def fit(self, raw_df: pd.DataFrame, train_end_inclusive: pd.Timestamp | str) -> "HistoricalAverage":
        train = raw_df.loc[raw_df.index <= pd.Timestamp(train_end_inclusive)]
        valid = ~create_missingness_mask(train).to_numpy()
        values = np.where(valid, train.to_numpy(dtype=np.float64), 0.0)

        n_cells = 7 * N_SLOTS
        slot = _slot_index(train.index)
        sums = np.zeros((n_cells, train.shape[1]))
        counts = np.zeros((n_cells, train.shape[1]))
        np.add.at(sums, slot, values)
        np.add.at(counts, slot, valid.astype(np.float64))

        # a cell with no valid reading falls back to the sensor's own training mean
        self.sensor_mean_ = values.sum(axis=0) / np.maximum(valid.sum(axis=0), 1)
        self.sums_, self.counts_ = sums, counts
        self.values_, self.valid_ = values, valid  # kept so train rows can drop their own reading
        self.start_ = train.index[0]
        self.sensor_ids_ = list(train.columns)
        self.train_end_ = pd.Timestamp(train_end_inclusive)
        return self

    def lookup(self, timestamps: pd.DatetimeIndex, sensor_codes: np.ndarray, exclude_self: bool = False) -> np.ndarray:
        """Average for (sensor, time). With exclude_self the reading at that exact
        timestamp is removed from its own average (leave-one-out), which is what a
        training row needs so its feature does not contain its own target.
        Timestamps must be inside the training period when exclude_self is set."""
        cell = _slot_index(timestamps)
        codes = np.asarray(sensor_codes)
        s, c = self.sums_[cell, codes], self.counts_[cell, codes]
        if exclude_self:
            pos = ((timestamps - self.start_) // pd.Timedelta(minutes=STEP_MINUTES)).to_numpy()
            if pos.min() < 0 or pos.max() >= len(self.values_):
                raise ValueError("exclude_self needs timestamps inside the training period")
            s = s - self.values_[pos, codes]
            c = c - self.valid_[pos, codes]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(c > 0, s / c, self.sensor_mean_[codes])

    def predict(self, origin_timestamps: pd.Series, sensor_codes: np.ndarray, horizon_steps: int) -> np.ndarray:
        """Look up the average at origin + horizon (the time being predicted)."""
        target_time = pd.DatetimeIndex(origin_timestamps) + pd.Timedelta(minutes=STEP_MINUTES * horizon_steps)
        return self.lookup(target_time, sensor_codes)

    def features(
        self,
        origin_timestamps: pd.Series,
        sensor_codes: np.ndarray,
        speed_now: np.ndarray,
        horizon_steps: int,
        training_rows: bool,
    ) -> pd.DataFrame:
        """Extra model inputs built from the sensor's own training history.

        ha_target     usual speed at the time being predicted
        ha_origin     usual speed now
        ha_delta      usual change between now and the target time
        anomaly_now   how far current speed is from usual
        training_rows must be True for train rows (leave-one-out) and False otherwise.
        """
        origin = pd.DatetimeIndex(origin_timestamps)
        target = origin + pd.Timedelta(minutes=STEP_MINUTES * horizon_steps)
        ha_t = self.lookup(target, sensor_codes, exclude_self=training_rows)
        ha_o = self.lookup(origin, sensor_codes, exclude_self=training_rows)
        return pd.DataFrame({
            "ha_target": ha_t.astype(np.float32),
            "ha_origin": ha_o.astype(np.float32),
            "ha_delta": (ha_t - ha_o).astype(np.float32),
            "anomaly_now": (np.asarray(speed_now, dtype=np.float64) - ha_o).astype(np.float32),
        })
