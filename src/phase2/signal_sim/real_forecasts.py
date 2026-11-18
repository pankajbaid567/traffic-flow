"""Simulated signal control driven by REAL test-period speeds and REAL model forecasts.

Still a software simulation (one signalised approach + one cross street, see
``simulator.py``), but the inputs are no longer synthetic:

* the main approach's speed follows a real METR-LA sensor on a real test weekday
  (06:00-21:00 stored clock, causal forward fill for unavailable readings);
* arrivals are generated from that speed with the simulator's Greenshields-shaped rule
  (this mapping is an assumption, not measured demand);
* every predictive policy sees the same three signals - current speed and the 15- and
  30-minute forecasts issued at the latest 5-minute origin - and only the forecast
  source differs: Phase 1 XGBoost+HA, Phase 2 models (seed-mean test predictions), or a
  perfect forecast (the observed future speed) as a reference upper bound.

Sensors are chosen from observed data only (most proxy congestion on test weekdays),
never from model errors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.phase2.common import ROOT
from src.phase2.signal_sim.simulator import (
    FixedSignalPolicy,
    IntersectionQueueSimulator,
    ReactiveSignalPolicy,
    SignalControlPolicy,
    SignalSimulationConfig,
)

H_STEPS = {"15min": 3, "30min": 6}
DAY_START, DAY_END = "06:00", "21:00"


class ForecastPolicy(SignalControlPolicy):
    """Sizes green for the worst of current, 15-min and 30-min forecast speed."""

    def __init__(self, config: SignalSimulationConfig, name: str) -> None:
        super().__init__(config)
        self.name = name

    def speed_signals(self, current_speed, forecast_speed_15m, forecast_speed_30m, upstream_neighbor_speed):
        return [current_speed, forecast_speed_15m, forecast_speed_30m]


def _load_observed(sensor_ids: Sequence[str]) -> pd.DataFrame:
    from src.data.load_data import load_raw_speed_data

    raw = load_raw_speed_data(ROOT / "data" / "raw" / "metr-la.h5")
    raw.columns = [str(c).strip() for c in raw.columns]
    raw = raw[list(sensor_ids)]
    return raw.where(np.isfinite(raw) & (raw > 0))


def load_forecast_grids(models: Iterable[str], pred_root: Path, sensor_ids: List[str]) -> Dict[str, pd.DataFrame]:
    """{f"{model}_{h}": DataFrame[origin x sensor]} for 15/30 min, test split.

    xgboost_ha comes from the Phase 1 parquet; Phase 2 models are the mean over seeds.
    """
    grids: Dict[str, pd.DataFrame] = {}
    p1 = pd.read_parquet(ROOT / "data" / "processed" / "predictions" / "person_c_test_predictions.parquet",
                         columns=["origin_timestamp", "sensor_id", "pred_xgboost_ha_15min", "pred_xgboost_ha_30min"])
    p1["sensor_id"] = p1["sensor_id"].astype(str)
    for h in H_STEPS:
        grids[f"xgboost_ha_{h}"] = p1.pivot(index="origin_timestamp", columns="sensor_id",
                                            values=f"pred_xgboost_ha_{h}")[sensor_ids]
    for m in models:
        files = sorted((pred_root / m).glob("seed*_test.npz"))
        if not files:
            continue
        preds = []
        for f in files:
            with np.load(f) as z:
                preds.append(z["pred"].astype(np.float64))
                origins = pd.to_datetime(z["origin_ns"])
                ids = [str(s) for s in z["sensor_ids"]]
        mean = np.mean(preds, axis=0)
        for k, h in ((1, "15min"), (2, "30min")):
            grids[f"{m}_{h}"] = pd.DataFrame(mean[:, k, :], index=origins, columns=ids)[sensor_ids]
    return grids


def select_sensors(observed: pd.DataFrame, test_days: Sequence[pd.Timestamp], ref: Dict[str, float],
                   n_sensors: int) -> List[str]:
    """Sensors with the largest share of proxy-congested readings on test weekdays (observed data only)."""
    share = {}
    for s in observed.columns:
        vals = pd.concat([observed.loc[f"{d.date()} {DAY_START}":f"{d.date()} {DAY_END}", s] for d in test_days])
        v = vals.dropna()
        share[s] = float((v < 0.6 * ref[s]).mean()) if len(v) else 0.0
    return [s for s, _ in sorted(share.items(), key=lambda kv: -kv[1])[:n_sensors]]


def run_real_forecast_benchmark(
    pred_root: Path,
    models: Sequence[str] = ("gru_local", "gat", "dcrnn"),
    n_sensors: int = 30,
    seed: int = 42,
    repetitions: int = 2000,
) -> tuple[pd.DataFrame, dict]:
    from src.phase2.evaluation.events import load_sensor_reference_speeds
    from src.phase2.evaluation.loaders import load_canonical_sensor_ids

    sensor_ids = load_canonical_sensor_ids()
    ref = load_sensor_reference_speeds()
    grids = load_forecast_grids(models, pred_root, sensor_ids)
    sources = ["xgboost_ha"] + [m for m in models if f"{m}_15min" in grids]
    observed = _load_observed(sensor_ids)
    origins = grids["xgboost_ha_15min"].index
    test_days = sorted({d for d in pd.DatetimeIndex(origins).normalize() if d.dayofweek < 5})
    # keep days whose 06:00-21:00 window is fully inside the test origins
    test_days = [d for d in test_days if pd.Timestamp(f"{d.date()} {DAY_START}") in origins
                 and pd.Timestamp(f"{d.date()} {DAY_END}") in origins]
    chosen = select_sensors(observed, test_days, ref, n_sensors)
    filled = observed.ffill()

    cfg = SignalSimulationConfig(seed=seed)
    C = cfg.cycle_length_s
    sim = IntersectionQueueSimulator(cfg)
    policies = [FixedSignalPolicy(cfg), ReactiveSignalPolicy(cfg)] + \
               [ForecastPolicy(cfg, f"forecast_{s}") for s in sources] + [ForecastPolicy(cfg, "forecast_perfect")]

    rows = []
    for d in test_days:
        t0 = pd.Timestamp(f"{d.date()} {DAY_START}")
        t1 = pd.Timestamp(f"{d.date()} {DAY_END}")
        n_cycles = int((t1 - t0).total_seconds() // C)
        cyc_t = t0 + pd.to_timedelta(np.arange(n_cycles) * C, unit="s")
        org = cyc_t.floor("5min")
        for s in chosen:
            cur = filled[s].reindex(org).to_numpy(float)
            if np.isnan(cur).any():
                continue
            fc = {}
            for src in sources:
                fc[src] = tuple(grids[f"{src}_{h}"][s].reindex(org).to_numpy(float) for h in H_STEPS)
            fc["perfect"] = tuple(filled[s].reindex(org + pd.Timedelta(minutes=5 * k)).to_numpy(float)
                                  for k in H_STEPS.values())
            for pol in policies:
                if pol.name.startswith("forecast_"):
                    f15, f30 = fc[pol.name[len("forecast_"):]]
                else:
                    f15 = f30 = None
                r = sim.run_simulation(pol, cur, f15, f30, None, scenario_name=f"{s}@{d.date()}").to_row()
                r.update({"sensor_id": s, "day": str(d.date())})
                rows.append(r)
    df = pd.DataFrame(rows)

    # Paired comparison vs the Phase 1 forecast, bootstrap over days (all sensors of a day together)
    piv = df.pivot_table(index=["day", "sensor_id"], columns="policy_name", values="mean_delay_s")
    days = sorted(piv.index.get_level_values(0).unique())
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(days), size=(repetitions, len(days)))
    comparisons = {}
    for pol in piv.columns:
        if pol == "forecast_xgboost_ha":
            continue
        diff = (piv["forecast_xgboost_ha"] - piv[pol]).groupby(level=0).agg(["sum", "count"]).loc[days]
        s_, n_ = diff["sum"].to_numpy(), diff["count"].to_numpy()
        boot = s_[draws].sum(1) / n_[draws].sum(1)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        comparisons[pol] = {"delay_reduction_vs_xgboost_ha_s": round(float(s_.sum() / n_.sum()), 3),
                            "ci95_low": round(float(lo), 3), "ci95_high": round(float(hi), 3)}

    summary = {
        "metadata": {
            "evaluation_type": "simulated_signal_control_real_inputs",
            "disclaimer": ("Software simulation of one signalised approach. Speeds are real METR-LA test-period "
                           "readings and forecasts are the trained models' test predictions, but arrivals are "
                           "generated from speed by an assumed rule; not a field result."),
            "window": f"test weekdays {DAY_START}-{DAY_END} stored clock, {len(days)} days",
            "sensors": chosen,
            "sensor_selection": "largest share of proxy-congested readings on test weekdays (observed data only)",
            "forecast_sources": sources + ["perfect (observed future speed, reference only)"],
            "policy_signals": "current speed + 15-min and 30-min forecasts from the latest 5-min origin",
            "cycle_length_s": C, "runs": int(len(df)), "bootstrap": f"{repetitions} day resamples, seed {seed}",
        },
        "policies": {
            pol: {"mean_delay_s": round(float(g["mean_delay_s"].mean()), 2),
                  "spillback_rate_pct": round(float(g["spillback_rate_pct"].mean()), 2),
                  "fuel_consumption_proxy": round(float(g["fuel_consumption_proxy"].mean()), 2)}
            for pol, g in df.groupby("policy_name")
        },
        "paired_vs_xgboost_ha": comparisons,
    }
    return df, summary
