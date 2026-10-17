# Handoff notes: Person C → Person E

Hi E,

Person C (baselines + XGBoost) is done. This note says what you can use, where it is, and the rules and pitfalls for scoring and plotting. Person D's Random Forest results are already in the repo (`reports/person_d_report.md`, `data/metadata/person_d_results.json`), so you have both sides for the Phase-1 comparison.

Full numbers and discussion are in [`person_c_report.md`](person_c_report.md). This note is the practical guide.

## What I built

| Model | Inputs | Notes |
|---|---|---|
| `persistence` | `speed_lag_0` | prediction = speed now |
| `historical_average` | none (lookup table) | mean valid speed per (sensor, day of week, 5-min slot) from **train only**, looked up at the **target** time. Same baseline family as DCRNN |
| `xgboost` | Person B's 32 features | one model per horizon, same inputs as Person D's RF, so this is the like-for-like comparison |
| `xgboost_ha` | the 32 features + 4 history features | **goes beyond Person B's 32-feature contract**; see below |

Both XGBoost models predict the change from the current speed (`target - speed_lag_0`) with an absolute-error loss, one model per horizon, early-stopped on validation. Only train was used to fit and test was scored once.

The 4 extra features of `xgboost_ha` (`ha_target`, `ha_origin`, `ha_delta`, `anomaly_now`) come from the sensor's own training history, so they are still local (no neighbours, no graph). On training rows they are computed leave-one-out so a row's own target is not in its average. They exist because the plain 32 features lose to the historical average at 60 min on test (see below).

## Files

```
data/processed/predictions/                      (git-ignored, ~90 MB, rebuild below)
├── person_c_val_predictions.parquet             707,112 rows
└── person_c_test_predictions.parquet            1,416,708 rows
data/metadata/person_c_results.json              every metric, all periods, importances
models/person_c/xgb_<h>.json, xgb_ha_<h>.json    8 trained boosters
reports/person_c_report.md                       tables + discussion
```

Rebuild everything (about 15 min, needs `xgboost` and `tables`; I used a project `.venv`):

```bash
.venv/bin/python -m src.data.dataset_builder      # only if data/processed/*.parquet is missing
.venv/bin/python scripts/run_person_c.py
```

## The predictions parquet (what you will plot from)

One row per (origin timestamp, sensor), 29 columns, same row order as Person B's parquet.

| Column | Meaning |
|---|---|
| `origin_timestamp`, `sensor_id` | the forecast origin and the sensor (`sensor_id` is a string) |
| `is_morning_peak`, `is_evening_peak` | Person B's peak flags **at the origin time** |
| `speed_lag_0` | speed at the origin (forward-filled input) |
| `target_<h>`, `target_valid_<h>` | raw future speed and whether it is usable |
| `pred_<model>_<h>` | prediction, where `<model>` is `persistence`, `historical_average`, `xgboost` or `xgboost_ha` |

`<h>` is `5min`, `15min`, `30min` or `60min`.

```python
import pandas as pd
from src.models.metrics import compute_metrics     # Person D's metric code, same as everyone uses

p = pd.read_parquet("data/processed/predictions/person_c_test_predictions.parquet")
h = "15min"
m = p[f"target_valid_{h}"]
compute_metrics(p.loc[m, f"target_{h}"].to_numpy(), p.loc[m, f"pred_xgboost_ha_{h}"].to_numpy())
```

Join sensor locations from `data/metadata/sensors.csv` (`sensor_id`, `latitude`, `longitude`) for congestion maps. Its `sensor_id` is a string too.

## Rules and pitfalls

1. **Always filter on `target_valid_<h>`** before scoring. Predictions exist for every row, including rows whose target is a zero/missing reading. Scoring those gives nonsense (this is the usual cause of absurd MAPE).
2. **Test is the final result.** Validation was used for early stopping and design choices. Report test as the headline and validation as supporting.
3. **Test and validation are not directly comparable.** Test has more outages (87.9% valid targets against 93.0% in validation and 92.9% in train) and is more volatile. Persistence MAE at 60 min is 5.38 on test against 4.81 on validation.
4. **Peak/off-peak uses the origin time.** Peak = weekday 07:00-10:00 or 16:00-19:00. Off-peak includes weekends. For 60 min the target is up to an hour later than the origin. If you prefer target-time periods, shift the timestamp yourself and say so.
5. **Units and clock.** Speeds are in source units (likely mph, unverified) and the timezone is unverified. Write "speed units (likely mph)" and "peak on the stored clock".
6. **Keep the two XGBoost models apart in the report.** `xgboost` vs RF is like-for-like. `xgboost_ha` also has extra history features, so its lead over RF mixes a model effect and a feature effect.

## Headline results (test MAE, valid targets only)

| Horizon | Persistence | Hist. avg | RF (D) | XGBoost (32) | XGBoost + HA |
|---|---:|---:|---:|---:|---:|
| 5 min | 2.814 | 4.188 | 2.453 | 2.362 | **2.312** |
| 15 min | 3.499 | 4.188 | 3.168 | 2.970 | **2.844** |
| 30 min | 4.206 | 4.188 | 3.871 | 3.563 | **3.275** |
| 60 min | 5.380 | 4.189 | 4.855 | 4.390 | **3.688** |

Things worth showing and being careful about:

- **XGBoost (32) does not beat the historical average at 60 min on test** (4.39 against 4.19), though it does on validation. That is a real finding, not a bug. The 32 features carry no sensor identity, and `xgboost_ha` fixes it. It also supports the Phase-2 story: independent local models are limited by what one sensor's own history can tell them.
- **RF has a slightly lower RMSE than XGBoost (32)** at 15, 30 and 60 min. XGBoost wins on MAE and MAPE. XGBoost uses absolute-error loss and RF squared error, so each favours its own metric. Show all three metrics, don't pick one.
- **Peak errors are much higher than off-peak, and evening is worse than morning.** For example at 15 min on test, `xgboost_ha` MAE is lower off-peak than at peak, with the exact values in `person_c_results.json` under `horizons -> <h> -> test -> xgboost_ha -> {peak, off_peak, morning_peak, evening_peak}`.
- **The historical average is horizon-independent by construction** (about 4.19 at every horizon). That is expected, not a bug. It also matches DCRNN's published historical-average figure for METR-LA (about 4.16).

## JSON layout

```
person_c_results.json
  metadata
  horizons -> <h> -> fit, fit_ha                      trees, fit time, params
                  -> feature_importance_gain(_ha)     normalised gain
                  -> val | test -> <model> -> <period> -> {mae, rmse, mape, n}
```

`<model>` is `persistence | historical_average | xgboost | xgboost_ha`; `<period>` is `all | peak | off_peak | morning_peak | evening_peak`. `n` is the number of valid rows scored.

## Loading a trained model

```python
import xgboost as xgb
booster = xgb.Booster(); booster.load_model("models/person_c/xgb_ha_15min.json")
# predicts the change; add speed_lag_0 to get the speed.
# xgb_ha needs the 32 features + ha_target, ha_origin, ha_delta, anomaly_now (from src.models.baselines.HistoricalAverage.features).
```

The prediction parquets already contain everything you need for evaluation and plots, so you should not need to load the models.

## Out of scope

Phase 1 is local only: no neighbours, no graph features, no signal-control claims. Nothing here relates to the graph model or SUMO.

Message me if anything does not add up.

— Person C
