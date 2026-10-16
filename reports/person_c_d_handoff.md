# Handoff notes: Person B → Person C and Person D

Hey C and D,

My part (Person B, features + dataset prep) is done, and the data is ready to train on. This note covers what I built, where it is, and the few rules we all need to follow so our numbers can be compared. Please read it before you start.

## What I did

Person A gave us the raw METR-LA speeds: 207 sensors with a reading every 5 minutes from 1 Mar to 27 Jun 2012. From that I:

1. loaded it and checked the sensor order against Person A's `sensors.csv`
2. handled missing readings (about 8.1% of readings are zero, and a zero means "no reading")
3. built 32 features per sensor per timestamp
4. cut train/val/test using Person A's split contract, without leaking future data
5. saved everything as parquet so you can just load it and train

## Files

```
data/processed/            (git-ignored, rebuild with the command below)
├── train.parquet          ← training
├── val.parquet            ← tuning / early stopping
├── test.parquet           ← final evaluation only
└── dataset_summary.json   ← feature lists, row counts, valid counts, sha256, training medians
data/metadata/
├── split_contract.json    ← Person A's split boundaries, don't change
└── sensors.csv            ← the 207 sensors
```

If you don't have the parquets, rebuild them from the project root (about 12 s):

```bash
/opt/anaconda3/bin/python -m src.data.dataset_builder
```

The output is deterministic. Check your sha256 values against `dataset_summary.json`. Mine start with train `a50806c6…`, val `92130f50…` and test `2d87322d…`.

## Loading

```python
from src.data.dataset_builder import load_partition

train_df = load_partition("train")
val_df   = load_partition("val")
test_df  = load_partition("test")

print(train_df.shape)   # (4961169, 42) = 23,967 origins x 207 sensors
```

Each file has 42 columns: `origin_timestamp`, `sensor_id`, 32 features, 4 targets and 4 validity flags. `sensor_id` loads as a pandas category. It is **not** in the feature list, so leave it out of `X` unless you deliberately want a sensor-ID feature and tell everyone.

## Split

| Split | Grid timestamps | Origins | Rows | First origin | Last origin |
|---|---:|---:|---:|---|---|
| Train | 23,990 | 23,967 | 4,961,169 | 2012-03-01 00:55 | 2012-05-23 06:05 |
| Val | 3,427 | 3,416 | 707,112 | 2012-05-23 07:05 | 2012-06-04 03:40 |
| Test | 6,855 | 6,844 | 1,416,708 | 2012-06-04 04:40 | 2012-06-27 22:55 |

Two rules decide which origins are kept:

- An origin needs 12 steps of history (t-11 … t).
- All four of its targets must fall inside the same split. Origins whose targets would cross into the next split are dropped.

History is allowed to come from the previous split, since that data was already observed at time t. That is why the first val origin (07:05) is the last train timestamp: its first target, 07:10, is already in val.

## The 32 features

All of these use only the sensor's own past speed, and every window ends at t.

**Lags (12):** `speed_lag_0` (now) through `speed_lag_11` (55 min ago).

**Rolling (8):**
- `rolling_mean_15min`, `rolling_mean_30min`, `rolling_mean_60min`
- `rolling_std_15min`, `rolling_std_30min`, `rolling_std_60min`
- `rolling_min_30min`, `rolling_max_30min`

**Trend (3):** `speed_change_5min`, `speed_change_15min` and `speed_change_30min`. Each is speed now minus speed 1, 3 or 6 steps ago.

**Time (7):** These are computed at the origin time, on the stored clock (the timezone is unverified, see Person A's notes).
- `hour_sin` and `hour_cos`
- `dow_sin` and `dow_cos`
- `is_weekend`
- `is_morning_peak`: weekday 07:00–10:00
- `is_evening_peak`: weekday 16:00–19:00

**Missingness (2):**
- `speed_missing_flag`: 1 if the reading at t was zero or missing
- `missing_count_60min`: how many of the last 12 readings were missing

## Targets and validity flags

| Target | Means | Flag |
|---|---|---|
| `target_5min` | speed at t+1 | `target_valid_5min` |
| `target_15min` | speed at t+3 | `target_valid_15min` |
| `target_30min` | speed at t+6 | `target_valid_30min` |
| `target_60min` | speed at t+12 | `target_valid_60min` |

Targets are raw readings. **I never filled them.** A zero target is still in the column, but its flag is `False`. **Filter on the flag before you train or score:**

```python
h = "15min"
m = train_df[f"target_valid_{h}"]
X_train = train_df.loc[m, features]
y_train = train_df.loc[m, f"target_{h}"]
```

Valid rows (each fraction is the same for all four horizons to 4 decimal places):

| Split | Valid targets per horizon (5 / 15 / 30 / 60 min) | Fraction |
|---|---|---:|
| Train | ~4.607M | 92.87% |
| Val | 657,759 / 657,752 / 657,743 / 657,751 | 93.02% |
| Test | 1,245,052 / 1,245,040 / 1,245,022 / 1,244,986 | 87.88% |

Exact counts are in `dataset_summary.json` under `partitions → <split> → target_statistics`. Person E needs these per horizon for the report.

## Units

The speeds are in source units. Person A could not confirm the units from the upstream metadata. They are probably mph, so write "speed units (likely mph)" in reports and don't state mph as fact.

## Person C: baselines and XGBoost

**Persistence** (do this first). The prediction is just `speed_lag_0`:

```python
h = "15min"
m = val_df[f"target_valid_{h}"]
mae = (val_df.loc[m, f"target_{h}"] - val_df.loc[m, "speed_lag_0"]).abs().mean()
```

**Historical average.** Use the mean speed per (sensor, day of week, time of day), with two catches:

- Fit it on **train only**, and skip zero readings when you compute the means (the upstream DCRNN baseline does the same).
- Look the average up at the **target** time, not the origin time. For the 60-min horizon that is `origin_timestamp + 60 min`. Derive hour/dow from `origin_timestamp + pd.Timedelta(minutes=...)`; don't reuse the sin/cos columns, which describe the origin.

**XGBoost.** Train one model per horizon, so four models:

```python
import json
import xgboost as xgb

summary = json.load(open("data/processed/dataset_summary.json"))
features = summary["feature_groups"]["full_features"]   # 32 columns

h = "15min"
mt, mv = train_df[f"target_valid_{h}"], val_df[f"target_valid_{h}"]
model = xgb.XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=42)
model.fit(train_df.loc[mt, features], train_df.loc[mt, f"target_{h}"],
          eval_set=[(val_df.loc[mv, features], val_df.loc[mv, f"target_{h}"])], verbose=50)
```

## Person D: Random Forest and ablations

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
rf.fit(train_df.loc[mt, features], train_df.loc[mt, f"target_{h}"])
```

Training a full RF on ~4.6M rows is slow and uses a lot of RAM. `max_samples=0.2` (bootstrap subsampling) helps a lot. Whatever you choose, use the same setting for all four ablations so they stay comparable.

The ablation groups are already in `dataset_summary.json → feature_groups`:

| Exp | Group | Features | Count |
|---|---|---|---:|
| A | `lag_only` | 12 lags | 12 |
| B | `lag_temporal` | lags + time | 19 |
| C | `lag_temporal_rolling` | lags + time + rolling | 27 |
| D | `full_features` | everything | 32 |

```python
for exp, key in [("A", "lag_only"), ("B", "lag_temporal"),
                 ("C", "lag_temporal_rolling"), ("D", "full_features")]:
    feats = summary["feature_groups"][key]
    ...
```

## How missing values were handled

1. Zero, NaN, inf and negative readings are all treated as missing.
2. Missing inputs are forward-filled from the last real reading. This only looks backwards; there is no backfill or interpolation.
3. If a sensor has no earlier reading at all, the gap is filled with that sensor's median. The median is computed from training timestamps only, and the values are saved in `dataset_summary.json → preprocessing → training_medians`.
4. Targets are left raw, which is why the flags exist.

Forward fill has no time limit. During long outages, including the 97 blocks where every sensor reads zero, the lags just repeat the last real value for hours. `speed_missing_flag` and `missing_count_60min` are there so the model can tell when that is happening.

## Rules

1. **Don't rebuild the splits yourself.** Use these parquets or rerun my builder.
2. **Don't look at test data until final evaluation.** Use val for all tuning.
3. **Filter on `target_valid_*`** before training and before scoring.
4. **Phase 1 is local only.** No neighbouring sensors and no graph features.
5. **Use one model per horizon.** No multi-output or recursive models.
6. **Use Person E's metric code** (MAE/RMSE/MAPE) once it is ready, so we all compute errors the same way.

## Sanity check: persistence MAE

Your models should beat these numbers. I recomputed them from the parquets on valid targets only.

| Horizon | Val | Test |
|---|---:|---:|
| 5 min | 2.630 | 2.814 |
| 15 min | 3.225 | 3.499 |
| 30 min | 3.830 | 4.206 |
| 60 min | 4.806 | 5.380 |

If XGBoost or RF can't beat persistence on val, something is wrong. The usual causes are unfiltered zero targets or a mismatched horizon column.

## If something looks off

- `reports/person_b_handoff.md` is Person A's contract to me (missing-value rules and split rules).
- `src/data/features.py` has the feature definitions.
- `src/data/dataset_builder.py` runs the whole pipeline.
- Tests: `/opt/anaconda3/bin/python -m pytest tests/`

Message me if anything doesn't add up.

— Person B
