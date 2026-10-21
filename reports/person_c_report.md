# Person C — Baselines and XGBoost

**Generated:** 2026-10-21 09:22 UTC
**Scope:** persistence, historical average and two XGBoost variants, one model per horizon, own-sensor features only (no neighbours, no graph).

Speeds are in source units (likely mph, unverified). Every score uses valid targets only (`target_valid_<h> == True`). Fitted on train and early-stopped on validation. Completion regenerates predictions from the frozen boosters without test-based tuning.

## Models

- **persistence**: prediction is the current speed.
- **historical average**: mean valid speed per (sensor, day of week, 5-min slot) from train, looked up at the **target** time (DCRNN-style baseline).
- **XGBoost (32 shared features)**: exactly Person B's 32 features, the same inputs as Person D's Random Forest, so this is the like-for-like comparison.
- **XGBoost + hist. avg features**: the same 32 features plus four extra inputs from the sensor's own training history (`ha_target`, `ha_origin`, `ha_delta`, `anomaly_now`). **This goes beyond Person B's 32-feature contract** and is kept as a separate, clearly labelled model. On training rows these are computed leave-one-out (a row's own reading is removed from its average) so they do not leak the target.

## Test results (final)

| Horizon | Model | MAE | RMSE | MAPE % |
|---|---|---:|---:|---:|
| 5min | persistence | 2.814 | 4.627 | 6.49 |
| 5min | historical average | 4.188 | 7.855 | 13.04 |
| 5min | XGBoost (32 shared features) | 2.362 | 4.151 | 5.76 |
| 5min | **XGBoost + hist. avg features** | 2.312 | 4.065 | 5.66 |
| 15min | persistence | 3.499 | 6.418 | 8.61 |
| 15min | historical average | 4.188 | 7.855 | 13.04 |
| 15min | XGBoost (32 shared features) | 2.970 | 5.806 | 7.92 |
| 15min | **XGBoost + hist. avg features** | 2.844 | 5.540 | 7.63 |
| 30min | persistence | 4.206 | 8.052 | 10.86 |
| 30min | historical average | 4.188 | 7.855 | 13.04 |
| 30min | XGBoost (32 shared features) | 3.564 | 7.129 | 10.27 |
| 30min | **XGBoost + hist. avg features** | 3.275 | 6.546 | 9.49 |
| 60min | persistence | 5.380 | 10.344 | 14.53 |
| 60min | historical average | 4.189 | 7.855 | 13.04 |
| 60min | XGBoost (32 shared features) | 4.393 | 8.720 | 13.74 |
| 60min | **XGBoost + hist. avg features** | 3.688 | 7.470 | 11.34 |

## Validation results (MAE)

| Horizon | Persistence | Hist. avg | XGBoost (32) | XGBoost + HA | XGBoost + HA RMSE | XGBoost + HA MAPE % | Gain vs persistence |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5min | 2.630 | 4.333 | 2.189 | **2.158** | 3.836 | 5.16 | 17.9% |
| 15min | 3.225 | 4.332 | 2.723 | **2.639** | 5.241 | 7.00 | 18.2% |
| 30min | 3.830 | 4.329 | 3.234 | **3.047** | 6.222 | 8.75 | 20.5% |
| 60min | 4.806 | 4.320 | 3.921 | **3.461** | 7.141 | 10.44 | 28.0% |

## Peak vs off-peak (test)

Peak = weekday 07:00–10:00 or 16:00–19:00 at the **origin** time (Person B's flags). Off-peak is everything else, weekends included.

| Horizon | Period | Rows | Persistence MAE | Hist. avg MAE | XGBoost (32) MAE | XGBoost + HA MAE | XGBoost + HA RMSE | XGBoost + HA MAPE % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 5min | all | 1,245,052 | 2.814 | 4.188 | 2.362 | **2.312** | 4.065 | 5.66 |
| 5min | peak | 250,366 | 3.293 | 6.567 | 2.978 | **2.929** | 5.090 | 9.25 |
| 5min | off_peak | 994,686 | 2.693 | 3.590 | 2.207 | **2.157** | 3.764 | 4.76 |
| 5min | morning_peak | 121,561 | 2.868 | 5.715 | 2.587 | **2.526** | 4.515 | 7.31 |
| 5min | evening_peak | 128,805 | 3.695 | 7.370 | 3.348 | **3.309** | 5.578 | 11.07 |
| 15min | all | 1,245,040 | 3.499 | 4.188 | 2.970 | **2.844** | 5.540 | 7.63 |
| 15min | peak | 250,050 | 4.619 | 6.625 | 4.170 | **3.992** | 7.374 | 13.10 |
| 15min | off_peak | 994,990 | 3.217 | 3.576 | 2.668 | **2.555** | 4.974 | 6.25 |
| 15min | morning_peak | 121,183 | 4.053 | 5.807 | 3.676 | **3.469** | 6.688 | 10.43 |
| 15min | evening_peak | 128,867 | 5.152 | 7.394 | 4.636 | **4.484** | 7.965 | 15.62 |
| 30min | all | 1,245,022 | 4.206 | 4.188 | 3.564 | **3.275** | 6.546 | 9.49 |
| 30min | peak | 249,616 | 5.987 | 6.649 | 5.304 | **4.863** | 8.896 | 16.51 |
| 30min | off_peak | 995,406 | 3.760 | 3.571 | 3.128 | **2.877** | 5.810 | 7.73 |
| 30min | morning_peak | 120,641 | 5.294 | 5.879 | 4.747 | **4.273** | 8.176 | 13.30 |
| 30min | evening_peak | 128,975 | 6.636 | 7.370 | 5.825 | **5.414** | 9.520 | 19.50 |
| 60min | all | 1,244,986 | 5.380 | 4.189 | 4.393 | **3.688** | 7.470 | 11.34 |
| 60min | peak | 250,056 | 8.144 | 6.500 | 6.565 | **5.563** | 10.125 | 19.36 |
| 60min | off_peak | 994,930 | 4.686 | 3.608 | 3.848 | **3.217** | 6.637 | 9.33 |
| 60min | morning_peak | 120,732 | 6.898 | 5.845 | 5.872 | **4.949** | 9.381 | 15.54 |
| 60min | evening_peak | 129,324 | 9.308 | 7.112 | 7.211 | **6.135** | 10.773 | 22.92 |

## Against Person D's Random Forest (test, full 32 features)

Person D's numbers are read from `data/metadata/person_d_results.json` and not modified.

| Horizon | RF MAE | XGBoost (32) MAE | XGBoost + HA MAE | RF RMSE | XGBoost (32) RMSE | XGBoost + HA RMSE | RF MAPE % | XGBoost + HA MAPE % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5min | 2.452 | 2.362 | **2.312** | 4.156 | 4.151 | 4.065 | 6.02 | 5.66 |
| 15min | 3.167 | 2.970 | **2.844** | 5.763 | 5.806 | 5.540 | 8.56 | 7.63 |
| 30min | 3.871 | 3.564 | **3.275** | 7.002 | 7.129 | 6.546 | 11.10 | 9.49 |
| 60min | 4.854 | 4.393 | **3.688** | 8.410 | 8.720 | 7.470 | 14.48 | 11.34 |

The like-for-like comparison is the RF against **XGBoost (32)**: same inputs. XGBoost + HA also uses the extra history features, so its lead over RF mixes a model effect and a feature effect.

## What the results say

- **XGBoost on the 32 shared features does not beat the historical average on test at 60min** (it does on validation). The shared features do not explicitly supply a sensor's target-time historical profile. The separate HA extension tests the value of that additional local information; this comparison does not establish a causal explanation for every error.
- Adding the historical-average features fixes this: XGBoost + HA beats every baseline on MAE and RMSE at every horizon on test.
- On the same 32 features, XGBoost beats Person D's Random Forest on MAE and MAPE at every horizon, but the RF has the lower RMSE at 15min, 30min, 60min. XGBoost is trained on absolute error and RF on squared error, so each favours its own metric.
- Errors are higher in the defined peak periods, especially evening origins. This is a descriptive result on the stored clock; no verified physical cause or graph-model advantage is inferred.

## Model details

Tree counts and fit times below are preserved original training metadata, not the elapsed time of prediction regeneration.

| Horizon | Model | Trees (early stopped) | Fit time (s) | Train rows |
|---|---|---:|---:|---:|
| 5min | XGBoost (32 shared features) | 1095 | 117.7 | 4,607,437 |
| 5min | XGBoost + hist. avg features | 984 | 111.1 | 4,607,437 |
| 15min | XGBoost (32 shared features) | 836 | 89.6 | 4,607,441 |
| 15min | XGBoost + hist. avg features | 815 | 93.3 | 4,607,441 |
| 30min | XGBoost (32 shared features) | 1048 | 108.1 | 4,607,439 |
| 30min | XGBoost + hist. avg features | 726 | 81.7 | 4,607,439 |
| 60min | XGBoost (32 shared features) | 1259 | 125.5 | 4,607,444 |
| 60min | XGBoost + hist. avg features | 699 | 79.7 | 4,607,444 |

Params: `tree_method=hist`, `objective=reg:absoluteerror`, `eval_metric=mae`, `max_depth=8`, `eta=0.1`, `min_child_weight=50`, `subsample=0.8`, `colsample_bytree=0.8`, `seed=42`

### Preserved training design

- **Residual target.** The preserved boosters predict `target − speed_lag_0`; current speed is added back at inference.
- **Absolute-error loss** (`reg:absoluteerror`) and validation early stopping are recorded in the saved model and fit metadata.
- Completion re-evaluates frozen models. It does not repeat the original hyperparameter search or use test errors to select new settings.
- **Leave-one-out history features on train rows.** Without it, a training row's own target sits inside its average and the model over-trusts the feature.

## Top features (gain, normalised)

- **5min, 32 features:** `speed_change_5min` 0.236, `speed_change_15min` 0.162, `speed_change_30min` 0.096, `speed_lag_0` 0.034, `rolling_std_15min` 0.031
- **5min, + HA:** `anomaly_now` 0.248, `speed_change_5min` 0.156, `speed_change_15min` 0.102, `speed_change_30min` 0.047, `speed_missing_flag` 0.032
- **15min, 32 features:** `speed_change_5min` 0.251, `speed_change_15min` 0.168, `speed_change_30min` 0.079, `is_weekend` 0.040, `hour_cos` 0.032
- **15min, + HA:** `anomaly_now` 0.247, `speed_change_5min` 0.149, `speed_change_15min` 0.092, `speed_missing_flag` 0.061, `ha_delta` 0.053
- **30min, 32 features:** `speed_change_5min` 0.210, `speed_change_15min` 0.135, `speed_change_30min` 0.072, `is_weekend` 0.050, `speed_lag_0` 0.043
- **30min, + HA:** `anomaly_now` 0.246, `speed_change_5min` 0.119, `ha_delta` 0.088, `speed_change_15min` 0.067, `speed_missing_flag` 0.056
- **60min, 32 features:** `speed_change_5min` 0.134, `speed_change_15min` 0.095, `speed_lag_0` 0.086, `is_weekend` 0.077, `is_evening_peak` 0.051
- **60min, + HA:** `anomaly_now` 0.240, `ha_delta` 0.146, `speed_lag_0` 0.105, `speed_change_5min` 0.068, `ha_target` 0.058

## Caveats

- The test period has more missing readings (87.9% valid targets against 92.9% in train), so test and validation errors are not directly comparable.
- Peak/off-peak uses the origin time. For the 60 min horizon the target falls up to an hour later.
- The timezone of the stored clock is unverified, so 'peak' means peak on the stored clock.
- Phase 1 uses each sensor's own history only. No neighbouring sensors or graph inputs.
- The history features use ~12 weeks of training days per (sensor, weekday, slot); holidays are not modelled.

## Handoff (Person E)

- `data/metadata/person_c_results.json` records model-run metrics at four decimals. After common evaluation, displayed headline tables use unrounded `phase1_metrics.csv` scores to avoid double rounding.
- `data/processed/predictions/person_c_{val,test}_predictions.parquet`: one row per (origin, sensor) with target, validity flag and all four models' predictions per horizon, for maps and plots. Rebuild with `python scripts/run_person_c.py`.
- `models/person_c/xgb_<h>.json` and `xgb_ha_<h>.json` are the trained boosters. They predict the change; add `speed_lag_0` to get speed.
- Code: `src/models/baselines.py`, `src/models/xgboost_model.py`, `scripts/run_person_c.py`, `tests/test_person_c.py`.
