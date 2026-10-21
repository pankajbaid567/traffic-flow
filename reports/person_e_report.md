# Phase 1 verified evaluation report

Generated from prediction rows at 2026-10-21T09:22:39.927880+00:00. Persons A through E's data engineering, features, local forecasting and evaluation are integrated. The missing forecast maps, RF temporal slices, paired uncertainty and numerical report consistency have been completed.

## Evaluation protocol

The preserved METR-LA file has 34,272 five-minute timestamps and 207 sensors, from 2012-03-01 00:00 through 2012-06-27 23:55 on the stored clock. Its 575,302 exact-zero cells (8.109%) are treated as unavailable benchmark readings. Physical speed units and timestamp timezone remain unverified. We do not identify the causes of individual zero events.

The canonical grid contains 23,990 training, 3,427 validation and 6,855 test timestamps. Forecasts use 12 observations ending at origin t and targets t+1, t+3, t+6 and t+12. All four targets must stay in one partition. This yields 4,961,169 training, 707,112 validation and 1,416,708 test rows. Earlier observed history may cross a partition boundary. Parameters and historical averages are fitted on training timestamps only; forward filling uses past inputs, with a training-median fallback for leading gaps. Ground truth is never imputed.

Five models use the same target-validity mask, sensor order and forecast origins. The 32-feature RF and XGBoost comparison is like for like. XGBoost plus HA is a separate 36-feature extension using four features derived from the same sensor's training history; training rows exclude their own lookup observation. No graph, coordinates, sensor ID or neighboring speeds enter Phase 1 predictive inputs. Pooled fitting across sensors does not provide neighboring observations to an individual forecast.

All metrics are recomputed from aligned prediction parquets. MAE and RMSE use finite positive observed targets. MAPE uses max(y, 1e-5) as its denominator. Positive near-zero targets remain included. Empty slices have no score, and nonfinite predictions on valid targets cause an error. Peak flags describe weekday forecast origins at [07:00,10:00) or [16:00,19:00), with no unverified timezone conversion.

## Test benchmark

| Horizon | Model | MAE | RMSE | MAPE % | Valid targets |
| --- | --- | --- | --- | --- | --- |
| 5min | Persistence | 2.814 | 4.627 | 6.49 | 1,245,052 |
| 5min | Historical average | 4.188 | 7.855 | 13.04 | 1,245,052 |
| 5min | Random Forest 32 | 2.452 | 4.156 | 6.02 | 1,245,052 |
| 5min | XGBoost 32 | 2.362 | 4.151 | 5.76 | 1,245,052 |
| 5min | XGBoost plus HA 36 | 2.312 | 4.065 | 5.66 | 1,245,052 |
| 15min | Persistence | 3.499 | 6.418 | 8.61 | 1,245,040 |
| 15min | Historical average | 4.188 | 7.855 | 13.04 | 1,245,040 |
| 15min | Random Forest 32 | 3.167 | 5.763 | 8.56 | 1,245,040 |
| 15min | XGBoost 32 | 2.970 | 5.806 | 7.92 | 1,245,040 |
| 15min | XGBoost plus HA 36 | 2.844 | 5.540 | 7.63 | 1,245,040 |
| 30min | Persistence | 4.206 | 8.052 | 10.86 | 1,245,022 |
| 30min | Historical average | 4.188 | 7.855 | 13.04 | 1,245,022 |
| 30min | Random Forest 32 | 3.871 | 7.002 | 11.10 | 1,245,022 |
| 30min | XGBoost 32 | 3.564 | 7.129 | 10.27 | 1,245,022 |
| 30min | XGBoost plus HA 36 | 3.275 | 6.546 | 9.49 | 1,245,022 |
| 60min | Persistence | 5.380 | 10.344 | 14.53 | 1,244,986 |
| 60min | Historical average | 4.189 | 7.855 | 13.04 | 1,244,986 |
| 60min | Random Forest 32 | 4.854 | 8.410 | 14.48 | 1,244,986 |
| 60min | XGBoost 32 | 4.393 | 8.720 | 13.74 | 1,244,986 |
| 60min | XGBoost plus HA 36 | 3.688 | 7.470 | 11.34 | 1,244,986 |

## Temporal comparison

| Horizon | Origin period | Persistence | Historical average | Random Forest 32 | XGBoost 32 | XGBoost plus HA 36 |
| --- | --- | --- | --- | --- | --- | --- |
| 15min | off_peak | 3.217 | 3.576 | 2.840 | 2.668 | 2.555 |
| 15min | morning_peak | 4.053 | 5.807 | 3.930 | 3.676 | 3.469 |
| 15min | evening_peak | 5.152 | 7.394 | 4.972 | 4.636 | 4.484 |

For XGBoost plus HA at 15 minutes, evening-origin MAE is 75.5% higher than off-peak MAE. This is an observed stored-clock difference, not evidence of a verified physical cause.

## Spatial evaluation and interpretation

The forecast maps compare observed speed, predicted speed and per-sensor forecast MAE on identical valid test rows, for both morning and evening origin periods. Separate maps cover the full-feature RF and XGBoost plus HA at 15 minutes. A congestion-frequency map uses the declared proxy speed < 60% of each sensor's training 85th-percentile speed; this is an operational diagnostic, not a validated physical congestion label.

Regions are deterministic latitude/longitude bins. Verified freeway IDs and traffic directions are not supplied, so these bins must not be presented as confirmed named freeway corridors. Spatial summaries use held-out weekday measurements rather than full-period hourly means including weekends. Speed-drop references are training 85th percentiles, weighted by each sensor's valid peak observations. They are reference-speed proxies, not measured free-flow speeds. Descriptive speed summaries slice raw observation times; forecast-error maps slice forecast-origin times. These are explicitly different row sets.

`data/metadata/sensor_forecast_metrics.csv` and `sensor_error_rankings.csv` contain actual prediction errors, counts, observed/predicted means and congestion frequencies. `region_forecast_metrics.csv` aggregates additive errors and valid counts, avoiding unweighted averaging of sensor MAEs. The historical `corridor_performance.csv` filename now holds explicitly labelled coordinate regions.

| Coordinate region | Sensors | Morning observed speed | Evening observed speed | Evening drop from reference % |
| --- | --- | --- | --- | --- |
| Central western | 9 | 43.62 | 32.36 | 50.51 |
| Southeastern | 4 | 43.13 | 32.12 | 49.95 |
| Central eastern | 17 | 46.59 | 37.26 | 38.39 |
| North central | 63 | 55.73 | 42.18 | 36.42 |
| Western | 37 | 45.53 | 44.84 | 30.85 |
| Remaining locations | 64 | 55.66 | 54.83 | 17.83 |
| Northern | 13 | 63.37 | 59.84 | 9.44 |

## Paired uncertainty and limitations

The paired comparison retains all sensors and observations within each stored calendar day and resamples circular blocks of two consecutive days 2,000 times with seed 42. Each resample recomputes the pooled valid-target MAE difference; positive values favor the candidate. The 95% percentile intervals describe sampling variation within this test period. They do not account for every longer dependence pattern, model-selection uncertainty or generalization to other cities/seasons.

The test interval and validation interval have different unavailable-target rates. All reported metrics describe observed targets only; missing-not-at-random behavior can bias the represented traffic conditions. This single test split, fixed random seed and selected model family do not establish a universal performance ceiling. Descriptive worst-error examples are selected retrospectively on the test set for inspection and never for model tuning. Tree impurity/gain importances are model-specific diagnostics, not causal effects.

Phase 1 does not train or test GAT, DCRNN or traffic-signal control. Increasing long-horizon errors and local-input restrictions motivate a future graph comparison; they do not prove measured shockwave propagation or guaranteed graph-model improvement. Any schematic is conceptual.

| Horizon | Baseline | Candidate | MAE reduction | 95% interval |
| --- | --- | --- | --- | --- |
| 5min | Persistence | XGBoost plus HA 36 | 0.501 | [0.481, 0.518] |
| 5min | Random Forest 32 | XGBoost 32 | 0.090 | [0.086, 0.096] |
| 5min | XGBoost 32 | XGBoost plus HA 36 | 0.049 | [0.043, 0.055] |
| 15min | Persistence | XGBoost plus HA 36 | 0.655 | [0.618, 0.688] |
| 15min | Random Forest 32 | XGBoost 32 | 0.197 | [0.185, 0.215] |
| 15min | XGBoost 32 | XGBoost plus HA 36 | 0.126 | [0.107, 0.145] |
| 30min | Persistence | XGBoost plus HA 36 | 0.931 | [0.845, 1.011] |
| 30min | Random Forest 32 | XGBoost 32 | 0.306 | [0.285, 0.334] |
| 30min | XGBoost 32 | XGBoost plus HA 36 | 0.289 | [0.238, 0.338] |
| 60min | Persistence | XGBoost plus HA 36 | 1.693 | [1.466, 1.894] |
| 60min | Random Forest 32 | XGBoost 32 | 0.460 | [0.427, 0.505] |
| 60min | XGBoost 32 | XGBoost plus HA 36 | 0.706 | [0.580, 0.827] |

## Generated evidence and reproduction

- `data/metadata/phase1_metrics.csv`: all five models, two partitions, four horizons and five temporal groups.
- `data/metadata/sensor_forecast_metrics.csv`: per-sensor errors and observed/predicted congestion rates.
- `data/metadata/region_forecast_metrics.csv`: weighted errors by coordinate region.
- `data/metadata/daily_forecast_errors.csv` and `paired_bootstrap.csv`: paired comparison evidence.
- `data/metadata/person_e_results.json`: generated summaries, reproduction checks, runtime versions and input SHA-256 hashes.
- `data/processed/predictions/person_{c,d}_{val,test}_predictions.parquet`: complete forecast rows (derived, git-ignored).
- `models/person_c`: eight preserved XGBoost boosters; `models/person_d`: regenerated RF ablation models.
- `figures/person_e_*.png`: regenerated benchmark, temporal, spatial forecast, congestion, ablation and conceptual figures.

Use the commands in the README. Frozen XGBoost boosters are re-scored without test-based tuning. RF uses the original fixed 100 trees, depth 15, max_samples 0.1 and seed 42 for all 16 fits. Saved-model inference and a fixed-parameter regeneration are distinct from searching for new settings on the test set. Raw files remain unchanged.
