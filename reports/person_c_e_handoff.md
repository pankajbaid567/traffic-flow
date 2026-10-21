# Person C to Person E verified forecast handoff

The eight frozen XGBoost boosters have been re-scored with official XGBoost 3.4.1. Validation and test predictions are saved in `data/processed/predictions/person_c_{val,test}_predictions.parquet`. Each row carries the origin timestamp, sensor ID, peak flags, current speed, raw targets, validity flags and all four Person C forecasts at all four horizons.

Join Person D forecasts only after verifying identical origin/sensor keys, targets, masks and row order. Use `scripts/run_person_e.py` to recompute metrics, rather than copying numerical summaries. Current all-model scores follow. The 32-feature RF/XGBoost comparison is like for like; the 36-feature HA extension is separate.

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

The boosters predict a change from current speed, which must be added back. HA lookups are fitted on training timestamps only, queried at the forecast target time, and leave-one-out for training features. Targets remain unimputed; score finite positive observations only. Peak flags refer to weekday origin times on the unverified stored clock. Use source speed units and verified coordinates without assigning unverified freeway identities. Frozen-model regeneration is for reproducibility and does not introduce test-based tuning. Full evidence and current limitations are in `person_e_report.md`.
