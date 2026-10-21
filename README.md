# Traffic Flow Capstone Phase 1

Phase 1 integrates all five roles: A data quality/EDA, B causal features, C baselines and XGBoost, D Random Forest/ablations, E independently recomputed temporal/spatial evaluation and synthesis.

Results and limits are generated in [the verified report](reports/person_e_report.md). The IEEE-style Phase 1 paper is in `reports/paper/`. This is an academic Phase 1 study, not a deployed traffic-control system.

## Data and protocol

METR-LA: 34,272 five-minute timestamps by 207 sensors; 2012-03-01 through 2012-06-27 on the stored clock. Speed units and timezone are unverified. The 8.109% exact-zero readings are treated as unavailable, with no imputed ground truth. Raw files and checksums are preserved. The chronological grid split is 70/10/20; 12 history observations predict 5, 15, 30 and 60 minutes. All four targets stay in the same partition. Only a sensor's own history enters its forecasts.

## Verified test results

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

RF and XGBoost 32 share exactly 32 features. XGBoost plus HA uses 36 and is labelled separately. Actual per-sensor forecast errors and predicted-versus-observed maps replace the previous speed-volatility proxy. Geographic regions are coordinate bins, not verified named highways. Claims of proven shockwave propagation or inevitable graph-model gains are excluded.

## Reproduction

Use Python 3.12. Install `requirements-phase1-executed.txt` in a fresh environment for the exact completion versions; `requirements.txt` provides broader supported ranges. The verified local environment is `.phase1-venv/bin/python`, which inherits the existing scientific packages and adds official XGBoost 3.4.1. The optional numeric JSON reader also matches a scalar walk and the official runtime on 96 validation rows for each saved model. New XGBoost training requires the official package.

```bash
python -m src.data.dataset_builder
# Re-score the eight saved XGBoost boosters without retraining
NUMBA_NUM_THREADS=4 python scripts/run_person_c.py --predict-only
# Regenerate all 16 fixed-parameter RF fits and save their models
python scripts/run_person_d.py --save-models
python scripts/predict_person_d.py
python scripts/run_person_e.py
python -m unittest discover -s tests -v
python scripts/build_notebook_03.py
python scripts/verify_phase1.py
python scripts/build_phase1_paper.py
```

For a full orchestration with logs and checkpoints use `python scripts/complete_phase1.py`. Set `MPLCONFIGDIR` and `XDG_CACHE_HOME` to writable project cache directories in restricted environments. `run_person_d.py --resume --save-models` resumes an interrupted fixed-parameter run. Datasets, predictions and large RF model files are derived artifacts; transfer or regenerate them separately.

## Roles and evidence

| Person | Delivery | Evidence |
|---|---|---|
| A | Raw validation, quality, EDA | `reports/verification.md`, notebooks 01/02 |
| B | 32 features and chronological partitions | `src/data/`, `reports/person_c_d_handoff.md` |
| C | Baselines, XGBoost 32 and HA 36 | `models/person_c/`, `person_c_results.json` |
| D | RF, 12/19/27/32-feature ablations | `person_d_results.json`, saved RF models |
| E | Common metrics, all-model temporal slices, spatial forecast errors, paired uncertainty | `phase1_metrics.csv`, `sensor_forecast_metrics.csv`, `paired_bootstrap.csv` |

The detailed completion audit is `reports/phase1_completion_verification.md`. No GAT/DCRNN training or signal simulation belongs to Phase 1.
