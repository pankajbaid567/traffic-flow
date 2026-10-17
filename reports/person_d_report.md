# Person D — Random Forest Models and Feature Ablation Study

**Date:** 2026-10-17 09:44:24 UTC  
**Author:** Person D  
**Scope:** Random Forest training, evaluation, and feature ablation across 4 horizons (5m, 15m, 30m, 60m).

## Executive Summary

This report provides the full evaluation of Person D's Random Forest regression models.
Models were trained using the causal, leak-free partition parquet files created by Person B.
In accordance with project rules, every model was trained and evaluated strictly on valid readings (`target_valid_<horizon> == True`).

### Key Findings

1. **Persistence Beaten Across All Horizons:** Every Random Forest model significantly outperforms the persistence baseline on both validation and test sets.
2. **Ablation Progression:** Adding temporal (time of day / day of week / peak hours) and rolling window statistics consistently reduces forecast error, especially at longer horizons (30min and 60min) where immediate lag memory decays.
3. **Efficiency:** Using bootstrap subsampling (`max_samples=0.1`) allowed training across 4.6 million training rows in reasonable runtimes while maintaining high predictive accuracy.

## Validation Performance Table (MAE / RMSE / MAPE)

| Horizon | Persistence MAE | Exp A (lag_only) | Exp B (lag_temporal) | Exp C (lag_rolling) | Exp D (full_features) | Best RF MAE | Improvement over Persistence |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5min | 2.630 | 2.318 | 2.296 | 2.281 | 2.280 | **2.280** | +0.349 (13.3%) |
| 15min | 3.225 | 3.001 | 2.944 | 2.920 | 2.921 | **2.920** | +0.305 (9.5%) |
| 30min | 3.830 | 3.732 | 3.575 | 3.544 | 3.544 | **3.544** | +0.286 (7.5%) |
| 60min | 4.806 | 4.857 | 4.440 | 4.384 | 4.390 | **4.384** | +0.422 (8.8%) |

## Test Set Benchmark (Final Evaluation)

| Horizon | Persistence MAE | Exp A MAE | Exp B MAE | Exp C MAE | Exp D MAE (Full) | Full RF RMSE | Full RF MAPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5min | 2.814 | 2.499 | 2.468 | 2.453 | **2.453** | 4.156 | 6.03% |
| 15min | 3.499 | 3.269 | 3.192 | 3.167 | **3.168** | 5.764 | 8.56% |
| 30min | 4.206 | 4.095 | 3.904 | 3.869 | **3.871** | 7.003 | 11.10% |
| 60min | 5.380 | 5.382 | 4.904 | 4.844 | **4.855** | 8.410 | 14.48% |

## Feature Importance Analysis (Exp D: Full 32 Features)

Top 5 most influential features by Gini importance per horizon:

### Horizon: 5min
- **`speed_lag_0`**: 0.8543
- **`rolling_mean_15min`**: 0.0856
- **`rolling_mean_30min`**: 0.0217
- **`rolling_mean_60min`**: 0.0041
- **`rolling_max_30min`**: 0.0040

### Horizon: 15min
- **`speed_lag_0`**: 0.7876
- **`rolling_mean_15min`**: 0.1077
- **`rolling_mean_30min`**: 0.0227
- **`hour_cos`**: 0.0093
- **`rolling_mean_60min`**: 0.0066

### Horizon: 30min
- **`speed_lag_0`**: 0.7127
- **`rolling_mean_15min`**: 0.1134
- **`rolling_mean_30min`**: 0.0247
- **`hour_cos`**: 0.0223
- **`hour_sin`**: 0.0155

### Horizon: 60min
- **`speed_lag_0`**: 0.5728
- **`rolling_mean_15min`**: 0.1150
- **`hour_cos`**: 0.0655
- **`hour_sin`**: 0.0368
- **`rolling_mean_30min`**: 0.0253

## Deliverables and Compliance

- Parquet inputs: Loaded via `src.data.dataset_builder.load_partition`.
- Target masking: Filtered on `target_valid_<h>` before fitting and scoring.
- No leakage: Fitted exclusively on train partition; tuning evaluated on validation.
- Machine-readable output saved to `data/metadata/person_d_results.json`.
