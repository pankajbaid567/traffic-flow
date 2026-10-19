# Phase 1 Capstone Synthesis Report: Applied Machine Learning Benchmarking

**Project Title:** Traffic Flow Prediction and Signal Control via Graph Attention Networks and Flow Models  
**Course / Capstone:** Applied Machine Learning & Deep Learning Project  
**Date:** 2026-10-19  
**Team Delivery:** Persons A, B, C, D, and E (Phase 1 Complete)  

---

## 1. Project Overview & Phase 1 Scope

This report serves as the comprehensive capstone synthesis of **Phase 1: Segment-Level Applied Machine Learning Baselines & Spatial Traffic Evaluation**.

The project addresses traffic congestion prediction across the real-world **METR-LA** highway sensor network in Los Angeles County. The overarching research question investigated in Phase 1 is:
> *How effectively can independent, segment-level applied machine learning models (Random Forest, XGBoost, and classical time-series baselines) forecast future traffic speeds across 207 highway sensors, and what are their fundamental physical and statistical limits when forecasting across extended horizons and rush-hour regimes?*

### Phase 1 Team Division of Responsibilities

```
                                      PROJECT LIFECYCLE (PHASE 1)
                                      
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ PERSON A: Data Engineering, Quality Audits & EDA                                      │
  │ • Canonical METR-LA HDF5 ingestion & byte preservation (34,272 timestamps × 207 sensors)│
  │ • Measured 8.109% exact zeros; established valid speed masking rule                    │
  │ • Produced 10 reviewed EDA figures & sensor metadata (data/metadata/sensors.csv)       │
  │ • Established shared chronological split contract (70% train, 10% val, 20% test)       │
  └──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                             │
  ┌──────────────────────────────────────────▼─────────────────────────────────────────────┐
  │ PERSON B: Feature Engineering & Leak-Free Dataset Builder                              │
  │ • Implemented 32 causal features (12 lags, 8 rolling, 3 trends, 7 temporal, 2 missing) │
  │ • Built causal forward-fill imputation for inputs; preserved raw targets               │
  │ • Generated train.parquet (4.96M rows), val.parquet (707K), test.parquet (1.42M)       │
  │ • Enforced boundary filtering: dropped origins whose targets cross split partitions    │
  └─────────────────────┬───────────────────────────────────────────┬──────────────────────┘
                        │                                           │
  ┌─────────────────────▼────────────────────┐┌─────────────────────▼──────────────────────┐
  │ PERSON C: Baselines & XGBoost Boosters   ││ PERSON D: Random Forest & Ablations        │
  │ • Persistence & Historical Average baselines││ • Multi-horizon Random Forest models       │
  │ • XGBoost (32 shared features) via L1 loss││ • 4 Feature Ablation experiments (Exp A-D)│
  │ • XGBoost + HA (36 features, residual fit)││ • Subsampling efficiency (max_samples=0.1)│
  │ • Early stopping on validation partition ││ • Gini impurity feature importance profile │
  └─────────────────────┬────────────────────┘└─────────────────────┬──────────────────────┘
                        │                                           │
  ┌─────────────────────▼───────────────────────────────────────────▼──────────────────────┐
  │ PERSON E: Lead Evaluator, Benchmark Synthesis, Spatial Mapping & Phase 2 Gap Bridge    │
  │ • Unified evaluation standards (masked MAE, RMSE, MAPE matching DCRNN benchmark)       │
  │ • Multi-model comparative benchmark matrix across all 4 horizons (5m, 15m, 30m, 60m)   │
  │ • Peak vs off-peak error analysis & morning vs evening rush-hour asymmetry             │
  │ • Spatial highway congestion maps across 207 sensors & corridor-level performance      │
  │ • Empirical gap analysis & mathematical blueprint for Phase 2 Graph Attention Networks │
  │ • Interactive storytelling notebook 03 & comprehensive viva defense guide              │
  └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Dataset & Quality Summary (Person A Foundations)

- **Dataset:** METR-LA loop detector series from Los Angeles County highways.
- **Dimensions:** 34,272 chronological timestamps sampled every 5 minutes $\times$ 207 unique sensors (7,094,304 total observations).
- **Date Span:** March 1, 2012 00:00 UTC to June 27, 2012 23:55 UTC (119 continuous days).
- **Missingness & Zeros:** 0 NaN, 0 infinity, 0 negative values. Exactly 575,302 readings (8.109%) are exact zeros, corresponding to sensor outages, transmission failures, and maintenance blocks (e.g., 2,148 all-sensor zero timestamps across 97 blocks).
- **Golden Rule of Evaluation:** Exact zeros are treated as missing/invalid benchmark data. Targets must strictly be masked on `target_valid_<h> == True`. Imputation is restricted to causal past features only; targets are never imputed.

---

## 3. Feature Pipeline & Split Protocol (Person B Foundations)

- **Chronological Split Boundaries:**
  - **Train:** 2012-03-01 00:55 to 2012-05-23 06:05 (23,967 origins $\times$ 207 = 4,961,169 rows; ~4.607M valid targets per horizon, 92.87% valid).
  - **Validation:** 2012-05-23 07:05 to 2012-06-04 03:40 (3,416 origins $\times$ 207 = 707,112 rows; ~658K valid targets, 93.02% valid).
  - **Test:** 2012-06-04 04:40 to 2012-06-27 22:55 (6,844 origins $\times$ 207 = 1,416,708 rows; ~1.245M valid targets, 87.88% valid).
- **32 Shared Local Features:**
  - 12 Autoregressive lags ($t, t-1, \dots, t-11$) covering 55 minutes of history.
  - 8 Backward-looking rolling stats (means, standard deviations, min, max across 15, 30, and 60 min).
  - 3 Short-term momentum/trend deltas ($\Delta_{5\text{m}}, \Delta_{15\text{m}}, \Delta_{30\text{m}}$).
  - 7 Diurnal and weekly cyclical time encodings (`hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `is_weekend`, peak flags).
  - 2 Missingness indicators (`speed_missing_flag`, `missing_count_60min`).

---

## 4. Master Benchmark Matrix (Synthesized by Person E)

All metrics are evaluated on out-of-time test data (1.245M valid predictions per horizon).

### Table 1: Complete Phase 1 Benchmark Scorecard

| Horizon | Model | Author | Inputs | Test MAE | Test RMSE | Test MAPE (%) | Gain vs Persistence (MAE) |
|---|---|---|---|---:|---:|---:|---:|
| **5 min** | Persistence | Person C | `speed_lag_0` | 2.814 | 4.627 | 6.49% | Baseline (0.0%) |
| | Historical Average | Person C | Day/Slot Lookup | 4.188 | 7.855 | 13.04% | -48.8% |
| | Random Forest | Person D | 32 Features | 2.453 | 4.156 | 6.03% | +12.8% |
| | XGBoost | Person C | 32 Features | 2.362 | 4.151 | 5.76% | +16.1% |
| | **XGBoost + HA** | **Person C** | **36 Features** | **2.312** | **4.065** | **5.66%** | **+17.8%** |
| **15 min** | Persistence | Person C | `speed_lag_0` | 3.499 | 6.418 | 8.61% | Baseline (0.0%) |
| | Historical Average | Person C | Day/Slot Lookup | 4.188 | 7.855 | 13.04% | -19.7% |
| | Random Forest | Person D | 32 Features | 3.168 | **5.764** | 8.56% | +9.5% |
| | XGBoost | Person C | 32 Features | 2.970 | 5.806 | 7.92% | +15.1% |
| | **XGBoost + HA** | **Person C** | **36 Features** | **2.844** | **5.540** | **7.63%** | **+18.7%** |
| **30 min** | Persistence | Person C | `speed_lag_0` | 4.206 | 8.052 | 10.86% | Baseline (0.0%) |
| | Historical Average | Person C | Day/Slot Lookup | 4.188 | 7.855 | 13.04% | +0.4% |
| | Random Forest | Person D | 32 Features | 3.871 | **7.003** | 11.10% | +8.0% |
| | XGBoost | Person C | 32 Features | 3.563 | 7.127 | 10.27% | +15.3% |
| | **XGBoost + HA** | **Person C** | **36 Features** | **3.275** | **6.546** | **9.49%** | **+22.1%** |
| **60 min** | Persistence | Person C | `speed_lag_0` | 5.381 | 10.344 | 14.53% | Baseline (0.0%) |
| | Historical Average | Person C | Day/Slot Lookup | 4.189 | 7.855 | 13.04% | +22.1% |
| | Random Forest | Person D | 32 Features | 4.855 | **8.410** | 14.48% | +9.8% |
| | XGBoost | Person C | 32 Features | 4.390 | 8.714 | 13.72% | +18.4% |
| | **XGBoost + HA** | **Person C** | **36 Features** | **3.688** | **7.470** | **11.34%** | **+31.5%** |

---

## 5. Key Empirical Discoveries & Methodological Insights

### 1. The Loss Function Duality: L1 (MAE) vs L2 (RMSE)
A critical insight discovered by Person E during model comparison is why Random Forest and XGBoost trade places depending on the metric:
- **XGBoost wins on MAE and MAPE:** Trained explicitly on absolute error loss (`reg:absoluteerror`), XGBoost estimates the conditional median of the residual distribution, minimizing median absolute discrepancies.
- **Random Forest wins on RMSE:** Minimizing variance / mean squared error across tree splits, Random Forest places higher penalty on large residual deviations, resulting in lower RMSE at 15m (5.764 vs 5.806), 30m (7.003 vs 7.127), and 60m (8.410 vs 8.714).
- **Takeaway:** Model comparison must report both metrics; neither model strictly dominates without specifying the operational loss function.

### 2. The 60-Minute Inflection Point & Feature Gap
- On test data at 60 minutes, **XGBoost (32 features) loses to the Historical Average baseline** (4.390 vs 4.189).
- The 32 shared features contain only local rolling speeds without sensor identity. When immediate lag correlation decays at +60 minutes, the model cannot distinguish between a high-speed corridor (I-210, 62 mph) and a chronically congested corridor (I-110, 52 mph).
- Adding the 4 leave-one-out historical features (`ha_target`, `ha_origin`, `ha_delta`, `anomaly_now`) restores superiority, driving test MAE down to **3.688**.

### 3. Rush-Hour Error Explosion & Asymmetry
- Forecasting during peak hours is markedly more difficult than off-peak hours.
- Evening peak traffic (16:00–19:00) incurs **75.5% higher MAE** than off-peak periods at 15 minutes (4.484 vs 2.555 for XGBoost + HA).
- The Historical Average baseline fails catastrophically during evening rush hour (7.394 MAE), confirming that fixed lookup tables cannot handle non-recurrent congestion waves.

### 4. Highway Corridor Vulnerability
- Spatial mapping by Person E revealed that congestion is heavily localized:
  - **I-405 (Sepulveda Pass):** 32.20% evening speed drop; highly prone to bottleneck shockwaves.
  - **I-10 (Santa Monica Fwy):** 31.26% evening speed drop.
  - **I-110 (Downtown LA Hub):** Highest sensor volatility ($\sigma = 16.12$ mph).
  - **I-210 (Foothill Fwy):** Lowest drop (13.92%) and lowest volatility ($\sigma = 8.94$ mph).

---

## 6. The Phase 2 Bridge: Transition to Graph Learning

Phase 1 successfully proved that applied machine learning models outperform persistence by up to 31.5%. However, all Phase 1 models are **segment-level isolated predictors**:
- They treat each of the 207 highway sensors as independent rows in a tabular matrix.
- They are physically blind to the directed adjacency matrix $\mathbf{A} \in \mathbb{R}^{207 \times 207}$ of the Los Angeles road graph.
- When an incident occurs downstream, backward-propagating traffic shockwaves cannot be detected until they reach the local sensor's own historical lags, resulting in severe degradation at 30 and 60 minutes.

### Phase 2 Roadmap:
1. **Model Architecture:** Deploy **Graph Attention Networks (GAT)** and **Diffusion Convolutional Recurrent Neural Networks (DCRNN)** over the road graph.
2. **Spatio-Temporal Fusion:** Combine graph convolution / attention across spatial neighbors with temporal sequence modeling (GRU / temporal attention).
3. **Simulation:** Evaluate traffic control policies using **SUMO (Simulation of Urban MObility)** or **CityFlow** without making unrealistic claims of live physical traffic actuation.

---

## 7. Phase 1 Verification & Deliverables Checklist

- [x] Canonical METR-LA speed dataset byte-verified (`data/raw/manifest.json`).
- [x] 10 EDA figures and descriptive metadata generated (`figures/`, `data/metadata/`).
- [x] Causal 32-feature pipeline and chronological partitions implemented without future leakage (`src/data/`).
- [x] Baseline models (Persistence, Historical Average) implemented and verified (`src/models/baselines.py`).
- [x] XGBoost multi-horizon models trained and persisted (`models/person_c/`, `src/models/xgboost_model.py`).
- [x] Random Forest multi-horizon models and ablation experiments A–D completed (`src/models/random_forest.py`).
- [x] Common evaluation metrics module (`src/models/metrics.py`) and benchmark engine (`src/models/evaluation.py`) implemented.
- [x] Spatial congestion maps and freeway corridor analytics completed (`src/visualization/spatial_mapping.py`, `figures/`).
- [x] Multi-horizon, peak vs off-peak, and feature importance comparison plots generated (`figures/`).
- [x] 15 comprehensive unit tests for Person E passing (`tests/test_person_e.py`).
- [x] End-to-end execution runner implemented (`scripts/run_person_e.py`).
- [x] Executed Phase 1 storytelling notebook (`notebooks/03_model_evaluation_and_congestion_maps.ipynb`).
- [x] Formal Person E report (`reports/person_e_report.md`) and Viva Defense Guide (`reports/person_e_viva_guide.md`) published.

**Phase 1 is officially complete and verified.**
