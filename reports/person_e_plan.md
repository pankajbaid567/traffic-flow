> Planning record. The completed implementation and verified results are in [person_e_report.md](person_e_report.md). Named freeway tags were not available; the completed spatial evaluation uses explicitly labelled coordinate regions, measured prediction errors and observed versus predicted congestion maps.

# Person E — Phase 1 Plan: Evaluation, Benchmark Synthesis, Spatial Mapping & Phase 2 Gap Analysis

**Role:** Person E (Lead Evaluator, Benchmark Synthesizer, Congestion Spatial Analyst & Phase 1 Integration Lead)  
**Project:** Traffic Flow Prediction and Signal Control via Graph Attention Networks and Flow Models  
**Repository:** `traffic-flow`  
**Date:** 2026-10-19  

---

## 1. Scope & Objectives

Person E is responsible for the independent, rigorous, and unified evaluation of all Phase 1 applied machine learning models, synthesizing work produced by Person A (data engineering & quality), Person B (feature engineering & leak-free chronological partitions), Person C (persistence, historical average, XGBoost baselines), and Person D (Random Forest multi-horizon models and feature ablation studies).

Person E's mandate directly fulfills the project requirements set forth in the capstone specification:
1. **Unified Evaluation Standards:** Formalize and maintain the common evaluation metric suite (`masked_mae`, `masked_rmse`, `masked_mape`, relative improvement ratios) aligned with the benchmark literature (Li et al., DCRNN, ICLR 2018).
2. **Fair Multi-Model Benchmarking:** Execute like-for-like evaluations across all 4 forecast horizons (5 min, 15 min, 30 min, 60 min) under identical chronological train/val/test splits, reporting both headline test set numbers and validation dynamics across time slices (all, overall peak, off-peak, morning peak, evening peak).
3. **Spatial Congestion Mapping:** Map predictions and forecast errors across the physical topology of the 207 METR-LA loop detectors in Los Angeles County, identifying specific highway corridor bottlenecks (I-5, I-10, I-105, I-110, I-210, I-405, SR-60, SR-91, US-101).
4. **Identify the Gap (The AML-to-Graph Bridge):** Document and empirically demonstrate the fundamental failure modes of independent, segment-level models (inability to model upstream/downstream shockwave propagation, spatial boundary blindness), motivating Phase 2's Graph Attention Networks (GAT) and Diffusion Convolutional models.
5. **Phase 1 Synthesis & Viva Preparation:** Deliver an executive capstone synthesis, interactive storytelling notebook (`03_model_evaluation_and_congestion_maps.ipynb`), and an exhaustive Viva Defense Guide for examination defense.

---

## 2. Model & Baseline Inventory for Comparison

| Model Identifier | Features | Horizon Support | Loss / Criterion | Source Person | Role in Benchmark |
|---|---|---|---|---|---|
| `persistence` | `speed_lag_0` (1 feature) | 5m, 15m, 30m, 60m | None (Naive) | Person C | Naive temporal baseline: $\hat{y}_{t+h} = y_t$ |
| `historical_average` | Train lookup table (Day, 5-min slot) | 5m, 15m, 30m, 60m | Mean valid speed | Person C | Time-of-week profile baseline (DCRNN standard) |
| `rf_exp_a` | Lag only (12 features) | 5m, 15m, 30m, 60m | Squared Error | Person D | Ablation: Pure autoregressive lags ($t, \dots, t-11$) |
| `rf_exp_b` | Lags + Temporal (19 features) | 5m, 15m, 30m, 60m | Squared Error | Person D | Ablation: Lags + diurnal & weekly cycles |
| `rf_exp_c` | Lags + Temporal + Rolling (27 features) | 5m, 15m, 30m, 60m | Squared Error | Person D | Ablation: Lags + short/long rolling stats |
| `rf_exp_d` (Full RF) | Full 32 features | 5m, 15m, 30m, 60m | Squared Error | Person D | Person D Headline Model (Random Forest) |
| `xgboost_32` | Full 32 shared features | 5m, 15m, 30m, 60m | Absolute Error (L1) | Person C | Person C Direct Like-for-Like Model |
| `xgboost_ha` | 32 features + 4 HA history features | 5m, 15m, 30m, 60m | Absolute Error (L1) | Person C | Person C Augmented History Model |

---

## 3. Execution Sequence & Work Breakdown

1. **Evaluation Engine & Metrics Formalization (`src/models/metrics.py`, `src/models/evaluation.py`):**
   - Confirm strict masking on valid target observations (`target_valid_<h> == True`).
   - Implement percentage change calculations, error distribution aggregators, and paired statistical testing.
   - Build automated loader and comparator for Person C and Person D machine-readable JSON metadata.
2. **Spatial Analytics & Congestion Mapping (`src/visualization/congestion_maps.py`):**
   - Ingest 207 sensor locations (`data/metadata/sensors.csv`) with latitude, longitude, and highway tags (`fwy`, `dir`).
   - Group sensors into major Los Angeles corridors: Santa Monica Fwy (I-10), San Diego Fwy (I-405), Golden State / Santa Ana Fwy (I-5), Harbor Fwy (I-110), Hollywood / Ventura Fwy (US-101), Foothill Fwy (I-210), Glenn Anderson Fwy (I-105).
   - Generate spatial maps comparing predicted vs actual congestion patterns during morning and evening rush hours.
3. **Benchmark Figures & Visual Analytics (`src/visualization/benchmark_plots.py`):**
   - Multi-metric horizon degradation curves (MAE, RMSE, MAPE over 5, 15, 30, 60 min).
   - Peak vs off-peak error bars and morning vs evening rush-hour asymmetry.
   - Feature importance meta-analysis comparing Random Forest Gini impurity vs XGBoost gain.
   - Spatial error heatmaps and corridor boxplots.
   - Graph propagation gap schematic demonstrating shockwave spillback blindness.
4. **Automated Pipeline Runner (`scripts/run_person_e.py`):**
   - CLI script to recompute all cross-model comparative metrics, serialize `data/metadata/person_e_results.json`, output corridor metrics, and render all figures.
5. **Testing Suite (`tests/test_person_e.py`):**
   - Unit tests verifying metric correctness, boundary checks, spatial joining, and JSON schemas.
6. **Storytelling Notebook (`notebooks/03_model_evaluation_and_congestion_maps.ipynb`):**
   - Executable, richly annotated notebook adhering to `ml-best-practices` and `notebook-guidance`.
7. **Technical Reports & Viva Documentation (`reports/`):**
   - `person_e_report.md`: Complete benchmark analysis and findings.
   - `phase1_synthesis_report.md`: Capstone summary of Phase 1.
   - `phase2_graph_gap_analysis.md`: Detailed transition to Graph Attention Networks and CityFlow/SUMO.
   - `person_e_viva_guide.md`: Comprehensive viva Q&A and defense cheat-sheet.
8. **README & Final Integration:**
   - Update `README.md` to showcase Person E's deliverables and execution commands.

---

## 4. Key Rules and Methodological Constraints

- **Filter on `target_valid_<h>`:** Missing/zero speed values (~8.1% of readings) must never enter error scoring.
- **Strict Chronological Separation:** All test evaluation is purely out-of-time (June 4, 2012 to June 27, 2012).
- **Units and Timezone Compliance:** All speed values reported as "speed units (likely mph, unverified)" and peak hours as "stored clock time", preserving Person A's quality warnings.
- **Like-for-Like Honesty:** Explicitly distinguish XGBoost (32 shared features) from XGBoost + HA (36 features) when comparing to Random Forest.
- **Traceable Deliverables:** Every figure, metric, and table backed by code and reproducible scripts.
