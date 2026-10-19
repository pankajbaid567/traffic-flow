# Traffic Flow Capstone — Phase 1 Complete (Persons A, B, C, D, and E)

Predict congestion in a road network and test whether graph-based learning improves on independent road segment forecasts.

This repository contains the complete deliverables for **Phase 1: Segment-Level Applied Machine Learning Baselines & Spatial Traffic Evaluation** on the METR-LA dataset.

---

## 1. Measured Dataset Summary

| Item | Verified Result |
| --- | --- |
| Speed table | 34,272 timestamps × 207 sensors (7,094,304 total readings) |
| Stored date range | 2012-03-01 00:00 to 2012-06-27 23:55 UTC |
| Sampling | Exactly 5 minutes; no duplicate or missing timestamps |
| NaN / infinity / negative readings | 0 / 0 / 0 |
| Exact zeros | 575,302 readings, 8.109% of all cells (treated as missing/invalid benchmark data) |
| All-sensor zero timestamps | 2,148, grouped into 97 contiguous blocks |
| Graph alignment | All speed sensor IDs and their order match the stored 207×207 adjacency |
| Constant / unusually low-variance sensors | None under the declared valid-speed rule |

The [quality report](reports/data_quality_report.md) contains the evidence and interpretation. Raw files are unchanged. Exact zeros are treated as benchmark missing/suspect, while positive near-zero speeds remain valid. No imputation is performed on targets. Speed units and timezone are unverified; plots use source units (likely mph) and the stored clock.

---

## 2. Phase 1 Team Roles & Deliverables

| Role | Author | Key Deliverables | Reports & Documentation |
|---|---|---|---|
| **Person A** | Person A | Canonical HDF5 ingestion, 10 EDA figures, sensor metadata, split contract | [person_a_plan.md](reports/person_a_plan.md), [data_quality_report.md](reports/data_quality_report.md), [verification.md](reports/verification.md) |
| **Person B** | Person B | 32-feature causal pipeline, train/val/test parquets, boundary rules | [person_b_handoff.md](reports/person_b_handoff.md), [person_c_d_handoff.md](reports/person_c_d_handoff.md) |
| **Person C** | Person C | Persistence, Historical Average, XGBoost (32), XGBoost + HA (36) | [person_c_report.md](reports/person_c_report.md), [person_c_e_handoff.md](reports/person_c_e_handoff.md) |
| **Person D** | Person D | Random Forest models (4 horizons), 4 feature ablation tiers (Exp A–D) | [person_d_report.md](reports/person_d_report.md) |
| **Person E** | **Person E** | **Unified benchmark synthesis, peak/off-peak slicing, spatial maps, Phase 2 gap bridge, Viva guide** | **[person_e_report.md](reports/person_e_report.md), [phase1_synthesis_report.md](reports/phase1_synthesis_report.md), [phase2_graph_gap_analysis.md](reports/phase2_graph_gap_analysis.md), [person_e_viva_guide.md](reports/person_e_viva_guide.md)** |

---

## 3. Master Benchmark Scorecard (Out-of-Time Test Set)

All metrics strictly filter out invalid/zero targets matching Li et al. (ICLR 2018, DCRNN benchmark). Test partition: June 4 – June 27, 2012 (~1.245M valid predictions per horizon).

| Horizon | Persistence | Hist. Avg | Random Forest (32) | XGBoost (32) | XGBoost + HA (36) | Best Model MAE | Improvement vs Persistence |
|---|---:|---:|---:|---:|---:|---:|---:|
| **5 min** | 2.814 | 4.188 | 2.453 | 2.362 | **2.312** | **XGBoost + HA** | **+17.8%** |
| **15 min** | 3.499 | 4.188 | 3.168 *(RMSE 5.764)* | 2.970 | **2.844** | **XGBoost + HA** | **+18.7%** |
| **30 min** | 4.206 | 4.188 | 3.871 *(RMSE 7.003)* | 3.563 | **3.275** | **XGBoost + HA** | **+22.1%** |
| **60 min** | 5.381 | 4.189 | 4.855 *(RMSE 8.410)* | 4.390 | **3.688** | **XGBoost + HA** | **+31.5%** |

### Key Benchmark Discoveries:
1. **L1 vs L2 Loss Duality:** On the identical 32 features, XGBoost wins on MAE and MAPE (trained with L1 absolute-error loss), while Random Forest wins on RMSE at 15m, 30m, and 60m (trained with L2 squared-error loss).
2. **The 60-Minute Inflection Point:** Plain XGBoost (32 features) loses to Historical Average at 60 min (4.390 vs 4.189) because features lack sensor identity; XGBoost + HA resolves this by adding 4 leave-one-out historical features (**3.688 MAE**).
3. **Rush-Hour Disparity:** Evening peak (16:00–19:00) error is **75.5% higher** than off-peak error at 15 min (4.484 vs 2.555 MAE).
4. **Spatial Corridor Bottlenecks:** The I-405 Sepulveda Pass (32.2% drop) and I-10 Santa Monica Freeway (31.3% drop) suffer the most severe congestion collapses, while the I-110 Downtown hub exhibits the highest speed variance ($\sigma = 16.12$ mph).

---

## 4. Run & Reproduce

```bash
# 1. Initialize environment
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt xgboost scikit-learn seaborn

# 2. Run Person E benchmark and spatial evaluation pipeline (~2.5s)
.venv/bin/python scripts/run_person_e.py

# 3. Run all unit tests for Person E (15 tests passing)
.venv/bin/python -m unittest tests/test_person_e.py -v

# 4. Run entire project unit tests (54+ tests)
.venv/bin/python -m unittest discover -s tests -v

# 5. Build and execute storytelling Notebook 03
.venv/bin/python scripts/build_notebook_03.py
```

---

## 5. Repository Structure

```
├── configs/
│   ├── features.yaml                       ← Person B feature configuration
│   └── data_quality.yaml                   ← Person A quality rules
├── data/
│   ├── metadata/
│   │   ├── sensors.csv                     ← 207 sensors with geo coordinates
│   │   ├── split_contract.json             ← Chronological split boundaries
│   │   ├── person_c_results.json           ← Person C baseline/XGBoost metrics
│   │   ├── person_d_results.json           ← Person D Random Forest metrics
│   │   ├── person_e_results.json           ← Person E master benchmark JSON
│   │   ├── corridor_performance.csv        ← Highway corridor speed drops
│   │   └── sensor_error_rankings.csv       ← Volatility rankings for 207 sensors
│   └── raw/
│       ├── graph_sensor_locations.csv      ← Sensor coordinates
│       ├── graph_sensor_ids.txt            ← Ordered sensor IDs
│       └── manifest.json                   ← Raw file checksums & provenance
├── figures/
│   ├── person_e_horizon_metrics_comparison.png
│   ├── person_e_peak_vs_offpeak_analysis.png
│   ├── person_e_congestion_map_morning_peak.png
│   ├── person_e_congestion_map_evening_peak.png
│   ├── person_e_spatial_error_corridors.png
│   ├── person_e_feature_importance_comparison.png
│   ├── person_e_rf_ablation_progression.png
│   └── person_e_graph_propagation_gap.png
├── models/
│   └── person_c/                           ← 8 trained XGBoost boosters
├── notebooks/
│   ├── 01_data_understanding.ipynb         ← Person A real-data EDA
│   ├── 02_data_quality.ipynb               ← Person A validation & zeros
│   └── 03_model_evaluation_and_congestion_maps.ipynb ← Person E executed notebook
├── reports/
│   ├── person_e_plan.md                    ← Person E master plan
│   ├── person_e_report.md                  ← Person E formal evaluation report
│   ├── phase1_synthesis_report.md          ← Complete Phase 1 capstone synthesis
│   ├── phase2_graph_gap_analysis.md        ← Gap analysis & bridge to GAT/DL
│   └── person_e_viva_guide.md              ← Comprehensive Viva Defense Guide
├── scripts/
│   ├── run_person_c.py                     ← Person C baseline/XGBoost runner
│   ├── run_person_d.py                     ← Person D Random Forest runner
│   ├── run_person_e.py                     ← Person E master evaluation runner
│   └── build_notebook_03.py                ← Notebook 03 builder & executor
├── src/
│   ├── models/
│   │   ├── metrics.py                      ← Unified evaluation metrics
│   │   ├── evaluation.py                   ← Multi-model benchmark matrix engine
│   │   ├── baselines.py                    ← Persistence & Historical Average
│   │   ├── xgboost_model.py                ← XGBoost trainers & boosters
│   │   └── random_forest.py                ← Random Forest models
│   └── visualization/
│       ├── spatial_mapping.py              ← 2D highway mapping & corridors
│       └── benchmark_plots.py              ← Multi-metric visual suite
└── tests/
    ├── test_person_c.py                    ← Person C unit tests
    ├── test_person_d.py                    ← Person D unit tests
    └── test_person_e.py                    ← Person E unit tests (15 passing)
```

---

## 6. Phase 2 Transition Roadmap

In Phase 2, the project shifts from segment-level independent applied machine learning models to **network-aware Deep Graph Learning**:
1. **Graph Attention Networks (GAT):** Dynamic spatial attention $\alpha_{ij}$ modeling backward shockwave spillbacks over the 207-node road graph.
2. **Diffusion Convolutional Recurrent Neural Networks (DCRNN):** Bidirectional graph random walks coupled with sequence-to-sequence GRU modeling.
3. **Simulated Signal Policy Experiments:** Traffic signal timing evaluation inside **SUMO (Simulation of Urban MObility)** or **CityFlow** (*strictly designated as simulated only; no live traffic control claimed*).
