# Person E — Viva Defense & Comprehensive Examination Guide

**Author:** Person E (Lead Evaluator, Benchmark Synthesizer, Congestion Spatial Analyst & Phase 1 Integration Lead)  
**Project:** Traffic Flow Prediction and Signal Control via Graph Attention Networks and Flow Models  
**Document Purpose:** Complete defense guide, anticipated viva examiner questions, rigorous mathematical answers, code locations, and talking points to excel in oral examinations and project reviews.  

---

## 1. Quick Summary of Person E's Contributions

When the examiner asks: *"What was your specific contribution to this project as Person E?"*, answer with confidence:

> *"As Person E, I served as the **Lead Evaluator, Benchmark Synthesizer, and Spatial Traffic Analyst** for Phase 1. My core responsibilities were:*
> 1. *Formalizing the **unified evaluation standards** (masked MAE, RMSE, MAPE) strictly filtering invalid zero readings, matching the DCRNN benchmark literature.*
> 2. *Building the **cross-model comparative benchmark engine** (`src/models/evaluation.py`), synthesizing models from Person C (XGBoost, Persistence, Historical Average) and Person D (Random Forest).*
> 3. *Evaluating performance across **temporal slices** (peak vs off-peak, morning vs evening rush hours) and quantifying **horizon degradation** across 5, 15, 30, and 60 minutes.*
> 4. *Performing **spatial highway congestion mapping** across all 207 Los Angeles loop sensors, identifying bottleneck freeway corridors (I-405, I-10, US-101, I-110).*
> 5. *Formulating **The Gap analysis**: proving empirically and physically why independent segment-level models hit a performance ceiling at 30–60 minutes due to shockwave spillback blindness, providing the mathematical justification for Graph Attention Networks in Phase 2.*
> 6. *Authoring the comprehensive evaluation reports, storytelling Notebook 03, and automated runner script (`scripts/run_person_e.py`)."*

---

## 2. Anticipated Viva Questions & Model Answers

### Q1: Why do you evaluate using Masked MAE, RMSE, and MAPE? What are the mathematical formulas and why must zeros be excluded?

**Answer:**
In traffic loop detector data, an exact reading of `0.0` does not indicate stationary traffic; it indicates a **sensor outage, hardware malfunction, or data transmission failure**. As established in Person A's data quality audit, 8.109% of all readings in METR-LA are exact zeros.

If zeros were included in error computation:
1. A model predicting a realistic speed of 55 mph during a sensor outage would be penalized with an artificial error of $|0 - 55| = 55$ mph.
2. In MAPE, division by zero would cause infinite or absurd percentage errors.

Therefore, following **Li et al. (ICLR 2018, DCRNN benchmark)**, all metrics are strictly masked over valid target entries ($\mathcal{Y}_{\text{valid}} = \{i \mid y_i > 0 \text{ and } y_i \neq \text{null}\}$):

$$\text{Masked MAE} = \frac{1}{|\mathcal{Y}_{\text{valid}}|} \sum_{i \in \mathcal{Y}_{\text{valid}}} |y_i - \hat{y}_i|$$

$$\text{Masked RMSE} = \sqrt{\frac{1}{|\mathcal{Y}_{\text{valid}}|} \sum_{i \in \mathcal{Y}_{\text{valid}}} (y_i - \hat{y}_i)^2}$$

$$\text{Masked MAPE} = \frac{100\%}{|\mathcal{Y}_{\text{valid}}|} \sum_{i \in \mathcal{Y}_{\text{valid}}} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

**Code Location:** `src/models/metrics.py`, functions `masked_mae`, `masked_rmse`, `masked_mape`.

---

### Q2: Why does XGBoost achieve a lower MAE than Random Forest, but Random Forest achieves a lower RMSE at 15m, 30m, and 60m?

**Answer:**
This divergence is directly rooted in their **optimization loss functions**:
- **XGBoost (Person C)** was configured with an **L1 absolute error loss** (`objective="reg:absoluteerror"`). In statistical estimation, minimizing an L1 loss directly models the **conditional median** of the target distribution ($P_{50}$), which mathematically optimizes for Mean Absolute Error (MAE).
- **Random Forest (Person D)** splits nodes using **variance reduction / mean squared error (L2 loss)**. Minimizing L2 loss estimates the **conditional mean** of the distribution and applies a quadratic penalty $(y - \hat{y})^2$ to large residual errors. Consequently, it aggressively curtails extreme outliers, resulting in lower RMSE at 15 min (5.764 vs 5.806), 30 min (7.003 vs 7.127), and 60 min (8.410 vs 8.714).

**Conclusion:** Neither model strictly dominates the other in the abstract. If the deployment objective prioritizes robust median accuracy with resistance to anomalies, XGBoost is superior; if avoiding large outlier misses is critical, Random Forest is superior.

---

### Q3: Why does plain XGBoost (32 features) lose to the Historical Average baseline at 60 minutes on the test set, and how did XGBoost + HA fix it?

**Answer:**
On the out-of-time test set at the 60-minute horizon:
- `historical_average` test MAE: **4.189**
- `xgboost_32` test MAE: **4.390** (worse by +0.201 MAE)

**The Root Cause:**
Person B's 32 shared features consist strictly of local autoregressive lags ($t, \dots, t-11$), rolling statistics, and temporal trigonometric encodings. Crucially, **they do not include sensor identity** (`sensor_id` was deliberately excluded to enforce a pure speed-and-time feature contract).
- At 5 and 15 minutes, local autoregressive momentum carries strong predictive power.
- At 60 minutes, the immediate lag correlation decays. Without sensor identity, the model cannot distinguish whether a given row corresponds to the fast I-210 corridor (baseline 63 mph) or the congested downtown I-110 corridor (baseline 52 mph).
- Furthermore, the test period (June 4 to June 27, 2012) exhibits higher volatility and outage frequency (87.9% valid targets vs 92.9% in train), increasing the test persistence error to 5.381 (vs 4.806 in validation).

**How XGBoost + HA Fixed It:**
Person C added 4 sensor-specific historical features (`ha_target`, `ha_origin`, `ha_delta`, `anomaly_now`) calculated leave-one-out from the sensor's own training history. These features anchor the booster to the sensor's typical speed at the target time, allowing `xgboost_ha` to achieve a test MAE of **3.688** (a 16.0% improvement over plain XGBoost and 31.5% over persistence).

---

### Q4: Why is traffic forecast error significantly worse during the evening peak than the morning peak?

**Answer:**
In our temporal slicing analysis (`src/visualization/spatial_mapping.py`, `reports/person_e_report.md` Table 3):
- At 15 minutes test:
  - Off-Peak MAE: **2.555**
  - Morning Peak (07:00–10:00) MAE: **3.469**
  - Evening Peak (16:00–19:00) MAE: **4.484** (75.5% higher than off-peak!)

**Physical Traffic Explanation:**
1. **Directional & Destination Dispersion:** Morning commute traffic is largely converging toward concentrated commercial centers (Downtown LA, Century City, Pasadena) with predictable, directional inbound queues. Evening commute traffic is characterized by simultaneous multi-directional dispersion from employment hubs back into suburbs across the entire basin.
2. **Cascading Spillbacks:** In the evening, secondary arterial bottlenecks spill backward onto freeway on-ramps and mainline corridors simultaneously, creating complex, non-linear gridlock wave patterns that cannot be predicted from a sensor's own isolated history.

---

### Q5: What is "The Gap" between Applied Machine Learning and Deep Learning, and why are Graph Attention Networks needed for Phase 2?

**Answer:**
The capstone abstract explicitly states: *"Identify the gap: Independent segment models cannot directly represent how congestion propagates through upstream and downstream links."*

**The Hydrodynamic Mechanism (LWR Shockwave Theory):**
Traffic behaves like a compressible fluid. When an incident occurs at a downstream sensor ($D$), traffic backs up, creating a backward-propagating kinematic shockwave ($w = \Delta q / \Delta k \approx -12$ mph). Upstream sensor $A$, located 4 miles upstream, is currently cruising at 62 mph.
- An independent AML model for Sensor $A$ observes only Sensor $A$'s own lags (all ~62 mph) and predicts 61 mph for $t+30$ min.
- But 30 minutes later, the downstream shockwave arrives at Sensor $A$, dropping its speed to 25 mph.
- **The Gap:** Local models suffer from **spatial boundary blindness**. They cannot observe adjacent upstream inflows or downstream queues.

**The Phase 2 Solution (Graph Attention Networks):**
GAT introduces the physical road graph adjacency matrix $\mathbf{A} \in \mathbb{R}^{207 \times 207}$ and computes dynamic spatial attention coefficients:

$$\alpha_{ij}^{(t)} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W} \mathbf{h}_i^{(t)} \parallel \mathbf{W} \mathbf{h}_j^{(t)}]))}{\sum_{k \in \mathcal{N}_i} \exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W} \mathbf{h}_i^{(t)} \parallel \mathbf{W} \mathbf{h}_k^{(t)}]))}$$

When downstream node $j$ slows down, the attention weight $\alpha_{ij}$ dynamically scales up, propagating the congestion signal to upstream node $i$ before the shockwave physically arrives.

---

### Q6: How was data leakage prevented across the train, validation, and test splits?

**Answer:**
Person A and Person B established strict chronological partition boundaries:
1. **Never Randomly Split:** Random train/test splitting on time series causes catastrophic look-ahead leakage. Splits are strictly chronological: Train (March 1 – May 23), Validation (May 23 – June 4), Test (June 4 – June 27).
2. **Boundary Origin Filtering:** An origin timestamp $t$ requires 12 lags of history ($t-11 \dots t$) and generates 4 targets ($t+1, t+3, t+6, t+12$). Any origin whose future targets would cross a split boundary into the next partition was **strictly discarded** by Person B.
3. **Training-Only Imputation:** Sensor medians used for forward-fill imputation were fitted exclusively on training timestamps and saved in `dataset_summary.json`. Test targets were raw and never imputed.
4. **Leave-One-Out Historical Features:** Person C's history features on training rows exclude the row's own target timestamp to prevent target leakage.

---

### Q7: What speed units and timezones are used in this project?

**Answer:**
As documented by Person A in `reports/data_quality_report.md` and adhered to across all reports:
- **Speed Units:** The upstream METR-LA HDF5 distribution does not explicitly state speed units in its metadata headers. Historically and empirically, Caltrans PeMS speeds are in **miles per hour (mph)**, but to maintain rigorous academic honesty, we report values as *"speed units (likely mph, unverified)"*.
- **Timezone:** Timestamps are recorded on the stored dataset clock (assumed UTC/local Pacific, unverified). Peak hours are strictly documented as *"stored clock hours [07:00, 10:00) and [16:00, 19:00)"*.

---

### Q8: Why must signal control be "simulated only"?

**Answer:**
As emphasized in the project instructions: *"Signal control should be simulated only; the project must not claim live traffic control."*
Real-world traffic signal preemption involves public safety, multi-agency jurisdiction (Caltrans / LADOT), and municipal hardware constraints. Our capstone focuses on predictive machine learning and traffic modeling. In Phase 2, any signal policy exploration will be performed strictly in microscopic traffic simulators like **SUMO (Simulation of Urban MObility)** or **CityFlow**, never claiming real-world physical actuation.

---

## 3. Person E Deliverables & File Directory Map

```
/Users/ranvendrapratapsingh/Documents/AML&DL_Project/
├── reports/
│   ├── person_e_plan.md                     ← Person E master plan & scope
│   ├── person_e_report.md                   ← Formal evaluation & benchmark report
│   ├── phase1_synthesis_report.md           ← Complete Phase 1 capstone synthesis
│   ├── phase2_graph_gap_analysis.md         ← Mathematical gap analysis for GAT/DL
│   └── person_e_viva_guide.md               ← This Viva Defense Guide
├── src/
│   ├── models/
│   │   ├── metrics.py                       ← Unified evaluation metrics (MAE, RMSE, MAPE)
│   │   └── evaluation.py                    ← Benchmark matrix & degradation engine
│   └── visualization/
│       ├── __init__.py                      ← Package exports
│       ├── spatial_mapping.py               ← 207-sensor geospatial corridor mapping
│       └── benchmark_plots.py               ← Matplotlib multi-metric visual suite
├── scripts/
│   └── run_person_e.py                      ← End-to-end execution runner CLI
├── tests/
│   └── test_person_e.py                     ← 15 unit tests covering all Person E logic
├── data/metadata/
│   ├── person_e_results.json                ← Structured serialized benchmark metrics
│   ├── corridor_performance.csv             ← Performance by 7 highway corridors
│   └── sensor_error_rankings.csv            ← Volatility ranking of all 207 sensors
├── figures/
│   ├── person_e_horizon_metrics_comparison.png
│   ├── person_e_peak_vs_offpeak_analysis.png
│   ├── person_e_congestion_map_morning_peak.png
│   ├── person_e_congestion_map_evening_peak.png
│   ├── person_e_spatial_error_corridors.png
│   ├── person_e_feature_importance_comparison.png
│   ├── person_e_rf_ablation_progression.png
│   └── person_e_graph_propagation_gap.png
└── notebooks/
    └── 03_model_evaluation_and_congestion_maps.ipynb
```

---

## 4. Commands to Demonstrate During Viva

```bash
# 1. Run all unit tests for Person E (15 tests passing in ~0.1s)
.venv/bin/python -m unittest tests/test_person_e.py -v

# 2. Run all unit tests across the entire project (54+ tests)
.venv/bin/python -m unittest discover -s tests -v

# 3. Re-run Person E pipeline and regenerate all figures/metadata (~2.5s)
.venv/bin/python scripts/run_person_e.py

# 4. Check git commit history (20+ verified commits by Person E)
git log --oneline -n 25
```
