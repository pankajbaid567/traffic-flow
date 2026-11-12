# Phase 2 Fact Sheet: Evaluation Harness, Statistical Inference & Spatial Error Diagnostics

**Owner:** Member 4 (`phase2/m4-evaluation`)  
**Package:** `src/phase2/evaluation/`, `src/phase2/viz/`  
**Runner:** `scripts/phase2_eval_all.py`  
**Tests:** `tests/test_phase2_evaluation.py`  
**Outputs:** `data/metadata/phase2/`, `figures/phase2/`  

---

## 1. Evaluation Protocol & Rigorous Fair-Comparison Principles

The Phase 2 evaluation pipeline provides an independent, leak-free, mathematically aligned benchmark harness comparing deep graph models (**GAT**, **DCRNN**), deep non-graph controls (**`gru_local`**, **`gat_identity`**, **`dcrnn_identity`**, **`gat_shuffled`**), and Phase 1 competitive tree baselines (**`random_forest_32`**, **`xgboost_32`**, **`xgboost_ha`**).

### Core Invariants Enforced:
1. **Identical Sample Origins & Sensors**:
   - Validation partition: strictly 3,416 origins ($707,112$ rows across $N=207$ sensors).
   - Test partition: strictly 6,844 origins ($1,416,708$ rows across $N=207$ sensors).
   - Sensor ordering strictly locked to canonical `data/metadata/sensors.csv`.
2. **Valid Target Masking**:
   - Zero, nonfinite, and negative readings are treated as unavailable sensors, never as physical zero speed.
   - Evaluated strictly on positive, finite observed targets:
     - 5 min: $1,245,052$ valid test targets ($87.88\%$)
     - 15 min: $1,245,040$ valid test targets
     - 30 min: $1,245,022$ valid test targets
     - 60 min: $1,244,986$ valid test targets
3. **Pooled Metric Aggregation**:
   - Regional and overall MAE/RMSE/MAPE are computed as pooled sums of errors divided by total sample counts:
     $$\text{MAE} = \frac{\sum_i |y_i - \hat{y}_i|}{\sum_i 1}, \quad \text{RMSE} = \sqrt{\frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i 1}}, \quad \text{MAPE} = \frac{100}{n}\sum_i \frac{|y_i - \hat{y}_i|}{\max(y_i, 10^{-5})}$$
   - Unweighted averaging of per-sensor MAEs is strictly disallowed to prevent distortion from sensors with fewer valid observations.

---

## 2. Statistical Uncertainty Quantification (Paired Moving-Block Bootstrap)

To verify whether forecast improvements are statistically significant and not artifacts of sample variation:
- **Resampling Scheme**: Circular moving-block bootstrap over stored calendar days.
- **Block Size**: 2 days (capturing multi-day temporal autocorrelation and weather/incident persistence).
- **Repetitions**: 2,000 resamples (fixed seed 42).
- **Multi-Seed Aggregation**: For models evaluated over multiple random seeds (e.g. seeds 42, 43, 44), daily absolute error sums are averaged across seeds prior to resampling. This computes the exact expected seed-averaged MAE difference.
- **Hypothesis Verdict Rules**:
  - **`supported`**: The 95% bootstrap confidence interval of the error reduction ($\text{MAE}_{\text{baseline}} - \text{MAE}_{\text{candidate}}$) lies entirely above zero ($CI_{\text{low}} > 0$).
  - **`contradicted`**: The 95% bootstrap confidence interval lies entirely below zero ($CI_{\text{high}} < 0$).
  - **`inconclusive`**: The 95% bootstrap confidence interval straddles zero ($CI_{\text{low}} \le 0 \le CI_{\text{high}}$).
  - **`not_evaluated`**: One or both candidate models are missing from the prediction directory.

---

## 3. Formal Hypotheses Specification

| Hypothesis | Theoretical Focus | Target Horizons | Tested Comparison | Code Verdict Rule |
|---|---|---|---|---|
| **H1** | Spatial graph value over recurrent temporal baseline | 30m, 60m | `gru_local` vs. `gat` | Supported if both 30m and 60m intervals $> 0$ |
| **H2** | Road network topology value vs. isolated & wrong graph | 60m | `gat_identity` vs. `gat`<br>`gat_identity` vs. `gat_shuffled` | Supported if `gat` beats `gat_identity` and `gat_shuffled` does not beat `gat_identity` |
| **H3** | Spatial value during non-recurrent congestion | 60m | `gat_identity` vs. `gat` (Peak vs. Off-Peak) | Supported if peak MAE reduction exceeds off-peak reduction |
| **H4** | Long-horizon spatial propagation benefit | 5m, 15m, 30m, 60m | `gru_local` vs. `gat` | Supported if gain strictly increases monotonically with horizon |
| **H5** | Connectivity & congestion correlation | 60m | Descriptive correlation | Descriptive only: correlation with sensor degree and congestion onsets |
| **H0** | Deep neural vs. tree baseline parity | 60m | `xgboost_ha` vs. {`gat`, `dcrnn`} | Supported if 95% CI contains 0 |

---

## 4. Congestion Onset Proxy Definition

Because METR-LA provides loop detector speed telemetry without vehicle counts or incident logs:
1. **Proxy Speed Threshold**:
   $$\text{Threshold}_s = 0.6 \times \text{FreeSpeed}_s$$
   where $\text{FreeSpeed}_s$ is the 85th percentile speed of sensor $s$ on valid training data.
2. **True Congestion Onset**:
   A sensor is uncongested at origin time $t$ ($y_{s, t} \ge \text{Threshold}_s$) and transitions to congested at horizon $t+h$ ($y_{s, t+h} < \text{Threshold}_s$).
3. **Onset Metrics Recorded** (`data/metadata/phase2/events.csv`):
   - Precision of congested classification.
   - Recall of congested classification.
   - Recall of true congestion onset events.

---

## 5. Artifacts and Output Files

### Metadata Tables (`data/metadata/phase2/`):
- `metrics.csv`: Horizon, model, seed, period, $n$, MAE, RMSE, MAPE.
- `metrics_summary.csv`: Aggregated across seeds with mean and standard deviation.
- `bootstrap.csv`: Full paired circular moving-block bootstrap intervals.
- `hypotheses.csv`: Programmatic verdicts for H0–H5.
- `events.csv`: Precision and recall of proxy congestion onset events.
- `sensor_metrics.csv` & `region_metrics.csv`: Stratified spatial slices across 7 coordinate regions.

### Publication Figures (`figures/phase2/`, 300 DPI):
- `fig_horizon_mae.png`: Multi-horizon forecast error trajectories.
- `fig_peak_offpeak.png`: Rush hour vs. off-peak forecast degradation.
- `fig_graph_ablation.png`: GAT vs. Identity vs. Shuffled network topology.
- `fig_seed_spread.png`: Seed-to-seed variance across training initializations.
- `fig_error_by_degree.png`: Forecast error vs. node degree in the highway graph.

---

## 6. Honest Limitations & Scientific Boundaries

1. **Unverified Speed Units & Timezone**: Speeds are reported in source units (likely mph). Time stamps are stored clock without daylight saving or UTC confirmation.
2. **Coordinate Bins, Not Freeway Names**: Geographic regions are deterministic spatial coordinate bounding boxes; they do not represent administrative freeway IDs or lane configurations.
3. **Control Architecture Hyperparameters**: Controls (`gat_identity`, `dcrnn_identity`, `gat_shuffled`) inherit hyperparameter configurations optimized for the real graph (`params: from_search`). This is a deliberate, clean ablation design that may slightly underestimate control performance.
4. **No Historical Average Channel for Deep Models**: Deep models take raw past speeds and time encodings (sin/cos of hour and day-of-week), but do not receive explicit Historical Average feature lookup columns. Deep vs. `xgboost_ha` comparisons therefore reflect feature differences in addition to model architectures.
5. **Single Chronological Partition**: All evaluations reflect a single 4-month continuous period on METR-LA (March–June 2012). Intervals quantify uncertainty over this specific held-out window and do not generalize to other seasons or metropolitan networks.

