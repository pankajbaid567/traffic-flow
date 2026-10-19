"""Build and execute Notebook 03: Model Evaluation, Congestion Maps & Gap Analysis."""
import json
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "03_model_evaluation_and_congestion_maps.ipynb"


def create_and_execute_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.5"
        }
    }

    cells = []

    # Cell 1: Title & Abstract
    cells.append(nbf.v4.new_markdown_cell(
"""# Notebook 03: Applied Machine Learning Evaluation, Spatial Congestion Mapping & Gap Analysis

**Author:** Person E (Lead Evaluator, Benchmark Synthesizer & Phase 1 Integration Lead)  
**Date:** 2026-10-19  
**Scope:** Phase 1 Applied Machine Learning Evaluation, Multi-Horizon Benchmark Synthesis, Highway Corridor Congestion Maps, Rush-Hour Asymmetry, and Phase 2 Gap Analysis.

### Objective
This notebook provides a complete visual and analytical synthesis of all Phase 1 traffic forecasting models on the METR-LA dataset. It compares naive baselines (Persistence, Historical Average) against tabular machine learning models (Random Forest, XGBoost 32, XGBoost + HA), explores spatial highway congestion across 207 sensors in Los Angeles County, and empirically illustrates why independent segment models fail at extended horizons, directly motivating Graph Attention Networks for Phase 2."""
    ))

    # Cell 2: Setup and Imports
    cells.append(nbf.v4.new_code_cell(
"""import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is on sys.path
ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.evaluation import (
    load_person_c_results,
    load_person_d_results,
    build_unified_test_benchmark,
    build_peak_offpeak_breakdown,
    build_rf_ablation_summary,
    compute_horizon_degradation,
)
from src.visualization.spatial_mapping import (
    load_sensor_geo_metadata,
    compute_corridor_statistics,
    plot_spatial_congestion_map,
    plot_corridor_error_distribution,
)
from src.visualization.benchmark_plots import (
    plot_horizon_comparison,
    plot_peak_vs_offpeak,
    plot_feature_importance_comparison,
    plot_rf_ablation_progression,
    plot_graph_gap_diagram,
)

print("Environment successfully initialized with Person E modules.")"""
    ))

    # Cell 3: Markdown Section 1
    cells.append(nbf.v4.new_markdown_cell(
"""## Section 1: Multi-Model Benchmark Synthesis across Forecast Horizons

We synthesize test set evaluations across four distinct forecasting horizons:
- **+5 min (t+1)**: Immediate short-term autoregressive inertia
- **+15 min (t+3)**: Tactical short-term traffic flow
- **+30 min (t+6)**: Operational network queue formation
- **+60 min (t+12)**: Strategic medium-term travel time forecasting

All evaluations strictly exclude zero/missing readings (`target_valid_<h> == True`), matching Li et al. (ICLR 2018, DCRNN benchmark)."""
    ))

    # Cell 4: Load and Display Unified Benchmark
    cells.append(nbf.v4.new_code_cell(
"""df_bench = build_unified_test_benchmark()
piv_mae = df_bench.pivot(index="model", columns="horizon", values="mae")[["5min", "15min", "30min", "60min"]]
piv_rmse = df_bench.pivot(index="model", columns="horizon", values="rmse")[["5min", "15min", "30min", "60min"]]
piv_mape = df_bench.pivot(index="model", columns="horizon", values="mape")[["5min", "15min", "30min", "60min"]]

print("=== TEST SET MAE (Speed Units / mph) ===")
display(piv_mae)

print("\\n=== TEST SET RMSE (Speed Units / mph) ===")
display(piv_rmse)

print("\\n=== TEST SET MAPE (%) ===")
display(piv_mape)"""
    ))

    # Cell 5: Markdown Analysis for Section 1
    cells.append(nbf.v4.new_markdown_cell(
"""### Analysis of Multi-Model Performance:
1. **Applied ML Consistently Outperforms Persistence:** Random Forest and XGBoost achieve substantial error reductions over persistence across all horizons (e.g. at 15 min: XGBoost + HA achieves 2.844 MAE vs 3.499 persistence, an 18.7% improvement).
2. **Loss Function Duality:**
   - **XGBoost (32 shared features)** dominates **Random Forest (Exp D)** on MAE (2.970 vs 3.168 at 15m) and MAPE (7.92% vs 8.56%) because it was trained under an L1 absolute error loss.
   - **Random Forest** achieves lower RMSE at 15m (5.764 vs 5.806), 30m (7.003 vs 7.127), and 60m (8.410 vs 8.714) because it minimizes mean squared error (L2 loss), penalizing residual outliers.
3. **The 60-Minute Inflection Point:** At 60 minutes, plain XGBoost (32 features) fails to beat the Historical Average (4.390 vs 4.189) because features lack sensor identity. XGBoost + HA resolves this, achieving **3.688 MAE**."""
    ))

    # Cell 6: Visualizing Multi-Horizon Performance
    cells.append(nbf.v4.new_code_cell(
"""# Plot multi-metric horizon curves
plot_horizon_comparison(df_bench)
img = plt.imread(ROOT / "figures" / "person_e_horizon_metrics_comparison.png")
fig, ax = plt.subplots(figsize=(16, 5), dpi=150)
ax.imshow(img)
ax.axis("off")
plt.title("Multi-Horizon Benchmark Degradation (MAE, RMSE, MAPE)", fontsize=13, fontweight="bold", pad=12)
plt.show()"""
    ))

    # Cell 7: Markdown Section 2
    cells.append(nbf.v4.new_markdown_cell(
"""## Section 2: Rush-Hour Asymmetry & Peak vs Off-Peak Regimes

Traffic flow displays severe diurnal non-stationarity. We evaluate model performance across four temporal regimes:
- **Morning Peak:** Weekdays 07:00–10:00 (121,561 test samples at 5m)
- **Evening Peak:** Weekdays 16:00–19:00 (128,805 test samples at 5m)
- **Combined Peak:** Weekdays 07:00–10:00 and 16:00–19:00 (250,366 test samples at 5m)
- **Off-Peak:** All other timestamps, including weekdays and weekends (994,686 test samples at 5m)"""
    ))

    # Cell 8: Code for Peak vs Off-Peak
    cells.append(nbf.v4.new_code_cell(
"""df_slices = build_peak_offpeak_breakdown()
h15_slices = df_slices[df_slices["horizon"] == "15min"][
    ["period", "rows", "persistence_mae", "historical_average_mae", "xgboost_mae", "xgboost_ha_mae"]
].set_index("period").reindex(["all", "off_peak", "peak", "morning_peak", "evening_peak"])

print("=== 15-MINUTE FORECAST MAE BY TEMPORAL REGIME ===")
display(h15_slices)

# Plot rush-hour analysis
plot_peak_vs_offpeak(df_slices)
img_peak = plt.imread(ROOT / "figures" / "person_e_peak_vs_offpeak_analysis.png")
fig, ax = plt.subplots(figsize=(14, 5.5), dpi=150)
ax.imshow(img_peak)
ax.axis("off")
plt.title("Rush-Hour Error Escalation & Morning vs Evening Peak Disparity", fontsize=13, fontweight="bold", pad=12)
plt.show()"""
    ))

    # Cell 9: Markdown Analysis for Section 2
    cells.append(nbf.v4.new_markdown_cell(
"""### Analysis of Rush-Hour Dynamics:
1. **Severe Peak Error Spike:** Forecasting errors jump dramatically during peak hours. At 15 min, off-peak MAE is **2.555**, whereas combined peak MAE is **3.992** (+56.2% error).
2. **Evening Rush Hour is Harder than Morning:** Evening peak error reaches **4.484 MAE**, which is 29.2% higher than morning peak error (3.469 MAE) and 75.5% higher than off-peak. Morning traffic exhibits structured directional convergence towards downtown LA, whereas evening traffic disperses simultaneously in multiple directions across suburban bottlenecks.
3. **Catastrophic Failure of Historical Average in Peak Periods:** The Historical Average baseline incurs a massive **7.394 MAE** during the evening peak, demonstrating that static time-of-week averages fail when traffic collapses into non-recurrent congestion."""
    ))

    # Cell 10: Markdown Section 3
    cells.append(nbf.v4.new_markdown_cell(
"""## Section 3: Spatial Highway Network Mapping & Corridor Analytics

Using coordinates from `data/metadata/sensors.csv` and diurnal hourly profiles from `data/metadata/hourly_mean_by_sensor.csv`, we map network speeds and error vulnerability across 7 major Los Angeles freeway corridors:
- **I-405 (San Diego Freeway / Sepulveda Pass)**
- **I-10 (Santa Monica Freeway)**
- **US-101 (Hollywood / Ventura Freeway)**
- **I-110 (Harbor Freeway / Downtown LA Hub)**
- **I-5 / I-710 Gateway**
- **I-5 / SR-134 Connectors**
- **I-210 (Foothill Freeway)**"""
    ))

    # Cell 11: Code for Corridor Analysis
    cells.append(nbf.v4.new_code_cell(
"""df_sensors = load_sensor_geo_metadata()
corridor_stats = compute_corridor_statistics(df_sensors)

print("=== CORRIDOR-LEVEL CONGESTION & SPEED DROPS ===")
display(corridor_stats)

# Plot Spatial Congestion Maps for Morning and Evening Peaks
plot_spatial_congestion_map(df_sensors, peak_type="morning")
plot_spatial_congestion_map(df_sensors, peak_type="evening")
plot_corridor_error_distribution(df_sensors)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
img_morn = plt.imread(ROOT / "figures" / "person_e_congestion_map_morning_peak.png")
img_eve = plt.imread(ROOT / "figures" / "person_e_congestion_map_evening_peak.png")
ax1.imshow(img_morn)
ax1.axis("off")
ax1.set_title("Morning Peak Congestion Map (07:00–10:00)", fontsize=11, fontweight="bold")
ax2.imshow(img_eve)
ax2.axis("off")
ax2.set_title("Evening Peak Congestion Map (16:00–19:00)", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.show()"""
    ))

    # Cell 12: Markdown Analysis for Section 3
    cells.append(nbf.v4.new_markdown_cell(
"""### Analysis of Spatial Highway Corridors:
1. **Critical Bottleneck Corridors:**
   - **I-405 (San Diego Freeway):** Suffers the highest speed collapse in Los Angeles, dropping **32.20%** during the evening rush hour from 58.76 mph to 39.84 mph.
   - **I-10 (Santa Monica Freeway):** Experiences a **31.26%** evening speed collapse down to 38.10 mph average.
   - **I-110 (Harbor Freeway / Downtown LA):** Exhibits the highest speed volatility in the entire network ($\sigma = 16.12$ mph), driven by complex weaving maneuvers at the I-10 / US-101 / I-110 interchange.
2. **Resilient Corridors:**
   - **I-210 (Foothill Freeway):** Maintains free-flow speeds exceeding 54.10 mph with the lowest volatility ($\sigma = 8.94$ mph) and only a 13.92% evening drop."""
    ))

    # Cell 13: Markdown Section 4
    cells.append(nbf.v4.new_markdown_cell(
"""## Section 4: Feature Importance Meta-Analysis & Person D Ablations

We examine which feature categories provide predictive power as the horizon scales from +5 min to +60 min, comparing Random Forest Gini impurity with XGBoost normalized gain."""
    ))

    # Cell 14: Code for Feature Importance & Ablations
    cells.append(nbf.v4.new_code_cell(
"""c_data = load_person_c_results()
d_data = load_person_d_results()

# Plot Feature Importance Shift
plot_feature_importance_comparison(c_data, d_data)
img_feat = plt.imread(ROOT / "figures" / "person_e_feature_importance_comparison.png")
fig, ax = plt.subplots(figsize=(15, 9), dpi=150)
ax.imshow(img_feat)
ax.axis("off")
plt.title("Feature Importance Shift: Immediate Lag Dominance (5m) vs Diurnal Cycles (60m)", fontsize=13, fontweight="bold", pad=12)
plt.show()

# Plot Random Forest Ablation Progression
plot_rf_ablation_progression(d_data)
img_abl = plt.imread(ROOT / "figures" / "person_e_rf_ablation_progression.png")
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
ax.imshow(img_abl)
ax.axis("off")
plt.title("Random Forest Ablation Progression across 4 Tiers", fontsize=12, fontweight="bold", pad=12)
plt.show()"""
    ))

    # Cell 15: Markdown Analysis for Section 4
    cells.append(nbf.v4.new_markdown_cell(
"""### Analysis of Feature Dynamics:
1. **Short Horizons (+5 min):** Dominated by `speed_lag_0` (>85% Gini importance in RF, >23% gain in XGBoost). Autoregressive inertia accounts for almost all predictive signal.
2. **Long Horizons (+60 min):** Immediate lag influence drops to 57% in RF and 8.6% in XGBoost. Rolling statistics (`rolling_mean_15min`) and cyclical diurnal features (`hour_cos`, `hour_sin`, `is_weekend`) surge in influence.
3. **Ablation Evidence:** Person D's ablation study proves that adding rolling and temporal features provides negligible benefit at 5 min (+1.87% gain over lags only), but yields a **+9.79% error reduction (0.527 MAE improvement)** at 60 minutes."""
    ))

    # Cell 16: Markdown Section 5
    cells.append(nbf.v4.new_markdown_cell(
"""## Section 5: The Gap Analysis — The Bridge to Phase 2 Graph Attention Networks

We present the theoretical and physical explanation of **The Gap**: why independent segment models reach an insurmountable performance barrier, and how Graph Attention Networks (GAT) will bridge this gap in Phase 2."""
    ))

    # Cell 17: Code for Gap Diagram
    cells.append(nbf.v4.new_code_cell(
"""plot_graph_gap_diagram()
img_gap = plt.imread(ROOT / "figures" / "person_e_graph_propagation_gap.png")
fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
ax.imshow(img_gap)
ax.axis("off")
plt.title("The AML-to-DL Gap: Backward Shockwave Spillback vs Spatial Boundary Blindness", fontsize=13, fontweight="bold", pad=12)
plt.show()"""
    ))

    # Cell 18: Final Summary Markdown Cell following notebook-guidance
    cells.append(nbf.v4.new_markdown_cell(
"""## Summary

### Q&A
- **Q: Did applied machine learning models outperform naive baselines?**  
  **A:** Yes. Random Forest (MAE 2.453 at 5m, 4.855 at 60m) and XGBoost (MAE 2.362 at 5m, 4.390 at 60m) significantly outperform Persistence (2.814 at 5m, 5.381 at 60m) across all four horizons. XGBoost + HA achieves the lowest overall test MAE (2.312 at 5m, 3.688 at 60m), delivering a 31.5% improvement over persistence at 60 minutes.
- **Q: Why did plain XGBoost lose to Historical Average at 60 minutes on test?**  
  **A:** Because Person B's 32 features are strictly local speed/time features without sensor identity. Beyond 30 minutes, autoregressive lag correlation decays, and the model cannot determine a sensor's baseline level during the more volatile test period. XGBoost + HA resolves this by adding 4 leave-one-out historical profile features.
- **Q: How does performance vary between peak and off-peak periods?**  
  **A:** Forecast error spikes during rush hours, with evening peak periods (16:00–19:00) exhibiting 75.5% higher MAE than off-peak hours (4.484 vs 2.555 at 15m), caused by simultaneous multi-directional bottleneck spillbacks across the Los Angeles basin.

### Data Analysis Key Findings
- **Loss Function Trade-off:** XGBoost (L1 loss) minimizes MAE and MAPE across all horizons (2.970 MAE, 7.92% MAPE at 15m), whereas Random Forest (L2 loss) minimizes RMSE (5.764 RMSE at 15m vs 5.806 for XGBoost).
- **Corridor Bottlenecks:** The I-405 Sepulveda Pass (32.20% speed drop) and I-10 Santa Monica Freeway (31.26% drop) suffer the most severe evening peak congestion, while the I-110 Downtown interchange exhibits the highest variance ($\sigma = 16.12$ mph).
- **Temporal Feature Value:** Random Forest ablation shows that temporal and rolling features provide only 1.87% improvement at 5 min, but contribute **9.79% error reduction (0.527 MAE)** at 60 min.

### Insights or Next Steps
- **The Phase 2 Bridge:** Independent segment models suffer from spatial boundary blindness; they cannot anticipate backward-propagating traffic shockwaves ($w \approx -12$ mph) until the queue physically reaches the local detector.
- **Next Steps:** In Phase 2, deploy Graph Attention Networks (GAT) and Diffusion Convolutional models (DCRNN) over the 207-node road graph adjacency matrix to model spatial shockwave propagation, coupled with optional simulated signal policy experiments in SUMO/CityFlow."""
    ))

    nb.cells = cells

    print(f"Executing {len(cells)} cells in {NOTEBOOK_PATH.name}...")
    client = NotebookClient(nb, timeout=300, kernel_name="python3")
    client.execute()

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"✓ Notebook successfully executed and saved to {NOTEBOOK_PATH}")


if __name__ == "__main__":
    create_and_execute_notebook()
