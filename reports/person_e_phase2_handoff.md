# Handoff Notes: Person E → Team (Phase 1 Completion & Phase 2 Kickoff)

Hey Team (A, B, C, D),

Person E's work for **Phase 1 (Applied Machine Learning Benchmark, Spatial Congestion Mapping & Gap Analysis)** is complete and pushed to GitHub. 

All models across naive baselines (Persistence, Historical Average) and applied machine learning (Random Forest, XGBoost 32, XGBoost + HA) are evaluated under unified, leak-free standards. The benchmark tables, spatial highway congestion maps, and storytelling Notebook 03 are ready.

This note summarizes **what I built**, **where everything is**, and **how we transition into Phase 2 (Deep Graph Learning & Traffic Simulation)**.

---

## 1. What Person E Delivered (Phase 1 Summary)

| Deliverable | Location | Description |
|---|---|---|
| **Unified Evaluation Suite** | `src/models/metrics.py` | Masked MAE, RMSE, and MAPE strictly filtering `target_valid_<h> == True` (DCRNN benchmark standard). |
| **Cross-Model Benchmark Engine** | `src/models/evaluation.py` | Benchmark matrix synthesizing Person C & D results across 4 horizons (5m, 15m, 30m, 60m). |
| **Spatial Highway Mapping** | `src/visualization/spatial_mapping.py` | 2D network mapping of 207 sensors and speed drops across 7 freeway corridors (I-405, I-10, US-101, I-110, I-210). |
| **Visual Analytics Suite** | `src/visualization/benchmark_plots.py` | 8 high-resolution 300 DPI figures (`figures/person_e_*.png`). |
| **Automated Runner** | `scripts/run_person_e.py` | End-to-end CLI script regenerating all tables, metadata, and figures in ~2.5s. |
| **Storytelling Notebook 03** | `notebooks/03_model_evaluation_and_congestion_maps.ipynb` | Fully executed 18-cell notebook with tables, plots, and markdown analyses. |
| **Comprehensive Reports** | `reports/` | [person_e_report.md](person_e_report.md), [phase1_synthesis_report.md](phase1_synthesis_report.md), [phase2_graph_gap_analysis.md](phase2_graph_gap_analysis.md), [person_e_viva_guide.md](person_e_viva_guide.md). |
| **Unit Test Suite** | `tests/test_person_e.py` | 15 passing unit tests verifying metrics, spatial mapping, and synthesis logic. |

---

## 2. Headline Phase 1 Numbers to Carry into Phase 2

All metrics are evaluated on the out-of-time test partition (June 4 – June 27, 2012, ~1.245M valid predictions per horizon):

| Horizon | Persistence | Historical Average | Random Forest (32) | XGBoost (32) | XGBoost + HA (36) | Target to Beat in Phase 2 |
|---|---:|---:|---:|---:|---:|---:|
| **5 min** | 2.814 | 4.188 | 2.453 | 2.362 | **2.312** | **< 2.31 MAE** |
| **15 min** | 3.499 | 4.188 | 3.168 *(RMSE 5.764)* | 2.970 | **2.844** | **< 2.84 MAE** |
| **30 min** | 4.206 | 4.188 | 3.871 *(RMSE 7.003)* | 3.563 | **3.275** | **< 3.27 MAE** |
| **60 min** | 5.381 | 4.189 | 4.855 *(RMSE 8.410)* | 4.390 | **3.688** | **< 3.68 MAE** |

### Key Benchmark Discoveries:
1. **The L1 / L2 Trade-Off:** XGBoost wins on MAE and MAPE (L1 loss), while Random Forest wins on RMSE at 15m, 30m, and 60m (L2 loss). Both must be reported when comparing against Deep Learning.
2. **The 60-Minute Local Model Ceiling:** Local models without graph awareness fail to capture shockwaves propagating from downstream bottlenecks. XGBoost 32 loses to Historical Average at 60 min (4.390 vs 4.189); XGBoost + HA requires manual training history features to reach 3.688.
3. **The Peak Penalty:** Evening rush hour (16:00–19:00) error is **75.5% higher** than off-peak error at 15 min (4.484 vs 2.555).

---

## 3. Phase 2 Team Responsibilities (Who Does What Next)

Phase 2 shifts the project from isolated segment-level AML models to **network-aware Deep Graph Learning & Traffic Simulation**:

```
                                  PHASE 2 WORKFLOW & ROLES
                                  
  ┌────────────────────────────────────────────────────────────────────────────────────────┐
  │ PERSON A / B: Graph Preprocessing & Spatio-Temporal Tensor Loader                      │
  │ • Load directed adjacency matrix A ∈ ℝ^{207 × 207} from data/raw/adj_mx.pkl            │
  │ • Compute normalized graph Laplacians (forward & backward random walk matrices)        │
  │ • Build spatio-temporal DataLoader: X ∈ ℝ^{B × T_in × 207 × F}, Y ∈ ℝ^{B × T_out × 207}│
  │ • Preserve Person A's exact chronological split (70% train, 10% val, 20% test)         │
  └──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                             │
  ┌──────────────────────────────────────────▼─────────────────────────────────────────────┐
  │ PERSON C / D: Deep Graph Model Architectures (PyTorch / PyTorch Geometric)             │
  │ • Person C: Implement Graph Attention Network (GAT) with dynamic spatial attention α_ij│
  │ • Person D: Implement Diffusion Convolutional Recurrent Neural Network (DCRNN)         │
  │ • Train multi-step spatio-temporal models using scheduled sampling / teacher forcing   │
  │ • Save trained model checkpoints in models/phase2/                                     │
  └─────────────────────┬───────────────────────────────────────────┬──────────────────────┘
                        │                                           │
  ┌─────────────────────▼────────────────────┐┌─────────────────────▼──────────────────────┐
  │ OPTIONAL / SIMULATION: SUMO or CityFlow  ││ PERSON E: Fair Comparison & Synthesis     │
  │ • Microscopic traffic network simulation ││ • Like-for-like comparison: GAT vs AML     │
  │ • Test simulated signal timing policies  ││ • Test if GAT resolves shockwave gap       │
  │ • RULE: Designate as SIMULATED ONLY      ││ • Multi-horizon comparison & congestion map│
  │ • Do not claim live traffic actuation    ││ • Phase 2 Final Report & Presentation      │
  └──────────────────────────────────────────┘└────────────────────────────────────────────┘
```

### Detailed Breakdown:
- **Person A / Person B (Graph Data Engineering):**
  - Load the sensor graph adjacency from `data/raw/adj_mx.pkl`.
  - Prepare the 4D input tensors $(B, T_{in}, N, F)$ where $T_{in}=12$ (1 hour), $N=207$ sensors, and $F=2$ (normalized speed and time-of-day).
  - Retain the exact same chronological split boundaries as Phase 1 (June 4, 2012 for test start).
- **Person C (Graph Attention Networks - GAT):**
  - Implement GAT using PyTorch Geometric (`torch_geometric.nn.GATConv`).
  - Use multi-head attention to dynamically capture downstream shockwaves and upstream volume demand.
- **Person D (Diffusion Graph Neural Networks - DCRNN):**
  - Implement bidirectional graph random walks ($\mathbf{D}_O^{-1} \mathbf{A}$ for vehicular flow, $\mathbf{D}_I^{-1} \mathbf{A}^T$ for shockwave spillback).
  - Train sequence-to-sequence recurrent graph convolution.
- **Optional Simulation Subteam (SUMO / CityFlow):**
  - Simulate a closed traffic network corridor (e.g. I-405 or Downtown grid).
  - Test adaptive signal policy adjustments.
  - **Crucial Rule:** As stated in the capstone abstract, signal control must strictly be described as *simulated only*; never claim live traffic actuation.
- **Person E (Lead Evaluation & Synthesis):**
  - Fair comparison: evaluate GAT and DCRNN against Phase 1 models on the exact same test horizons (5, 15, 30, 60 min).
  - Verify whether GAT successfully eliminates the 60-minute inflection point and reduces evening peak error.
  - Final Phase 2 synthesis report and presentation slides.

---

## 4. Ground Rules & Contracts for Phase 2

1. **Keep the Common Metric Suite:** Use `src/models/metrics.py` (`compute_metrics`). Exclude zeros from target scoring (`target_valid_<h> == True`).
2. **Keep the Exact Chronological Split:** Do not shuffle or alter split timestamps. All Phase 2 models must be scored on the identical test period (June 4 – June 27, 2012).
3. **Like-for-Like Reporting:** Report all three metrics (MAE, RMSE, MAPE) across all four horizons (5, 15, 30, 60 min) and split by peak vs off-peak.
4. **Units:** Speeds remain in source units (likely mph, unverified). Write "speed units (likely mph)" in all reports.

Phase 1 code, data, and reports are fully documented in `README.md`. Reach out if anything is unclear!

— Person E
