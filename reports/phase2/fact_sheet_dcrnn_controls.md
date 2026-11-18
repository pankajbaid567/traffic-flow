# Phase 2 Fact Sheet: DCRNN & Non-Graph Deep Controls

**Owner:** Member 3 (`phase2/m3-dcrnn-controls`)  
**Architecture files:** `src/phase2/models/dcrnn.py`, `src/phase2/models/controls.py`  
**Run configurations:** `configs/phase2/runs_dcrnn.yaml`, `configs/phase2/runs_gru_local.yaml`  
**Tests:** `tests/test_phase2_dcrnn.py`, `tests/test_phase2_controls.py`  

---

## 1. Architectures & Design Choices

### A. Non-Graph Baseline: `gru_local` (`src/phase2/models/controls.py`)
- **Philosophy**: Establishes the performance of an equally expressive recurrent model that does **not** see road network neighbours.
- **Structure**:
  - Shared weights across all $N=207$ sensors.
  - Standard GRU processes the 12 past steps per sensor independently: $x \in \mathbb{R}^{B \times 12 \times N \times 7} \to \text{GRU} \to H \in \mathbb{R}^{B \times N \times \text{hidden}}$.
  - Multi-layer perceptron (MLP) head predicts residual $\Delta y$ for 4 horizons relative to $x[:, -1, :, 0]$.
  - Zero spatial mixing: Sensor $i$'s output is strictly independent of sensor $j$'s input ($i \neq j$), verified by unit tests.

### B. Graph Model: `dcrnn` (`src/phase2/models/dcrnn.py`)
- **Philosophy**: Spatiotemporal model incorporating bidirectional graph random walks via diffusion convolutions.
- **Diffusion Convolutions (DiffusionConv)**:
  - Supports calculated internally from directed graph $A$ (`graph: given`):
    - Forward random walk: $P_f = D_O^{-1} A$
    - Reverse random walk: $P_r = D_I^{-1} A^T$
    - Safe division handles isolated nodes.
    - Chebyshev/diffusion polynomial expansion up to $K$ hops: $[I, P_f, P_r, \dots, P_f^K, P_r^K]$ ($2K + 1$ matrices).
  - When $K=0$ or `supports: identity`, the layer reduces to an ordinary linear projection without neighbour aggregation.
- **Encoder-Decoder DCGRU**:
  - **Encoder**: 12 recurrent steps of DCGRU over history $t-11 \dots t$.
  - **Decoder**: Autoregressively rolls out 12 future steps ($t+1 \dots t+12$). The forecasts at step indices `[0, 2, 5, 11]` (+5, +15, +30, +60 minutes) are extracted to match the project's exact 4-horizon evaluation contract.
  - Residual formulation: Predictions are anchored to the last known input speed $x[:, -1, :, 0]$.

---

## 2. Search Candidates & Exact Parameter Counts

### `gru_local` Candidates (`runs_gru_local.yaml`):
| Candidate | Hidden | Layers | Dropout | Parameter Count |
|---|---|---|---|---|
| Candidate 0 | 32 | 1 | 0.0 | **5,124** |
| Candidate 1 | 64 | 1 | 0.0 | **18,436** |
| Candidate 2 | 32 | 2 | 0.1 | **11,460** |
| Candidate 3 | 64 | 2 | 0.2 | **43,396** |

### `dcrnn` Candidates (`runs_dcrnn.yaml`):
| Candidate | Hidden | Layers | $K$ (Hops) | Dropout | Parameter Count |
|---|---|---|---|---|---|
| Candidate 0 | 32 | 1 | 1 | 0.0 | **20,961** |
| Candidate 1 | 32 | 1 | 2 | 0.0 | **34,785** |
| Candidate 2 | 64 | 1 | 2 | 0.1 | **131,009** |
| Candidate 3 | 32 | 2 | 2 | 0.1 | **96,417** |

---

## 3. Sanity Range, Baseline Targets & Leakage Red Flags

### Phase 1 Baselines to Beat / Match:
| Horizon | Persistence | Historical Average | XGBoost 32 | XGBoost + HA 36 |
|---|---|---|---|---|
| 5 min | 2.814 | 4.188 | 2.362 | **2.312** |
| 15 min | 3.499 | 4.188 | 2.970 | **2.844** |
| 30 min | 4.206 | 4.188 | 3.564 | **3.275** |
| 60 min | 5.380 | 4.189 | 4.393 | **3.688** |

### Reference Literature (Sanity Check Only):
- The original DCRNN paper on METR-LA (Li et al. 2018) reported MAE:
  - 5 min: **2.18**
  - 15 min: **2.67**
  - 30 min: **3.08**
  - 60 min: **3.56**
- *Note*: The original DCRNN code used a different sample definition (stride 1 over all minutes without Phase 1's explicit origin exclusion rules).
- > [!WARNING]
  > **Leakage Red Flag**: Any 5-minute test MAE substantially below **~2.00** would strongly indicate data leakage (e.g. leaking future ground truth into causal imputation or targets).

---

## 4. Compute & Runtime Benchmarks

### Measured Local Speed (Apple Silicon M3, 8 GB RAM):
- **`gru_local` Smoke Test**: ~0.05–0.10s per step on M3 MPS.
- **`dcrnn` Smoke Test**: ~0.27–0.95s per step on M3 MPS.
- 5% Data Subset on M3 MPS:
  - `gru_local`: ~0.6s per epoch.
  - `dcrnn` (Candidate 1, $K=2$): ~3.2s per epoch.

### Measured full-data training (submitted runs, Apple M3, MPS)

The estimates that used to be here (DCRNN "4-6 minutes per 60-epoch run on a T4") were wrong by more than an order of magnitude. Measured values:

| Run | Seconds per epoch | Epochs | Stop | Time |
|---|---|---|---|---|
| `gru_local` search c0-c3 | 11-39 | 23-60 | 1 epoch limit, 3 time cap (15 min) | 56 min total |
| `gru_local` finals (c1, 3 seeds) | about 19 | 50-60 | 2 early stop, 1 epoch limit | 53 min total |
| `dcrnn` search c0 (h32, K=1) | about 95 | 19 | time cap (30 min) | 30 min |
| `dcrnn` search c1 (h32, K=2) | about 140 | 13 | time cap | 30 min |
| `dcrnn` search c2 (h64, K=2), picked | about 200 | 9 | time cap | 30 min |
| `dcrnn` search c3 (h32, 2 layers) | about 300 | 6 | time cap | 30 min |
| `dcrnn` finals (c2, 3 seeds) + `dcrnn_identity` (1 seed) | about 215 | 14 each | time cap (50 min) | 201 min total |

For comparison, a Colab T4 ran DCRNN c0 at about 62 s/epoch and c1 at about 92 s/epoch, roughly 1.5 times faster than the M3. The Colab sessions did not last long enough to finish, so they were not used.

## 5. Results (test split, source speed units)

Search (validation MAE, mean over horizons): c0 2.826, c1 2.836, **c2 2.783 (picked)**, c3 2.820 (`data/metadata/phase2/search_dcrnn.csv`). Final validation MAE: 2.736 / 2.749 / 2.754 (seeds 42/43/44); `dcrnn_identity` 3.015.

| Model | 5 min | 15 min | 30 min | 60 min |
|---|---|---|---|---|
| `gru_local` | 2.369 | 2.975 | 3.560 | 4.379 ± 0.006 |
| `dcrnn_identity` (1 seed) | 2.375 | 2.977 | 3.564 | 4.385 |
| `dcrnn` | 2.305 | 2.799 | 3.220 | 3.734 ± 0.036 |
| XGBoost + HA 36 (Phase 1) | 2.312 | 2.844 | 3.275 | 3.688 |

- DCRNN vs `dcrnn_identity` at 60 min: +0.650 [+0.507, +0.785] (paired day-block bootstrap; positive = DCRNN lower MAE).
- DCRNN vs XGBoost + HA: +0.007 [-0.004, +0.018] (5 min), +0.045 [+0.024, +0.067] (15), +0.055 [+0.017, +0.093] (30), -0.046 [-0.100, +0.004] (60).
- 5-minute MAE 2.305 is well above the ~2.00 leakage red flag.
- Every DCRNN run stopped at its wall-clock cap, after 14 final epochs, with the best epoch at 10-14. DCRNN is therefore probably under-trained; with more time its errors would likely fall. The published DCRNN reaches 3.56 at 60 min, under a different sample definition.
