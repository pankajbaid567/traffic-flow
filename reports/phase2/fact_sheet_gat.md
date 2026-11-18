# Phase 2 Fact Sheet: Graph Attention Network (GAT)

**Owner:** Member 2 (`phase2/m2-gat`)  
**Architecture file:** `src/phase2/models/gat.py`  
**Run configuration:** `configs/phase2/runs_gat.yaml`  
**Tests:** `tests/test_phase2_gat.py`  
**Attention analysis:** `scripts/phase2_gat_attention_stats.py`  

---

## 1. Architecture & Design Choices

The Graph Attention Network (GAT) models spatiotemporal traffic speed dependencies by dynamically weighting representations of neighbouring road sensors.

### Key Components:
1. **Shared Per-Sensor Temporal Encoder**:
   - Each sensor's 12-step input history `[B, 12, 7]` is encoded into a hidden representation `[B, hidden]`.
   - The encoder weights are shared across all $N=207$ sensors (mirroring how tree baselines and local models share feature representations).
   - Supported encoder types:
     - `gru`: A single-layer Gated Recurrent Unit (GRU).
     - `conv`: Two stacked 1D temporal convolutions with ReLU and adaptive pooling.
2. **Dense Multi-Head Masked Attention (Plain PyTorch)**:
   - For each head $k \in \{1 \dots K\}$:
     $$e_{ij}^{(k)} = \text{LeakyReLU}\left(a_{\text{src}}^{(k)\top} W^{(k)} h_i + a_{\text{dst}}^{(k)\top} W^{(k)} h_j\right)$$
   - **Graph Masking**: For disconnected pairs ($adj_{ij} == 0$ and $i \neq j$), logits are masked with $-10^9$ before softmax. Self-loops are always permitted ($adj_{ii} = 1$).
   - Normalised attention weights:
     $$\alpha_{ij}^{(k)} = \frac{\exp(e_{ij}^{(k)})}{\sum_{l \in \mathcal{N}_i \cup \{i\}} \exp(e_{il}^{(k)})}$$
   - Multi-head outputs are concatenated and projected back to the hidden dimension.
   - Includes residual connection $h + \text{GAT}(h)$, LayerNorm, and Dropout.
3. **Residual Horizon Forecasting Head**:
   - Instead of predicting raw speeds from scratch, an MLP predicts the **residual speed change** $\Delta y$ relative to the most recent filled speed input $x[:, -1, :, 0]$:
     $$\hat{y} = x[:, -1, :, 0] + \text{MLP}(h_{\text{final}})$$
   - Horizons predicted: 4 steps corresponding to +5 min ($t+1$), +15 min ($t+3$), +30 min ($t+6$), and +60 min ($t+12$).
4. **Attention Map Extraction**:
   - `attention_maps(x, adj)` exports multi-head attention weights `[B, heads, N, N]` per layer without altering model evaluation state.

---

## 2. Tensor Shapes Contract

| Tensor | Shape | Notes |
|---|---|---|
| Input `x` | `[B, 12, 207, 7]` | Channels: 0 z-speed, 1 missing flag, 2-3 hour sin/cos, 4-5 DoW sin/cos, 6 is_weekend |
| Adjacency `adj` | `[207, 207]` | Dense float32 (`bidirectional`, `identity`, or `shuffled`) |
| Temporal Embedding $h$ | `[B, 207, hidden]` | Output of per-sensor temporal encoder |
| Attention Logits $E$ | `[B, heads, 207, 207]` | Masked dense pairwise attention |
| Attention Weights $\alpha$ | `[B, heads, 207, 207]` | Sum along dim=-1 equals 1.0; non-edges strictly 0.0 |
| Output $\hat{y}$ | `[B, 4, 207]` | Forecasts in z-scored speed units |

---

## 3. Search Candidates & Exact Parameter Counts

The search block in `configs/phase2/runs_gat.yaml` defines exactly 4 candidates evaluated on validation data (seed 42):

| Candidate | Hidden | Heads | Layers | Dropout | Encoder | Parameter Count |
|---|---|---|---|---|---|---|
| **Candidate 0** | 32 | 2 | 1 | 0.1 | `gru` | **6,276** |
| **Candidate 1** | 64 | 2 | 1 | 0.1 | `gru` | **22,788** |
| **Candidate 2** | 32 | 4 | 2 | 0.2 | `gru` | **7,428** |
| **Candidate 3** | 32 | 2 | 1 | 0.1 | `conv` | **6,148** |

*All counts verified via `tests/test_phase2_gat.py::test_parameter_counts`.*

---

## 4. Final Runs & Experimental Controls

The best configuration selected by `src.phase2.cli pick` (lowest mean validation MAE across the 4 horizons) is evaluated across 3 seeds (`[42, 43, 44]`) on test data, alongside two structural controls:
1. `gat` (graph: `bidirectional`, seeds: `[42, 43, 44]`): Main model with real road adjacency.
2. `gat_identity` (graph: `identity`, seeds: `[42, 43, 44]`): Architecture control isolating the GAT mechanism without spatial graph neighbour mixing.
3. `gat_shuffled` (graph: `shuffled`, seed: `[42]`): Topology control preserving sensor degree distribution while assigning incorrect neighbour connections.

---

## 5. Measured search and final runs (Apple M3, MPS)

Search (seed 42, validation only, 20-minute cap per candidate; tracked copy in `data/metadata/phase2/search_gat.csv`):

| Candidate | Params | Epochs | Stop | Val MAE (mean of 4 horizons) | Val MAE 60 min |
|---|---|---|---|---|---|
| 0 (h32, 2 heads, 1 layer, GRU) | 6,276 | 60 | max epochs | 2.966 | 3.828 |
| **1 (h64, 2 heads, 1 layer, GRU)** | 22,788 | 46 | time cap | **2.921** | 3.729 |
| 2 (h32, 4 heads, 2 layers, GRU) | 7,428 | 32 | time cap | 2.984 | 3.852 |
| 3 (h32, 2 heads, 1 layer, conv) | 6,148 | 50 | early stop | 2.993 | 3.864 |

Candidate 1 was picked (`configs/phase2/final/gat.yaml`). Final runs (40-minute cap, 60 epochs, patience 8; from each `seed<S>_meta.json`):

| Run | Seeds | Epochs | Best epoch | Best val MAE | s / epoch | Minutes per run | Stop |
|---|---|---|---|---|---|---|---|
| `gat` | 42, 43, 44 | 60 | 54-60 | 2.916-2.923 | 26.4 | 26.4 | max epochs |
| `gat_identity` | 42, 43, 44 | 46-55 | 38-47 | 3.006-3.011 | 26.3 | 20-24 | early stop |
| `gat_shuffled` | 42 | 60 | 54 | 2.998 | 26.4 | 26.4 | max epochs |

The `gat` and `gat_shuffled` runs were still improving slightly when they reached the
pre-registered 60-epoch limit (best epoch 54-60), so they may be a little under-trained. The
limit was not raised after seeing results. Whole GAT family (search plus 7 final runs): about
4 h 05 min on the M3.

### Test results (source speed units)

| Model | 5 min | 15 min | 30 min | 60 min |
|---|---|---|---|---|
| `gru_local` | 2.369 | 2.975 | 3.560 | 4.379 ± 0.006 |
| `gat_identity` | 2.367 | 2.971 | 3.554 | 4.364 ± 0.003 |
| `gat_shuffled` (1 seed) | 2.365 | 2.963 | 3.536 | 4.324 |
| `gat` | 2.331 | 2.905 | 3.443 | 4.175 ± 0.007 |

60-minute paired bootstrap (positive = first model has lower MAE):

| Comparison | Interval |
|---|---|
| GAT vs GAT identity | +0.189 [+0.154, +0.221] |
| GAT vs GAT shuffled | +0.149 [+0.124, +0.174] |
| GAT shuffled vs GAT identity | +0.040 [+0.026, +0.054] |
| GAT vs GRU Local | +0.205 [+0.168, +0.238] |

H1 is supported. H2 is contradicted: part a is supported, but part b is contradicted because the shuffled graph also beat the identity graph.

Attention (seed 42, 12 test origins): 0.10 on the sensor itself and 0.90 on neighbours, almost identical at peak and off-peak (`attention_stats.csv`).

## 6. Scientific Limitations & Claims Disclaimer

> [!IMPORTANT]
> **Attention Weights Are NOT Causal Proof**:
> Attention weights indicate learnable correlation and dynamic feature aggregation within this specific statistical model. They do **not** constitute proof of physical vehicle shockwave propagation, traffic fluid dynamics, or causal influence.

### Additional Limitations:
- The speed dataset (`metr-la.h5`) uses stored clock timestamps and unverified speed units.
- The adjacency matrix reflects spatial distance Gaussian thresholds rather than lane-level directionality.
- Controls inherit hyperparameter configurations tuned for the bidirectional graph (`params: from_search`), which is an intentional ablation design that may slightly underestimate control performance.
