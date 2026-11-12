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

## 5. Compute & Runtime Benchmarks

### Measured Local Speed (Apple Silicon M3, 8 GB RAM):
- **5% Data Subset (1,198 train origins)**:
  - M3 MPS: ~1.2 s per epoch (warm). Peak RSS: ~780 MB (+1.1 GB driver memory).
  - M3 CPU: ~3.0 s per epoch.
- **Estimated Full Epoch on M3 MPS**:
  - $1.2\text{ s} \times 20 \approx 24\text{ s}$ per full epoch.

### Expected Colab T4 GPU Runtime (Training Strategy):
- On Google Colab Nvidia T4 (CUDA), batched matrix multiplications are ~3–4x faster than M3 MPS.
- Expected full epoch on T4: ~6–8 s per epoch.
- Over 60 epochs with early stopping (patience 8; typical convergence around 25–35 epochs):
  - **Search candidate run**: ~3.5 to 5 minutes per candidate (the CLI `--time-limit` cap applies **per candidate**, not per total search).
  - **Total search (4 candidates, sequential)**: 4 × ~3.5–5 min = **~14 to 20 minutes total** (each candidate may use up to the per-candidate cap independently).
  - **Final runs (3 seeds gat + 3 seeds gat_identity + 1 seed gat_shuffled = 7 runs)**: ~30 to 45 minutes total.
  - Worst-case total on Colab T4: ~1.5 to 2.5 hours across all runs.

---

## 6. Scientific Limitations & Claims Disclaimer

> [!IMPORTANT]
> **Attention Weights Are NOT Causal Proof**:
> Attention weights indicate learnable correlation and dynamic feature aggregation within this specific statistical model. They do **not** constitute proof of physical vehicle shockwave propagation, traffic fluid dynamics, or causal influence.

### Additional Limitations:
- The speed dataset (`metr-la.h5`) uses stored clock timestamps and unverified speed units.
- The adjacency matrix reflects spatial distance Gaussian thresholds rather than lane-level directionality.
- Controls inherit hyperparameter configurations tuned for the bidirectional graph (`params: from_search`), which is an intentional ablation design that may slightly underestimate control performance.
