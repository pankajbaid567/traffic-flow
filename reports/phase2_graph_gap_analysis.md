# Phase 2 Transition & Gap Analysis: From Segment-Level AML to Network-Aware Graph Learning

**Author:** Person E (Lead Evaluator, Benchmark Synthesizer & Phase 1 Integration Lead)  
**Date:** 2026-10-19  
**Focus:** Physical, mathematical, and empirical analysis of "The Gap", motivating Graph Attention Networks (GAT) and Diffusion Convolutional models for Phase 2.  

---

## 1. Problem Statement: The Applied Machine Learning Ceiling

In Phase 1, the team demonstrated that tabular applied machine learning models (Random Forest and XGBoost) achieve significant error reductions over classical persistence (up to 31.5% improvement at 60 min for XGBoost + HA).

However, an empirical ceiling is encountered as the forecast horizon extends to 30 and 60 minutes:
- At 60 minutes, plain XGBoost (32 features) fails to outperform the unconditional Historical Average baseline on the test set (MAE 4.390 vs 4.188).
- During evening rush hours, forecast error escalates by 75–90% compared to off-peak periods across all trained models.
- Corridor volatility mapping reveals that major highway interchanges (I-405 Sepulveda Pass, I-10 / I-110 Hub) suffer massive error spikes that local tree ensembles cannot suppress.

This limitation is not a defect of hyperparameter tuning or tree depth; it is a **fundamental structural limitation of independent segment-level models**.

---

## 2. The Physics of Traffic Flow: Hydrodynamic Shockwaves

Traffic on a highway network is physically coupled by hydrodynamic laws governing fluid flow. Under the classical **Lighthill-Whitham-Richards (LWR)** continuum model:

$$\frac{\partial k}{\partial t} + \frac{\partial q}{\partial x} = 0$$

where:
- $k(x, t)$ is traffic density (vehicles per mile),
- $q(x, t) = k \cdot v$ is traffic flow rate (vehicles per hour),
- $v(x, t)$ is spatial mean speed.

When traffic transitions from uncongested to congested flow (e.g., due to an incident, merge bottleneck, or signal queue), a **kinematic shockwave** is generated at the boundary between flow states:

$$w = \frac{\Delta q}{\Delta k} = \frac{q_2 - q_1}{k_2 - k_1}$$

Because congested density $k_2$ is substantially greater than free-flow density $k_1$, the shockwave velocity $w$ is **negative**:
$$w \approx -10 \text{ to } -15 \text{ mph}$$

### The Failure Mode of Independent Models:
- The shockwave travels **backwards (upstream)**, opposite to the direction of vehicular travel.
- Suppose Sensor $D$ is at the bottleneck, Sensor $C$ is 2 miles upstream, Sensor $B$ is 4 miles upstream, and Sensor $A$ is 6 miles upstream.
- At time $t_0$, an accident occurs at Sensor $D$, dropping its speed from 60 mph to 15 mph.
- At time $t_0$, Sensor $A$ is completely unobstructed; vehicles are cruising at 62 mph.
- Person B's 32 features for Sensor $A$ (lags, rolling averages, trend deltas) observe only historical speeds at Sensor $A$. Consequently, all 12 lag features read ~62 mph.
- When tasked with forecasting speed at Sensor $A$ at $t+30$ minutes, **an independent AML model has no physical mechanism to know that a shockwave is propagating towards it**. It predicts ~60 mph.
- 30 minutes later, the shockwave arrives at Sensor $A$, collapsing actual speed to 25 mph. The model incurs an error of **35 mph**.

```
                           TRAFFIC SHOCKWAVE SPILLBACK PROPAGATION
                           
     Traffic Flow Direction (Eastbound: A -> B -> C -> D)   ════════════════════════════════►
     
     Sensor A (Mile 0)        Sensor B (Mile 2)        Sensor C (Mile 4)        Sensor D (Mile 6)
    ┌────────────────┐       ┌────────────────┐       ┌────────────────┐       ┌────────────────┐
    │  v = 62 mph    │       │  v = 58 mph    │       │  v = 38 mph    │       │  v = 15 mph    │
    │  (Free Flow)   │       │  (Transition)  │       │  (Queue Tail)  │       │  (Bottleneck)  │
    └────────────────┘       └────────────────┘       └────────────────┘       └────────────────┘
            ▲                        ▲                        ▲                        │
            │                        │                        │                        │
            └────────────────────────┴────────────────────────┴────────────────────────┘
                   Congestion Shockwave Propagates Upstream (Westbound: D -> C -> B -> A)
                   Shockwave Speed: w = Δq / Δk ≈ -12 mph
                   Arrival at Sensor A: ~30 minutes
```

---

## 3. Mathematical Blueprint for Phase 2: Graph Neural Networks

In Phase 2, the road network is represented as a directed graph $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{A})$, where:
- $\mathcal{V}$ is the set of $|\mathcal{V}| = 207$ loop detector sensors,
- $\mathcal{E}$ is the set of directed edges representing highway connectivity,
- $\mathbf{A} \in \mathbb{R}^{207 \times 207}$ is the spatial adjacency matrix constructed from physical network distance:

$$A_{ij} = \exp \left( -\frac{\text{dist}(v_i, v_j)^2}{\sigma^2} \right) \quad \text{for } \text{dist}(v_i, v_j) \le \kappa$$

### 3.1 Architecture 1: Graph Attention Networks (GAT)
Graph Attention Networks (Veličković et al., ICLR 2018) dynamically compute time-varying spatial attention coefficients between connected road segments:

$$\alpha_{ij}^{(t)} = \frac{\exp \left( \text{LeakyReLU} \left( \mathbf{a}^T [\mathbf{W} \mathbf{h}_i^{(t)} \parallel \mathbf{W} \mathbf{h}_j^{(t)}] \right) \right)}{\sum_{k \in \mathcal{N}_i} \exp \left( \text{LeakyReLU} \left( \mathbf{a}^T [\mathbf{W} \mathbf{h}_i^{(t)} \parallel \mathbf{W} \mathbf{h}_k^{(t)}] \right) \right)}$$

$$\mathbf{h}_i^{(t+1)} = \sigma \left( \sum_{j \in \mathcal{N}_i} \alpha_{ij}^{(t)} \mathbf{W} \mathbf{h}_j^{(t)} \right)$$

#### Why GAT Solves the Shockwave Problem:
- Unlike static graph convolutions, the attention coefficient $\alpha_{ij}^{(t)}$ is **data-dependent and dynamic**.
- When downstream Sensor $D$ collapses in speed, the feature divergence $[\mathbf{h}_A \parallel \mathbf{h}_D]$ triggers a sharp increase in attention weight $\alpha_{AD}^{(t)}$, immediately propagating the downstream congestion alert to upstream node $A$.
- Multi-head attention allows the model to simultaneously dedicate distinct attention heads to **upstream inflow monitoring** (volume demand) and **downstream queue spillback** (shockwave alert).

### 3.2 Architecture 2: Diffusion Convolutional Recurrent Neural Network (DCRNN)
To capture bidirectional traffic dynamics, DCRNN (Li et al., ICLR 2018) models traffic flow as a diffusion process on a directed graph using bidirectional random walks:

$$\mathbf{X}_{:, j \star \mathcal{G}} f_\theta = \sum_{k=0}^{K-1} \left( \theta_{k, 1} (\mathbf{D}_O^{-1} \mathbf{A})^k + \theta_{k, 2} (\mathbf{D}_I^{-1} \mathbf{A}^T)^k \right) \mathbf{X}_{:, j}$$

where:
- $\mathbf{D}_O^{-1} \mathbf{A}$ represents the **forward random walk transition matrix** (modeling forward physical vehicular movement),
- $\mathbf{D}_I^{-1} \mathbf{A}^T$ represents the **backward random walk transition matrix** (modeling backward shockwave propagation),
- $K$ is the diffusion step radius (typically $K=2$ or $K=3$).

---

## 4. Phase 2 Implementation Roadmap

1. **Framework Stack:**
   - PyTorch (`torch>=2.2.0`)
   - PyTorch Geometric (`torch_geometric>=2.5.0`)
   - NetworkX (`networkx>=3.2`)
2. **Input Pipeline Integration:**
   - Sensor graph adjacency $\mathbf{A} \in \mathbb{R}^{207 \times 207}$ loaded from canonical `data/raw/adj_mx.pkl`.
   - Node feature tensor: $\mathcal{X} \in \mathbb{R}^{B \times T_{in} \times N \times F}$, where batch size $B$, history window $T_{in} = 12$ (1 hour), sensor count $N = 207$, and feature dimension $F = 2$ (normalized speed + time-of-day).
   - Target tensor: $\mathcal{Y} \in \mathbb{R}^{B \times T_{out} \times N \times 1}$, where $T_{out} = 12$ (1 hour forecasting horizon).
3. **Strict Experimental Discipline (Preserving Phase 1 Standards):**
   - Retain Person A's exact chronological split contract: train (70%), validation (10%), test (20%).
   - Retain Person E's common evaluation suite: masked MAE, RMSE, and MAPE strictly filtering `target_valid_<h> == True`.
   - Fair like-for-like comparison: evaluate Phase 2 DL models against Phase 1 AML models (RF and XGBoost) on identical test timestamps.
4. **Traffic Signal Policy Simulation (Optional Extension):**
   - Utilize **SUMO (Simulation of Urban MObility)** or **CityFlow** to model synthetic traffic signal adjustments at network intersections.
   - **Crucial Rule:** As mandated by the capstone prompt, signal control experiments must strictly be designated as *simulated only*; the project must never claim live traffic actuation or physical control.

---

## 5. Summary Table: Phase 1 AML vs Phase 2 DL

| Dimension | Phase 1 Applied ML (RF & XGBoost) | Phase 2 Deep Learning (GAT & DCRNN) |
|---|---|---|
| **Spatial Scope** | Segment-level independent (207 isolated models/rows) | Network-aware graph topology ($\mathbf{A} \in \mathbb{R}^{207 \times 207}$) |
| **Spatial Modeling** | None (purely local autoregressive & rolling features) | Dynamic Graph Attention ($\alpha_{ij}$) / Diffusion Convolutions |
| **Shockwave Detection** | Blind until shockwave reaches local sensor lags | Detects downstream queues immediately via adjacency walk |
| **Horizon Scaling** | Severe degradation at 30m–60m (loses to Hist. Avg) | Sustained predictive accuracy across 15m, 30m, and 60m |
| **Input Representation** | 2D tabular matrix $(N \cdot T) \times 32$ | 4D spatio-temporal tensor $B \times T \times N \times F$ |
| **Computational Footprint** | Extremely fast training (<2 min), CPU friendly | Requires GPU acceleration for tensor backpropagation |
