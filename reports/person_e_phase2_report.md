# Person E — Phase 2 Synthesis & Viva Guide
## Graph Forecasting on METR-LA and a Simulated Signal-Control Experiment

**Author:** Ranvendra Pratap Singh (Person E / Member 5)
**Repository:** `traffic-flow` (originally on branch `phase2/m5-signal-paper`, merged to `main`)
**First written:** 2026-11-13 · **Corrected:** 2026-11-17 · **Results added:** 2026-11-18
**Status:** **Complete.** All models trained (Apple M3, MPS) and evaluated; paper, slides and notebook built from the evaluation outputs.

> **Correction (2026-11-17).** The 2026-11-13 version of this report said Phase 2 was complete. It described results that do not exist:
> - **Forecast results:** GAT 3.58 / DCRNN 3.54 at 60 min, and "H1 and H2 supported". No Phase 2 model had been trained; those numbers were typed into the paper script.
> - **Attention findings:** self-attention ≈ 0.65 and "+280% at onsets". There is no trained model to extract attention from.
> - **Sensor 717468 "case study":** the figure was drawn from hand-made curves, not data. It has been removed.
> - **Signal results:** "44.4% less delay" and "spillback 64% → 26%". The simulator file was committed empty, so these could not have come from it. The rebuilt simulator gives very different results (Section 4).
> - **Latency and edge-hardware claims:** 0.82 ms / 4.80 ms. There was no measurement behind them.
>
> Do **not** use the old numbers in the viva. The measured results are below (and in `data/metadata/phase2/`).

---

## Summary

- **Phase 1 recap.** Local models used only each sensor's own history. The best was XGBoost + historical-average features, with test MAE 2.312 / 2.844 / 3.275 / 3.688 at 5 / 15 / 30 / 60 min in source speed units. Error grows with horizon, which is consistent with a model that cannot see neighbouring sensors. Phase 1 did not test that explanation.
- **Phase 2 question.** Does giving a forecaster neighbouring sensors through the road graph improve forecasts over equally capable models without the graph? It uses the same split, masks and history window. "No detectable gain" is an acceptable answer.
- **What exists now.**
  - GAT, DCRNN and the controls (`gru_local`, `gat_identity`, `gat_shuffled`, `dcrnn_identity`).
  - A trainer that resumes after interruptions (Colab disconnects or a killed local process).
  - An evaluation harness that reproduces the Phase 1 MAEs exactly.
  - A simulated signal-control experiment.
  - A paper builder that reads results from files.
  - 107 Phase 2 unit tests, all passing.
- **Headline results (60-min test MAE).** GAT 4.175 ± 0.007, DCRNN 3.734 ± 0.036, GRU Local 4.379, XGBoost + HA 3.688.
  - The road graph helps both graph models against their identity-graph controls: GAT +0.189 [+0.154, +0.221], DCRNN +0.650 [+0.507, +0.785].
  - A shuffled graph also helps GAT a little (+0.040 [+0.026, +0.054]), so H2 as pre-registered is contradicted, even though the real graph beats the shuffled one (+0.149).
  - DCRNN beats XGBoost + HA at 15 and 30 min; at 5 and 60 min the difference is not detectable.

---

## 1. Why look at the road graph

Phase 1 models saw only their own sensor's last hour: 12 lags, rolling statistics and calendar features, plus historical-average profiles in the 36-feature model. Kinematic-wave theory (Lighthill–Whitham–Richards) says congestion propagates along a corridor. A sensor's neighbours can therefore carry information about what will reach it. That is a **motivation**, not a result: whether METR-LA forecasts actually improve is what the H1–H4 tests decide.

## 2. Models and controls

- **GAT** (`src/phase2/models/gat.py`). A shared GRU or 1-D convolution encodes each sensor's history. Masked multi-head attention mixes neighbours (logits for unconnected pairs are set to −1e9), followed by a residual head. Candidate 0 has 6,276 parameters.
- **DCRNN** (`src/phase2/models/dcrnn.py`). K-hop diffusion with forward `D_out⁻¹A` and reverse `D_in⁻¹Aᵀ` random-walk supports, inside a DCGRU encoder–decoder. Candidate 2 has 131,009 parameters.
- **Controls:**
  - `gru_local`: a shared per-sensor GRU with no spatial mixing (5,124 parameters for candidate 0).
  - `gat_identity` / `dcrnn_identity`: the same architecture with self-loops only.
  - `gat_shuffled`: a node-permuted graph with the same degree structure.

  Controls reuse the tuned parameters of their main model, which can slightly understate them.

Parameter counts are asserted in the unit tests and printed in paper Table IV.

## 3. Hypotheses (fixed before results; verdicts computed by code)

| ID | Hypothesis | Verdict |
|---|---|---|
| H1 | `gat` beats `gru_local` at 30 and 60 min | supported (+0.118 at 30 min, +0.205 at 60 min) |
| H2 | `gat` beats `gat_identity`; `gat_shuffled` no better than `gat_identity` | contradicted: part a supported (+0.189 at 60 min), part b contradicted (shuffled +0.040 [+0.026, +0.054]) |
| H3 | `gat` vs `gat_identity` gain larger in peak than off-peak | supported (+0.109 [+0.043, +0.180] at 60 min) |
| H4 | gain grows with horizon | supported (each step's interval above zero) |
| H5 | gain larger at sensors with more neighbours / around congestion onsets | descriptive: Spearman 0.28 with degree; 60-min gain 2.22 at proxy onsets vs 0.10 elsewhere |
| H0 | `gat`, `dcrnn` do not differ from `xgboost_ha` | contradicted: GAT worse at every horizon; DCRNN better at 15/30 min, no detectable difference at 5/60 min |

A claim counts as supported only if the whole 95% paired-bootstrap interval of the MAE reduction is above zero (2-day blocks, 2,000 resamples, seed 42).

## 4. Simulated signal control (software only)

`src/phase2/signal_sim/simulator.py`, run by `scripts/phase2_run_signal_sim.py`:

- **Setup.** One main approach (2 lanes) and one cross approach (1 lane, 500 veh/h). 90 s cycle, 10 s lost time, green between 15 and 60 s (default 35 s), saturation flow 1,800 veh/h/lane, storage 60 vehicles.
- **Arrivals and delay.** Greenshields-shaped arrivals. Delay is Webster's uniform term plus one cycle for each vehicle left in the queue.
- **Experiment A (synthetic).** Speeds are synthetic. Forecasts are "true future speed + noise" sized to measured test MAEs: 2.970 for the local forecast (Phase 1 XGBoost 32, 15 min) and GAT's 2.905 / 3.443 (15 / 30 min) for the graph forecast, plus a two-cycle upstream lead. 4 scenarios × 4 policies × 5 seeds = 80 runs.
- **Experiment B (real inputs).** Real test-period weekday speeds (06:00-21:00) at the 30 sensors with the most proxy congestion, and each model's actual test forecasts; 3,780 runs. Only the forecast source differs between predictive policies.

Experiment A: mean delay per vehicle in seconds, with spillback rate (`data/metadata/phase2/signal_sim_summary.json`):

| Scenario | Fixed-time | Reactive | Local predictive | Graph predictive |
|---|---:|---:|---:|---:|
| Morning peak | 222.3 (44.3%) | 26.9 (0%) | 24.9 (0%) | 23.9 (0%) |
| Evening peak | 447.0 (63.5%) | 27.4 (0%) | 25.7 (0%) | 24.7 (0%) |
| Incident | 297.5 (46.5%) | 27.6 (0%) | 26.3 (0%) | 25.7 (0%) |
| Off-peak | 19.6 (0%) | 27.5 (0%) | 25.7 (0%) | 23.9 (0%) |

Experiment B: delay saved per vehicle relative to the XGBoost + HA forecast (`signal_sim_real_summary.json`): DCRNN +0.01 s [-0.02, +0.04], GAT -0.19 s, GRU Local -0.31 s, reactive -0.67 s, perfect forecast +0.40 s. Fixed timing cannot keep up with the assumed full-day demand (delay grows without bound).

**What this supports.**
- In congested conditions, any adaptive policy is far better than fixed timing.
- Forecast accuracy changes delay by fractions of a second per vehicle; even a perfect forecast adds only 0.4 s in this simulator.
- Off-peak (experiment A), fixed timing is best.

These are properties of the simulator under its assumptions. They say nothing about real signals.

---

## 5. Viva questions and defensible answers

**Q1. Did the graph models beat XGBoost?**
*"Partly. DCRNN has lower test MAE than XGBoost + HA at 15 and 30 minutes, and at 5 and 60 minutes the interval includes zero. GAT is worse than XGBoost + HA at every horizon. But XGBoost + HA has historical-average features the deep models don't get, so this comparison does not isolate the graph. Only `gat` vs `gat_identity` / `gru_local` and `dcrnn` vs `dcrnn_identity` do, and those show clear gains."*

**Q2. How would you know a gain came from the graph and not from model capacity?**
*"Through the controls. `gru_local` has the same temporal capacity with no neighbours. `gat_identity` is the same architecture with self-loops only. `gat_shuffled` keeps the degree structure but connects the wrong sensors. If GAT beats the first two and the shuffled graph doesn't help, the gain is tied to the real road graph. In our results the shuffled graph helped a little (+0.040 at 60 min) while the real graph helped much more (+0.189), so most but not all of the gain is tied to the true links. That is why H2 is reported as contradicted. The bootstrap verdicts decide each step; we don't decide by eye."*

**Q3. Why a simulation instead of real signals?**
*"Testing an unvalidated controller on real signals would be unsafe, and we don't have access to one. The simulator is a simple queue model with synthetic inputs. We report it as simulated and use it only to show what a given forecast accuracy is worth under stated assumptions."*

**Q4. Your simulation shows less than 1 s/veh benefit from a better forecast. Isn't that disappointing?**
*"It's the honest result. With real speeds and real forecasts, even a perfect forecast saves only 0.4 s per vehicle over the XGBoost forecast. The big effect is adaptive versus fixed timing. In this simple queue model green time is sized from the worst speed signal, so small forecast errors rarely change the decision."*

**Q5. What about Sensor 717468 from Phase 1?**
*"Paper Fig. 9 now plots the observed speed at that sensor on 2012-06-11 against every model's real test forecast. It is one hand-picked event, chosen after evaluation, so it is illustrative only; the aggregate tables carry the evidence. The earlier figure was removed because it was not based on data."*

**Q6. Do attention weights show shockwaves?**
*"No. Attention shows what the model weighted, not how traffic moves. For the trained GAT, about 90% of attention goes to neighbours and 10% to the sensor itself, almost the same at peak and off-peak. That says the model uses neighbours, not that it tracks shockwaves."*

---

## 6. Status checklist

| Item | Status |
|---|---|
| `scripts/phase2_check_phase1_frozen.py` | OK |
| `python -m unittest discover -s tests -p "test_phase2_*.py"` | 107 tests, OK |
| Training (GRU Local, GAT + controls, DCRNN + control) | done, Apple M3 (MPS), 11.2 h; `run_meta.csv` |
| `scripts/phase2_eval_all.py` on real predictions | done; Phase 1 parity passed |
| Signal simulation (`scripts/phase2_run_signal_sim.py`) | experiment A 80 runs, experiment B 3,780 runs, reproducible |
| Paper, slides, notebook | built from evaluation outputs |
| Claims QA (`reports/phase2/qa_claims_checklist.md`) | closed |
| Reproduction guide (`reports/phase2/reproduction.md`) | complete with measured timings |
