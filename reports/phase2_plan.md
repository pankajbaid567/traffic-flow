# Phase 2 Plan: Graph Attention and Diffusion Forecasting on METR-LA, plus Simulated Signal Control

Single source of truth for Phase 2. Lives in the same repo as Phase 1 (`traffic-flow-capstone`, remote `github.com/pankajbaid567/traffic-flow`, branch `main`).
Five members, **one prompt per member** (section 8). Each prompt is a bulk job: paste it once into your AI coding agent (Kiro) opened on this repo, and the agent does the whole workstream.

Contents: 0 Kickoff · 1 Phase 1 recap · 2 Repo rules · 3 Compute plan · 4 Shared contract · 5 Experiment protocol · 6 Timeline and final assembly · 7 Risks · 8 The five prompts

---

## 0. Kickoff (humans, about 20 minutes, once)

1. **Protect Phase 1.** In the repo root:
   `git tag phase1-final 138aa57 && git push origin phase1-final`
   (`138aa57` = "update redme", the last published Phase 1 commit on `main`; it sits on top of `f1e4f22` "Complete Phase 1 verification, results, and IEEE paper".)
2. **Commit and push this plan** so every agent can read it: `git add reports/phase2_plan.md && git commit -m "Add Phase 2 plan" && git push`.
3. **Share the git-ignored files through Google Drive.** Create `Phase2_shared/` with subfolders `inputs/` and `out/`, shared with all five (Editor). From the repo root, upload to `inputs/`:
   - `data/raw/metr-la.h5` (54 MB; sha256 must be `64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05`)
   - `data/raw/adj_mx.pkl` (668 KB; sha256 `a35687c6e15aa228dc45027b0ed2a0ea0f4ec78f573deb992c595092d12f61b3`)
   - `phase1_predictions.zip`, made with `zip -j phase1_predictions.zip data/processed/predictions/*.parquet` (182 MB, four files)
4. **Every member** clones the repo, then downloads those files into the same paths (`data/raw/` and `data/processed/predictions/`) and checks the two hashes with `shasum -a 256`.
5. **Member 1 should be the person who owns the repo and has the data on disk** (recommended: Pankaj). The other four can start their prompts at the same moment as Member 1; section 4 is the interface they all code against.

Coordination rule (the only cross-member dependency): Member 1 pushes the core (`src/phase2/{data,graph,train,cli,registry}.py`) first. Members 2 and 3 write and unit-test their model code against random tensors first, which does not need the core, and run the integration smoke test (`python -m src.phase2.cli smoke ...`) after pulling Member 1's branch. If the core is not there yet when they reach that step, they finish everything else and list the smoke test as pending in their final report.

---

## 1. Where Phase 1 left us (read from the repo)

| Item | Fact | Where |
|---|---|---|
| Data | METR-LA, 34,272 five-minute steps x 207 sensors, 2012-03-01 to 2012-06-27 on the stored clock | `data/raw/metr-la.h5` |
| Unavailable readings | Exact zeros are 8.109% of cells and are treated as missing. 2,148 all-sensor-zero timestamps in 97 blocks. | paper Sec. III |
| Split | 70/10/20 chronological. Train 23,990, validation 3,427, test 6,855 grid steps. Train ends 2012-05-23 07:05, validation 2012-06-04 04:40. | `data/metadata/split_contract.json` |
| Samples | History t-11..t, targets t+1, t+3, t+6, t+12 (5/15/30/60 min), all four in one partition. Origins: 23,967 / 3,416 / 6,844. Rows (x207): 4,961,169 / 707,112 / 1,416,708. | `src/data/split.py` |
| Graph | `adj_mx.pkl`: 207x207 float32, directed, diagonal 1, 1,722 non-zeros. Sensor order matches the speed table and `data/metadata/sensors.csv`. Not used as a model input in Phase 1. | inspected |
| Peak rule | Weekday origin hour in [7,10) or [16,19), stored clock. Off-peak = everything else. | `src/models/prediction_evaluation.py::temporal_masks` |
| Metrics | MAE/RMSE on finite positive targets; MAPE with denominator `max(y, 1e-5)`. | `src/models/metrics.py` |
| Uncertainty | Paired circular moving-block bootstrap over stored days: 2-day blocks, 2000 resamples, seed 42. | `paired_daily_bootstrap` |
| Predictions | Persistence, historical average, XGBoost 32, XGBoost+HA 36 in `person_c_{val,test}_predictions.parquet`; RF 32 in `person_d_*`. Git-ignored. | `data/processed/predictions/` |
| Limits | Speed units and timezone unverified. Regions are coordinate bins, not freeways. Congestion = proxy (speed < 60% of the sensor's training 85th percentile). One split, one seed. No graph or signal result in Phase 1. | reports |

Phase 1 test MAE (the numbers to beat or match):

| Model | 5 min | 15 min | 30 min | 60 min |
|---|---|---|---|---|
| Persistence | 2.814 | 3.499 | 4.206 | 5.380 |
| Historical average | 4.188 | 4.188 | 4.188 | 4.189 |
| Random Forest 32 | 2.452 | 3.167 | 3.871 | 4.854 |
| XGBoost 32 | 2.362 | 2.970 | 3.564 | 4.393 |
| XGBoost + HA 36 | 2.312 | 2.844 | 3.275 | 3.688 |

Phase 1 already wrote the Phase 2 ground rules (`reports/phase2_graph_gap_analysis.md`, `reports/person_e_phase2_handoff.md`): freeze split and horizons, fit preprocessing on train only, compare graph models with equally capable non-graph temporal models, repeat seeds, use paired day-block comparisons, label any signal experiment as simulated. This plan follows them.

Sanity range only (not a head-to-head): `DCRNN/README.md` in the AML folder lists DCRNN on METR-LA at MAE 2.18 / 2.67 / 3.08 / 3.56 for 5/15/30/60 min. Its sample definition differs from ours. It suggests a graph gain over the best tree model (2.312 / 2.844 / 3.275 / 3.688) is plausible but modest.

**The Phase 2 question:** does giving a forecaster neighbouring sensors through the road graph improve forecasts over equally capable models without the graph, on the same chronological split, masks and history window? "No detectable gain" is an acceptable, reportable answer.

Deliverables (from the project brief): a forecast comparison with congestion maps, horizon-wise MAE/RMSE/MAPE (with peak vs off-peak), and an optional **simulated** signal-policy experiment. We never claim live traffic control.

---

## 2. Working inside the Phase 1 repo

**Phase 1 is frozen.** Existing tracked files are never edited, with three exceptions owned by Member 1: `.gitignore` (append), `README.md` (append a Phase 2 pointer), and new `requirements-phase2.txt` (new file; `requirements.txt` stays untouched). Reuse Phase 1 code by importing it (`from src.data.load_data import ...`), never by editing it. If a Phase 1 function does not fit, wrap it in a Phase 2 module.

**Do NOT run these Phase 1 scripts.** They overwrite tracked Phase 1 outputs (metadata, figures, reports, paper): `complete_phase1.py`, `verify_phase1.py`, `run_person_c.py`, `run_person_d.py`, `run_person_e.py`, `build_phase1_paper.py`, `build_notebook_03.py`, `execute_notebooks.py`, `run_phase1_tests.py`, `acquire_data.py`, `verify_saved_boosters.py`.
Run tests with `python -m unittest discover -s tests -p "test_phase2_*.py" -v` (Phase 1 uses `unittest`, not pytest).

**Frozen check.** Member 1 provides `scripts/phase2_check_phase1_frozen.py`, which lists any Phase 1 file changed since tag `phase1-final` (except the three exceptions). It must print "OK" before anything is merged or submitted.

**New paths (all Phase 2 work lives here; each has one owner):**

| Path | Owner | Contents |
|---|---|---|
| `src/phase2/{__init__,data,graph,registry,train,cli}.py`, `src/phase2/models/__init__.py` | M1 | core pipeline, trainer, CLI |
| `src/phase2/models/gat.py` | M2 | GAT model |
| `src/phase2/models/dcrnn.py`, `src/phase2/models/controls.py` | M3 | DCRNN, `gru_local` |
| `src/phase2/evaluation/`, `src/phase2/viz/` | M4 | scoring, bootstrap, maps |
| `src/phase2/signal_sim/` | M5 | signal simulation |
| `configs/phase2/` | M1 base; `runs_gat.yaml` M2; `runs_dcrnn.yaml`, `runs_gru_local.yaml` M3; `final/` filled at the end | configs |
| `scripts/phase2_*.py` | the owner named in the prompt | entry points |
| `tests/test_phase2_<area>.py` | the area's owner | unittest |
| `notebooks/phase2_colab_train.ipynb` | M1 | Colab runner |
| `notebooks/phase2_error_analysis.ipynb` | M4 | analysis |
| `data/metadata/phase2/` | M4 (metrics), M5 (signal results) | small tracked CSV/JSON |
| `figures/phase2/` | M4 (forecast figures), M5 (signal figures) | tracked PNG |
| `models/phase2/` | M2/M3 | best checkpoints of final runs, each < 10 MB, tracked |
| `reports/phase2/`, `reports/paper_phase2/` | M5 (M2, M3, M4 each add one fact sheet) | write-ups, paper |
| `data/processed/phase2/` | everyone, git-ignored | predictions, checkpoints, search logs |

**Git flow.** One branch per member: `phase2/m1-core`, `phase2/m2-gat`, `phase2/m3-dcrnn-controls`, `phase2/m4-evaluation`, `phase2/m5-signal-paper`. Agents commit to their branch and do **not** push or merge; the human pushes and opens the pull request. Because every member only adds files in their own paths, merges should not conflict. Models register themselves with a decorator and are auto-imported, so no shared registry file is edited.

---

## 3. Compute plan: Mac M3 (8 GB) and Colab free tier

| Work | Where | Why |
|---|---|---|
| Writing code, unit tests, 1-epoch smoke runs on a 5% subset | M3 locally (device `mps`, fall back to `cpu`) | quick and free |
| Scoring models, bootstrap, maps, error analysis, signal simulation, paper | M3 locally | array-based and small |
| Hyperparameter search and final training (GAT, GRU, DCRNN) | Colab free tier (GPU if granted) | M3 is slow and shares 8 GB RAM |

Colab free tier usually gives a T4-class GPU, but the GPU is not guaranteed, sessions can be cut after a few hours or when idle, and daily GPU time is limited. The plan is built for that:
- **Checkpoint every epoch to Drive and resume automatically.** Re-running "Run all" after a disconnect continues where it stopped.
- **Wall-clock cap per run** (`train.max_minutes`); a run stops at the cap and keeps its best validation checkpoint.
- **Whole dataset on the accelerator**: one (T=34,272, N=207, C=7) float32 tensor is about 200 MB. Windows are gathered by index, so there are no DataLoader workers.
- Fixed fp32 (no AMP) so results are comparable across devices.
- Only `metr-la.h5`, `adj_mx.pkl` and the tracked metadata are needed on Colab.

Time caps (chosen, **not measured**; Member 1 measures real speed and the members can lower the caps in their YAML):

| Family | Search (4 candidates, seed 42) | Final runs | Worst-case total |
|---|---|---|---|
| GAT (M2) | 4 x 20 min | `gat` x3 seeds, `gat_identity` x3, `gat_shuffled` x1, each 40 min | about 6 h |
| `gru_local` (M3) | 4 x 15 min | x3 seeds, each 30 min | about 2.5 h |
| DCRNN (M3) | 4 x 30 min | `dcrnn` x3 seeds, `dcrnn_identity` x1, each 50 min | about 5.5 h |

Early stopping (patience 8) usually ends runs before the cap. These totals spread over two or three Colab sessions. Idle teammates can run part of the list on their own Colab account: `--only dcrnn:43` runs one seed, and everyone writes to the shared `Phase2_shared/out/`.

---

## 4. Shared contract (binding for all five prompts)

### 4.1 Reuse of Phase 1 code
Load, mask, impute and split with Phase 1 functions so parity is automatic:
`src.data.load_data` (`load_raw_speed_data`, `create_missingness_mask`, `fit_training_sensor_medians`, `apply_causal_imputation`, `load_graph_metadata`, `check_sensor_order`), `src.data.split` (`load_split_contract`, `get_partition_info`, `get_eligible_sample_origins`; partition names are `train`, `validation`, `test`), `src.models.metrics` (`compute_metrics`), `src.models.prediction_evaluation` (`temporal_masks`, `align_predictions`, `score_frame`, `paired_daily_bootstrap`, `region_scores`, `geographic_regions`), `src.visualization.spatial_mapping` (`load_sensor_geo_metadata`, `plot_prediction_map`, `plot_congestion_prediction_map`).

### 4.2 Data semantics
- A reading is *valid* if finite and > 0. Zeros, NaN, inf and negatives are unavailable, never "zero traffic". Targets are read from the raw array and never filled.
- Inputs use Phase 1's causal fill (forward fill, then the sensor's **training** median for leading gaps).
- Scaler: one global mean and std over valid **training** cells only.
- Origin counts must equal Phase 1: 23,967 / 3,416 / 6,844. Never shuffle.
- Peak flags: use `temporal_masks` semantics on the origin timestamp (stored clock). No timezone conversion.
- Say "source speed units" and "stored clock". Say "coordinate regions", "proxy congestion", "simulated".

### 4.3 Model input and output
- `x`: float32 `[B, 12, N=207, C=7]`; channels: 0 z-scored filled speed, 1 missing flag, 2-3 hour sin/cos, 4-5 day-of-week sin/cos, 6 is_weekend.
- `adj`: float32 `[207, 207]`, chosen by the run's `graph` key.
- Output: `[B, 4, N]` **z-scored** speed for horizons 5/15/30/60 min. The trainer inverts the scaler. The loss is masked MAE in source speed units on valid targets. All models are supervised only at these 4 horizons.
- Early stopping and `pick` use validation MAE averaged over the four horizons.

### 4.4 Registry and model API
```python
# src/phase2/registry.py (M1)
from src.phase2.registry import register_model

@register_model("gat")                      # registry name
def build(cfg: dict) -> torch.nn.Module:    # cfg = {num_nodes:207, in_channels:7, history:12,
    ...                                      #        num_horizons:4, **run["params"]}

class MyModel(nn.Module):
    needs_graph: bool                        # False for graph-free models
    def forward(self, x, adj): ...           # [B,12,N,C], [N,N] -> [B,4,N]
    def attention_maps(self, x, adj): ...    # OPTIONAL: list per layer of [B, heads, N, N]
```
`src/phase2/models/__init__.py` imports every module in the folder, so adding a file registers its models.

### 4.5 Graph variants (`src/phase2/graph.py`, M1)
`load_graph(variant, seed=0)` returns a dense float32 `[207,207]`:

| variant | meaning |
|---|---|
| `given` | `adj_mx.pkl` as supplied (directed) |
| `bidirectional` | `max(A, A^T)` |
| `identity` | self-loops only; the non-graph control with the same architecture |
| `shuffled` | node-permuted `A`, fixed seed, same degree structure; wrong-neighbour control |
| `random_matched` | random graph with the same edge count |

Edge direction (which axis is "source") must be documented in the docstring and tested on a 3-node example, following `DCRNN/lib/utils.py` (`D^-1 A` forward, `A^T` reverse). If `DCRNN/data/sensor_graph/distances_la_2012.csv` is present, regenerate the thresholded Gaussian kernel and compare with `adj_mx.pkl`; if absent, skip and say so.

### 4.6 Run files and CLI
`configs/phase2/runs_<name>.yaml`:
```yaml
name: gat
train: {lr: 0.001, weight_decay: 0.0, batch_size: 32, epochs: 60, patience: 8,
        grad_clip: 5.0, max_minutes: 40, scheduler: plateau}
groups:
  - search:                       # validation only, seed 42, no test predictions written
      model: gat
      graph: bidirectional
      max_minutes: 20
      candidates:                 # exactly 4
        - {hidden: 32, heads: 2, layers: 1, dropout: 0.1}
        - ...
    final:                        # "from_search" = params of the best candidate (see pick)
      - {model_id: gat,          model: gat, graph: bidirectional, params: from_search, seeds: [42, 43, 44]}
      - {model_id: gat_identity, model: gat, graph: identity,      params: from_search, seeds: [42, 43, 44]}
      - {model_id: gat_shuffled, model: gat, graph: shuffled,      params: from_search, seeds: [42]}
```
CLI (`python -m src.phase2.cli <cmd> --runs <yaml> [--only model_id[:seed]] [--device auto|cpu|mps|cuda] [--subset 0.05]`):

| cmd | does |
|---|---|
| `search` | trains each candidate (seed 42) on train, early-stops on validation, writes `<out>/search/<name>.csv` with validation MAE per horizon and the mean. Writes no test predictions. |
| `pick` | picks the candidate with the lowest mean validation MAE per group and writes `<out>/final/<name>.yaml` with concrete params |
| `final` | trains every final run and seed, writes validation and test predictions, `meta.json`, best checkpoint; exports attention if the model has `attention_maps` |
| `all` | `search`, `pick`, `final` in order; skips finished items, resumes interrupted ones |
| `smoke` | 2 train steps and one prediction pass for every model in the file on real data, on a tiny subset; checks shape, finiteness and gradients |

`<out>` is `$PHASE2_OUT` if set, else `data/processed/phase2`. On Colab it points to `Phase2_shared/out` on Drive. Controls (`gat_identity`, `gat_shuffled`, ...) reuse the tuned params of their group's main model. This clean ablation can slightly understate the control, and the paper says so.

### 4.7 Prediction files
`<out>/predictions/<model_id>/seed<S>_<split>.npz`, `split` in `{val, test}`:
`pred` float32 `[n_origins, 4, 207]` in source speed units; `origin_ns` int64 (origin timestamps, ascending); `sensor_ids` (string array in `sensors.csv` order). Same folder: `seed<S>_meta.json` with git commit, config hash, seed, device, torch version, parameter count, epochs run, best epoch, best mean validation MAE, train seconds, stop reason (`early_stop` / `max_epochs` / `time_cap`).
Origins and sensor order must equal the Phase 1 prediction parquets (6,844 test and 3,416 validation origins). Predictions must be finite. Targets are not stored; the harness takes them from the Phase 1 frames.
Attention: `<out>/attention/<model_id>_seed<S>.npz` with `attn` float16 `[12 origins, layers, heads, 207, 207]` and `origin_ns`. The 12 origins are the test origins nearest 08:00, 17:30 and 13:00 on each of the first four test weekdays.

### 4.8 Result files (M4 writes, M5 reads; schemas mirror Phase 1)
In `data/metadata/phase2/`:

| File | Columns |
|---|---|
| `metrics.csv` | `split,horizon,model,seed,period,n,mae,rmse,mape` with `horizon` in `5min/15min/30min/60min`, `period` in `all/peak/off_peak/morning_peak/evening_peak`. Per-seed rows have `seed` 42..44; Phase 1 models have `seed` empty. |
| `metrics_summary.csv` | `split,horizon,model,period,n_seeds,n,mae_mean,mae_sd,rmse_mean,rmse_sd,mape_mean,mape_sd,params,train_seconds_mean` |
| `bootstrap.csv` | same columns as Phase 1 `paired_bootstrap.csv` plus `period` |
| `sensor_metrics.csv`, `region_metrics.csv` | same columns as Phase 1 `sensor_forecast_metrics.csv` / `region_forecast_metrics.csv`, `model` = seed-averaged id |
| `hypotheses.csv` | `id,statement,horizon,period,comparison,mae_difference,ci95_low,ci95_high,verdict` |
| `events.csv` | congestion-onset precision/recall by model and horizon (proxy) |
| `signal_sim_results.csv` | M5 (see prompt) |

Figures in `figures/phase2/` (PNG, 300 dpi): `fig_horizon_mae`, `fig_peak_offpeak`, `fig_seed_spread`, `fig_graph_ablation`, `fig_map_{morning|evening}_{15|60}min` (observed, predicted, MAE for the best Phase 1 model and the best Phase 2 model, plus a difference map), `fig_congestion_map_{morning|evening}_{15}min`, `fig_attention_example`, `fig_case_study`, `fig_error_by_degree`, `fig_error_by_availability`, `fig_signal_*`.

### 4.9 Seeds, determinism, test lock
- Search uses seed 42. Final seeds are 42, 43, 44. Seed python, numpy, torch.
- Test predictions are written **only** by `final`, with configs produced by `pick` from validation. Search never touches test. Test metrics are produced only by M4's harness, never printed by the trainer. This makes the test lock structural, not a promise.
- Phase 1 models have one fitted seed. Never imply they have seed spread.

---

## 5. Experiment protocol (fixed before any Phase 2 result exists)

### 5.1 Model matrix
| Tier | Model id | Sees neighbours? | Owner |
|---|---|---|---|
| Phase 1 | `persistence`, `historical_average`, `random_forest_32`, `xgboost_32`, `xgboost_ha` | no | reused parquet |
| Deep, no graph | `gru_local` | no | M3 |
| Graph | `gat`, `dcrnn` | yes | M2, M3 |
| Controls | `gat_identity` (same architecture, self-loops only), `gat_shuffled` (wrong neighbours), `dcrnn_identity` | no / wrong | M2, M3 |
| Optional | `gat_random_matched` | wrong | M2 if time |

Phase 1's `xgboost_ha` uses historical-average features that the deep models do not get (no HA channel in Phase 2). The deep models do receive time-of-day and day-of-week channels. A deep-vs-`xgboost_ha` difference is therefore **not** a graph effect; only comparisons within the deep family isolate the graph.

### 5.2 Fair-comparison rules
Same split, origins, history length (12), horizons and valid-target mask for every model (the harness checks). Scaling fitted on training data only. Four candidates and the same early-stopping rule per family. Seeds 42/43/44 for `gat`, `gat_identity`, `gru_local`, `dcrnn`.

### 5.3 Hypotheses (do not edit after results exist)
| ID | Hypothesis | Test |
|---|---|---|
| H1 | `gat` has lower MAE than `gru_local` at 30 and 60 min | paired bootstrap, all origins |
| H2 | `gat` beats `gat_identity`; `gat_shuffled` is no better than `gat_identity` | paired bootstrap |
| H3 | The `gat` vs `gat_identity` gain is larger in peak than off-peak | gain by period with intervals |
| H4 | The gain grows with horizon (5 < 15 < 30 < 60 min) | gain by horizon |
| H5 | The gain is larger at sensors with more neighbours and around congestion onsets (proxy) | descriptive only |
| H0 | `gat` and `dcrnn` do not differ from `xgboost_ha` | paired bootstrap; report either direction |

**Verdict rule** (computed by code, not by hand): "A beats B" is *supported* at a horizon if the 95% interval of the MAE reduction (B minus A) lies entirely above 0, *contradicted* if entirely below 0, *inconclusive* if it contains 0. A hypothesis that needs several horizons is *supported* only if every named horizon is supported.

### 5.4 Statistics
Phase 1 method: circular moving-block bootstrap over stored calendar days, 2-day blocks, 2000 resamples, seed 42, all sensors of a day kept together. With several seeds, the per-day absolute-error sum is the mean over that model's seeds (this is the mean seed-wise MAE, exactly). Comparisons: `gat` vs {`gru_local`, `gat_identity`, `gat_shuffled`, `dcrnn`, `xgboost_ha`}, `dcrnn` vs {`gru_local`, `dcrnn_identity`, `xgboost_ha`}, at four horizons, all origins (32 intervals), plus peak and off-peak for `gat` vs `gat_identity` and `gat` vs `gru_local`. Intervals are conditional on this test period, not multiplicity-adjusted, and say nothing about other cities or seasons. Controls with one seed are noted as such.

### 5.5 Allowed and not allowed claims
Allowed: "On this chronological split of METR-LA, model X changes MAE by Y relative to Z, interval [a, b]." "In a synthetic simulation, policy P changed average delay by ...".
Not allowed: live or real-world control; attention weights as proof of causal or shockwave propagation; named freeways; mph or UTC; generalisation beyond this split; "graph is better" unless H1/H2 verdicts support it.

---

## 6. Timeline, gates, final assembly

No deadline was given, so this is in working days from the day the prompts are launched. Compress or cut the optional parts if the deadline is shorter (cut order: SUMO, `gat_random_matched`, error-by-availability, slides pptx).

| Stage | Days | What happens |
|---|---|---|
| A. Agents work | 0-2 | All five prompts run; M1 pushes the core first. Each member runs unit tests and local smoke runs on M3. |
| B. Colab training | 2-6 | M2 and M3 run `notebooks/phase2_colab_train.ipynb` (and helpers run extra seeds). Results land on Drive. |
| C. Assembly | 6-8 | Merge, score, signal sim, paper, second-person reproduction. |
| Buffer | 8-10 | Fixes. |

| Gate | Pass condition |
|---|---|
| G1 core | M1: parity test passes (persistence reproduces Phase 1 MAE 2.814 / 3.499 / 4.206 / 5.380 and valid counts 1,245,052 / 1,245,040 / 1,245,022 / 1,244,986); origin counts 23,967 / 3,416 / 6,844; leakage tests pass |
| G2 integration | `python -m src.phase2.cli smoke --runs configs/phase2/runs_gat.yaml` and the DCRNN/GRU files pass on M3 |
| G3 training | all required `final` runs and seeds exist in `Phase2_shared/out/predictions` |
| G4 evaluation | `scripts/phase2_eval_all.py` runs end to end and Phase 1 numbers are reproduced by the harness within 1e-3 |
| G5 paper | `scripts/build_phase2_paper.py` builds the PDF; claims checklist has no unlinked claim |
| G6 reproduction | a teammate who did not write the code follows `reports/phase2/reproduction.md` on a clean clone; `phase2_check_phase1_frozen.py` prints OK |

**Final assembly commands (humans, no prompts):**
1. Merge the pull requests in this order: M1, M4, M2, M3, M5.
2. `python -m src.phase2.cli smoke --runs configs/phase2/runs_gat.yaml` (and the two other run files).
3. M2 and M3 run the Colab notebook (open it, set the run file, "Run all"; repeat after any disconnect).
4. Download `Phase2_shared/out/predictions` into `data/processed/phase2/predictions/`; copy `out/final/*.yaml` into `configs/phase2/final/`; copy the best checkpoints into `models/phase2/`; commit.
5. `python scripts/phase2_eval_all.py`
6. `python scripts/phase2_run_signal_sim.py`
7. `python scripts/build_phase2_paper.py`
8. `python scripts/phase2_check_phase1_frozen.py` and `python -m unittest discover -s tests -p "test_phase2_*.py" -v`

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| Graph models do not beat the strong Phase 1 trees or the non-graph deep control | Hypotheses and verdict rule fixed up front; negative results reported plainly |
| Accidentally changing Phase 1 outputs inside the shared repo | Tag `phase1-final`, frozen check script, "do not run" list, new paths only |
| Colab disconnects or no GPU | Per-epoch Drive checkpoints, resume, wall-clock caps, `--only` for splitting seeds across accounts |
| Unfair comparison | One pipeline built on Phase 1 loaders; parity gate; one scorer (M4) |
| Test-set tuning | Structural lock (4.9) |
| 8 GB RAM on M3 | GPU/accelerator-resident (T,N,C) tensor, no materialised windows, harness works on `[origins, 4, 207]` arrays |
| Integration mismatch between independently written pieces | Contract in section 4, `smoke` command, merge order |
| Large derived files not in git | Drive hand-off, hashes checked at kickoff |
| Over-claiming | Section 5.5, M5 claims checklist and wording scan |
| SUMO unavailable | Python simulator is the primary implementation; SUMO is an optional adapter capped at about 1 hour of effort |

---

## 8. The five prompts

Paste exactly one block into your own agent session. Each begins with the same ground rules, so the block is self-contained apart from this plan file.

### 8.1 Member 1: core pipeline, trainer, CLI, Colab runner

```text
Work on branch `phase2/m1-core`. Read `reports/phase2_plan.md` sections 2–4 first.
Rules: Phase 1 is frozen (never edit Phase 1 files except appending `.gitignore`, appending `README.md`, and adding `requirements-phase2.txt`).
Deliverables:
1. `src/phase2/{__init__,data,graph,registry,train,cli}.py` and `src/phase2/models/__init__.py` implementing the Section 4 contract (causal fill + train-only scaler, 5 graph variants in `load_graph`, `@register_model` auto-import, per-epoch checkpointing + wall-clock cap, masked MAE loss in source speed units, `smoke` / `search` / `pick` / `final` CLI commands).
2. `scripts/phase2_check_phase1_frozen.py` verifying zero Phase 1 modifications since tag `phase1-final`.
3. `notebooks/phase2_colab_train.ipynb` for Colab T4 training with Drive checkpointing and resume.
4. `tests/test_phase2_core.py` covering persistence parity, data leakage, graph direction/variants, and CLI smoke run.
Verify with `python -m unittest discover -s tests -p "test_phase2_*.py" -v` and `python scripts/phase2_check_phase1_frozen.py`.
```

### 8.2 Member 2: Graph Attention Network

```text
Work on branch `phase2/m2-gat`. Read `reports/phase2_plan.md` sections 2–4 first.
Rules: Phase 1 is frozen; pure PyTorch only (no PyTorch Geometric); never write test predictions during search.
Deliverables:
1. `src/phase2/models/gat.py` registered as `"gat"` (`needs_graph = True`), with shared GRU/Conv temporal encoder, masked multi-head attention (`A[i,j] > 0` + self-loops, `-1e9` mask), residual head, and `attention_maps(x, adj)` returning `[B, heads, N, N]` per layer.
2. `configs/phase2/runs_gat.yaml` with 4 search candidates (`seed: 42`, `bidirectional`) and final runs (`gat` x3 seeds, `gat_identity` x3 seeds, `gat_shuffled` x1 seed).
3. `scripts/phase2_gat_attention_stats.py` computing per-sensor entropy and self/upstream/downstream attention mass split by peak vs off-peak (`data/metadata/phase2/attention_stats.csv`).
4. `reports/phase2/fact_sheet_gat.md` documenting architecture, exact parameter counts, and per-candidate runtime estimates.
5. `tests/test_phase2_gat.py` covering registration, shape/finiteness, softmax masking, permutation equivariance, identity-graph isolation, connected-neighbour influence, overfit, exact parameter counts, and attention stats.
Verify with `python -m unittest discover -s tests -p "test_phase2_*.py" -v` and `python scripts/phase2_check_phase1_frozen.py`.
```

### 8.3 Member 3: DCRNN and non-graph deep controls

```text
Work on branch `phase2/m3-dcrnn-controls`. Read `reports/phase2_plan.md` sections 2–4 first.
Rules: Phase 1 is frozen; pure PyTorch only; never edit shared registry files (use `@register_model`).
Deliverables:
1. `src/phase2/models/controls.py` implementing `gru_local` (`needs_graph = False`, shared per-sensor GRU with zero spatial mixing).
2. `src/phase2/models/dcrnn.py` implementing `dcrnn` (`needs_graph = True`, DiffusionConv with K-hop forward `D_out^-1 A` and reverse `D_in^-1 A^T` random-walk supports + DCGRU encoder-decoder).
3. `configs/phase2/runs_gru_local.yaml` (4 search candidates + `gru_local` x3 seeds) and `configs/phase2/runs_dcrnn.yaml` (4 search candidates + `dcrnn` x3 seeds + `dcrnn_identity` x1 seed).
4. `reports/phase2/fact_sheet_dcrnn_controls.md` with architecture, parameter counts, and runtime caps.
5. `tests/test_phase2_controls.py` and `tests/test_phase2_dcrnn.py` verifying neighbour independence (`gru_local`), diffusion supports, equivariance, and overfit.
Verify with `python -m unittest discover -s tests -p "test_phase2_*.py" -v` and `python scripts/phase2_check_phase1_frozen.py`.
```

### 8.4 Member 4: evaluation, statistics, maps, error analysis

```text
Work on branch `phase2/m4-evaluation`. Read `reports/phase2_plan.md` sections 2, 4, and 5 first.
Rules: Phase 1 is frozen; reuse Phase 1 `compute_metrics`, `temporal_masks`, `paired_daily_bootstrap`, `region_scores`, and `spatial_mapping` without modifying Phase 1 files.
Deliverables:
1. `src/phase2/evaluation/` (`scoring.py`, `bootstrap.py`, `hypotheses.py`) for unified MAE/RMSE/MAPE scoring (overall and peak/off-peak), paired 2-day block bootstrap (`2000` resamples, seed `42`), and H1–H5 hypothesis verdicts.
2. `src/phase2/viz/` for horizon MAE comparisons, peak vs off-peak breakdowns, attention maps, and sensor error maps saved to `figures/phase2/`.
3. `scripts/phase2_eval_all.py` writing tracked summary CSVs to `data/metadata/phase2/{metrics,bootstrap,hypotheses}.csv`.
4. `notebooks/phase2_error_analysis.ipynb` and `reports/phase2/fact_sheet_evaluation.md`.
5. `tests/test_phase2_evaluation.py` including a Phase 1 parity check against saved Phase 1 predictions.
Verify with `python -m unittest discover -s tests -p "test_phase2_*.py" -v` and `python scripts/phase2_check_phase1_frozen.py`.
```

### 8.5 Member 5: simulated signal control, paper, slides, QA

```text
Work on branch `phase2/m5-signal-paper`. Read `reports/phase2_plan.md` sections 2, 4, and 5 first.
Rules: Phase 1 is frozen; label every signal experiment and congestion metric as simulated/proxy; never claim live traffic control.
Deliverables:
1. `src/phase2/signal_sim/` implementing a reproducible Python simulated signal-policy evaluation driven by Phase 2 forecasts, outputting tracked CSV/JSON to `data/metadata/phase2/` and figures to `figures/phase2/`.
2. `reports/phase2/` and `reports/paper_phase2/` assembling the Phase 2 technical report / IEEE paper and presentation slides using M2/M3/M4 fact sheets and M4 evaluation tables.
3. Claims/wording QA check verifying "source speed units", "stored clock", "proxy congestion", and "simulated" terminology, plus `tests/test_phase2_signal_sim.py`.
Verify with `python -m unittest discover -s tests -p "test_phase2_*.py" -v` and `python scripts/phase2_check_phase1_frozen.py`.
```
