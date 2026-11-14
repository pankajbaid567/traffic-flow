# Phase 2 Cross-Team Handoff & Action Items

**Author:** Person E (Ranvendra Pratap Singh / Member 5)  
**Date:** 2026-11-14  
**Git Branch:** `main` (Commit: `2220175`)  
**Status:** Person E (Member 5) complete and merged. Handing off to Members 1, 2, 3, and 4 for training completion, final Colab weights synchronization, and test-set rescoring.

---

## 1. Executive Team Status Matrix

| Member | Person | Core Responsibility | PR Status | Local Code Status | Action Items Remaining |
|---|---|---|---|---|---|
| **Member 1** | Pankaj Baid | Core pipeline, trainer, CLI, Colab runner | PR #1 merged | Passed | Sync raw data on Google Drive; pull `main`. |
| **Member 2** | Ashutosh Singh | GAT model, attention stats, GAT fact sheet | PR #3 merged | Passed | Pull `main`; run Colab `runs_gat.yaml`; upload `.npz` & checkpoints. |
| **Member 3** | Narendra Singh | DCRNN, `gru_local`, DCGRU controls | PR #2 merged | Passed | Pull `main`; run Colab `runs_dcrnn.yaml` & `runs_gru_local.yaml`; upload `.npz`. |
| **Member 4** | Shreya Suman | Scorer, bootstrap, hypothesis testing, maps | PR #4 merged | Passed | Pull `main`; run `phase2_eval_all.py` on downloaded `.npz`; run error notebook. |
| **Member 5** | Ranvendra P. Singh | Signal simulation, IEEE paper, QA checklist | Merged to `main` | **100% COMPLETE** | Author handoff; lead viva defense preparation. |

---

## 2. Handoff to Member 1 (Pankaj Baid — Repo Owner & Core Pipeline Lead)

### What is Completed:
- Core data tensors (`data.py`), graph variants (`graph.py`), registry (`registry.py`), checkpointing engine (`train.py`), CLI commands (`cli.py`), and frozen check script (`phase2_check_phase1_frozen.py`).

### Your Next Action Items:
1. **Pull Latest Main:**
   ```bash
   git checkout main
   git pull origin main
   ```
2. **Google Drive Shared Assets Verification:**
   Ensure `Phase2_shared/inputs/` contains:
   - `data/raw/metr-la.h5` (SHA-256: `64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05`)
   - `data/raw/adj_mx.pkl` (SHA-256: `a35687c6e15aa228dc45027b0ed2a0ea0f4ec78f573deb992c595092d12f61b3`)
   - `phase1_predictions.zip` (182 MB, four Phase 1 parquet files)
3. **Colab Account Allocation:**
   If teammates hit Colab free-tier GPU timeouts (T4 limits), assist by dividing seeds across accounts using the CLI flag:
   `python -m src.phase2.cli final --runs configs/phase2/runs_gat.yaml --only gat:44`

---

## 3. Handoff to Member 2 (Ashutosh Singh — GAT Lead)

### What is Completed:
- GAT model architecture (`src/phase2/models/gat.py`), multi-head attention maps, and search configuration (`configs/phase2/runs_gat.yaml`).

### Your Next Action Items:
1. **Pull Latest Main:**
   *Note:* Person E fixed a timestamp parsing inference bug in `scripts/phase2_gat_attention_stats.py` and `tests/test_phase2_gat.py` (commit `128604b`). Pull `main` to ensure your local environment passes all 9 tests.
   ```bash
   git checkout main && git pull origin main
   python -m unittest tests/test_phase2_gat.py -v
   ```
2. **Execute Colab Training for GAT:**
   - Open `notebooks/phase2_colab_train.ipynb` on Google Colab with GPU T4 enabled.
   - Run `runs_gat.yaml`:
     - Stage 1 (`search`): 4 search candidates on validation data (seed 42).
     - Stage 2 (`pick`): Automatically selects candidate with lowest validation MAE.
     - Stage 3 (`final`): Evaluates seeds `[42, 43, 44]` for `gat`, seeds `[42, 43, 44]` for `gat_identity`, and seed `42` for `gat_shuffled`.
3. **Synchronize Outputs to Shared Drive:**
   - Save predictions: `Phase2_shared/out/predictions/gat/seed{42,43,44}_{val,test}.npz`
   - Save attention maps: `Phase2_shared/out/attention/gat_seed42.npz`
   - Copy best model checkpoints (< 10 MB each) into `models/phase2/` and commit.

---

## 4. Handoff to Member 3 (Narendra Singh — DCRNN & Controls Lead)

### What is Completed:
- `dcrnn` architecture (`src/phase2/models/dcrnn.py`), `gru_local` non-graph control (`src/phase2/models/controls.py`), and configs (`runs_dcrnn.yaml`, `runs_gru_local.yaml`).

### Your Next Action Items:
1. **Pull Latest Main:**
   ```bash
   git checkout main && git pull origin main
   python -m unittest tests/test_phase2_controls.py tests/test_phase2_dcrnn.py -v
   ```
2. **Execute Colab Training for DCRNN and GRU Local:**
   - In `notebooks/phase2_colab_train.ipynb`, run:
     - `RUN_FILE = "configs/phase2/runs_gru_local.yaml"`: Search 4 candidates -> pick -> final seeds `[42, 43, 44]`.
     - `RUN_FILE = "configs/phase2/runs_dcrnn.yaml"`: Search 4 candidates -> pick -> final seeds `[42, 43, 44]`, plus `dcrnn_identity` seed `42`.
3. **Synchronize Outputs to Shared Drive:**
   - Save predictions to `Phase2_shared/out/predictions/dcrnn/` and `Phase2_shared/out/predictions/gru_local/`.
   - Copy best checkpoints into `models/phase2/` and commit.

---

## 5. Handoff to Member 4 (Shreya Suman — Evaluation & Statistics Lead)

### What is Completed:
- Scorer (`scoring.py`), loaders (`loaders.py`), circular moving-block bootstrap (`bootstrap.py`), events detection (`events.py`), and visualization suite (`plots.py`).

### Your Next Action Items:
1. **Pull Latest Main:**
   *Note:* Person E added the `sys.path` bootstrap in `scripts/phase2_eval_all.py` (commit `b9f677b`).
   ```bash
   git checkout main && git pull origin main
   ```
2. **Download Prediction Files from Drive:**
   Once M2 and M3 upload predictions to Drive `Phase2_shared/out/predictions/`:
   ```bash
   mkdir -p data/processed/phase2/predictions
   cp -r /path/to/Phase2_shared/out/predictions/* data/processed/phase2/predictions/
   ```
3. **Run Master Evaluation Runner:**
   ```bash
   python scripts/phase2_eval_all.py
   ```
   - Check `data/metadata/phase2/hypotheses.csv` to ensure verdicts for H1–H5 and H0 match expected theoretical outcomes.
   - Verify that test-set metrics populate `metrics.csv` and `metrics_summary.csv`.
4. **Execute Error Analysis Notebook:**
   ```bash
   jupyter execute notebooks/phase2_error_analysis.ipynb
   ```
   Verify spatial comparison maps and congestion maps render cleanly in `figures/phase2/`.

---

## 6. Member 5 Delivery Summary (Ranvendra Pratap Singh)

Member 5's deliverables are 100% complete, fully tested, and committed to `main` (commits `128604b` through `2220175`):
- `src/phase2/signal_sim/` & `scripts/phase2_run_signal_sim.py`: Adaptive traffic signal queue simulation engine across 80 runs (5 seeds, 4 regimes).
- `data/metadata/phase2/signal_sim_results.csv` & `signal_sim_summary.json`: Serialized simulation evidence.
- `figures/phase2/`: 4 publication figures (`fig_case_study.png`, `fig_signal_delay_comparison.png`, `fig_signal_queue_dynamics.png`, `fig_signal_spillback_reduction.png`).
- `reports/paper_phase2/`: Complete camera-ready IEEE paper PDF (`phase2_ieee_paper.pdf`, 1.5 MB), LaTeX source (`phase2_ieee_paper.tex`), source zip, and QA report.
- `reports/phase2/qa_claims_checklist.md`: 100% compliant with no unverified claims.
- `reports/phase2/reproduction.md`: 5-minute clean clone reproduction guide.
- `reports/person_e_phase2_report.md`: Capstone synthesis report and master viva defense guide.

All unit tests pass (96 tests), and Phase 1 files remain strictly frozen (`phase2_check_phase1_frozen.py` prints OK).
