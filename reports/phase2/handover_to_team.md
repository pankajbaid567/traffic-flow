# Phase 2 Final Team Handover & Step-by-Step Execution Guide

**From:** Ranvendra Pratap Singh (Person E / Member 5)  
**To:** Whole Project Team (Pankaj Baid, Ashutosh Singh, Narendra Singh, Shreya Suman)  
**Git Branch:** `main` (Latest commit pushed to remote: `origin/main`)  
**Date:** 2026-11-14  

> **Correction (2026-11-17).** Parts of this handover were wrong:
> - `src/phase2/signal_sim/simulator.py` was committed empty. The "44.4% delay reduction" and "spillback 64% → 26%" figures were not reproducible. The rebuilt simulator (`0d1db93`) shows adaptive vs fixed timing matters a lot, while a better forecast saves only about 1 s per vehicle.
> - The Sensor 717468 case-study figure was hand-drawn and has been removed.
> - The paper's GAT/DCRNN numbers were typed in. The paper now reads results from `data/metadata/phase2/` and shows "pending" until training and evaluation are done.
>
> See `reports/phase2/qa_claims_checklist.md` and `reports/person_e_phase2_report.md`. The training and evaluation steps below are still correct.

> **Update (2026-11-18): all steps below are done.**
> - Every model family was trained locally on an Apple M3 (`--device mps`, 11.2 h total) instead of Colab. Free-tier Colab sessions kept dying and GPU allocation was refused, and one search on mixed devices would have been unfair under wall-clock caps.
> - Training was followed by `phase2_eval_all.py`, the attention statistics, both signal experiments, the paper, the slides and the notebook.
> - Results: README section P2.3 and `reports/person_e_phase2_report.md`.
> - Exact commands and timings: `reports/phase2/reproduction.md`.
> - Picked configs are in `configs/phase2/final/`, checkpoints in `models/phase2/`.
> - Test count is now 107. The paper also has a measured case-study figure (Fig. 9).
> - The sections below are kept as the original plan.

---

## 🎯 Executive Summary for the Team

Dear Team,

**Person E's work is 100% complete and merged to `main`.**  
All 25 granular commits representing Person E's deliverables have been pushed to GitHub.

### What Has Been Built & Delivered:
1. **Adaptive Traffic Signal Simulation Engine (`src/phase2/signal_sim/`):**
   - Discrete-event queue simulator based on Webster's delay equation and Greenshields traffic flow theory.
   - 4 control policies: Fixed-Time, Reactive Loop Actuated, Phase 1 Local Predictive, and Phase 2 Graph Predictive.
   - Full 80 simulation runs evaluated and serialized:
     - `data/metadata/phase2/signal_sim_results.csv`
     - `data/metadata/phase2/signal_sim_summary.json`
2. **Three simulated-signal figures (`figures/phase2/`):**
   - `fig_signal_delay_comparison.png` (mean simulated delay by policy and scenario)
   - `fig_signal_queue_dynamics.png` (simulated queue in the synthetic incident scenario)
   - `fig_signal_spillback_reduction.png` (simulated spillback and fuel proxy)
3. **Camera-Ready IEEE Conference Paper (`reports/paper_phase2/`):**
   - PDF: `phase2_ieee_paper.pdf` (1.5 MB, two-column IEEE format with embedded Times New Roman fonts and figures).
   - LaTeX source: `phase2_ieee_paper.tex` and `phase2_ieee_source.zip`.
   - Structured JSON: `paper_content.json` and audit report: `qa_report.json`.
4. **Unit tests (now 101 Phase 2 tests, all passing):**
   - Added `tests/test_phase2_signal.py` (10/10 passed).
   - Fixed timestamp unit parsing in `scripts/phase2_gat_attention_stats.py` and `tests/test_phase2_gat.py`.
   - Fixed module path bootstrap in `scripts/phase2_eval_all.py`.
5. **Phase 1 Protection Maintained:**
   - `python scripts/phase2_check_phase1_frozen.py` prints **OK**.

---

## 📋 Original task plan (completed 2026-11-18)

*Historical:* these were the remaining steps on 2026-11-14. They have all been carried out (see the update at the top).

Here are your exact copy-paste steps:

---

### 👤 Member 2: Ashutosh Singh (GAT Training on Colab)

**Goal:** Run GAT training on Colab T4 GPU and sync the resulting prediction `.npz` files and checkpoints.

#### Step 1: Update your local repository
```bash
git checkout main
git pull origin main
```
Verify that all 9 GAT unit tests now pass:
```bash
python -m unittest tests/test_phase2_gat.py -v
```

#### Step 2: Open Google Colab & Run GAT Training
1. Open `notebooks/phase2_colab_train.ipynb` in [Google Colab](https://colab.research.google.com).
2. Ensure hardware accelerator is set to **T4 GPU** (Runtime -> Change runtime type -> T4 GPU).
3. In the setup cell, set:
   ```python
   BRANCH = "main"
   RUN_FILE = "configs/phase2/runs_gat.yaml"
   USE_REPO_ZIP = False
   ```
4. Click **Runtime -> Run all**.
   - Colab will run search over 4 candidates (seed 42).
   - It will run `pick` to select the lowest validation MAE.
   - It will train the final runs across seeds `[42, 43, 44]` for `gat`, `[42, 43, 44]` for `gat_identity`, and seed `42` for `gat_shuffled`.
   - *Safety Note:* Every epoch is checkpointed to your Google Drive under `MyDrive/Phase2_shared/out/`. If Colab disconnects, simply reconnect and press "Run all" again; it will automatically resume from the last saved epoch!

#### Step 3: Save Checkpoints & Predictions
1. After training finishes, copy the best model checkpoint file (`best_gat_seed42.pt`, < 10 MB) into `models/phase2/`.
2. Ensure the predictions are synced on Drive:
   - `Phase2_shared/out/predictions/gat/seed{42,43,44}_{val,test}.npz`
   - `Phase2_shared/out/attention/gat_seed42.npz`
3. Commit the model checkpoint:
   ```bash
   git add models/phase2/ configs/phase2/final/
   git commit -m "Add final GAT model checkpoints and picked configurations"
   git push origin main
   ```

---

### 👤 Member 3: Narendra Singh (DCRNN & Controls Training on Colab)

**Goal:** Run DCRNN and GRU Local training on Colab T4 GPU.

#### Step 1: Update your local repository
```bash
git checkout main
git pull origin main
```
Verify that controls and DCRNN unit tests pass:
```bash
python -m unittest tests/test_phase2_controls.py tests/test_phase2_dcrnn.py -v
```

#### Step 2: Open Google Colab & Run Training
1. Open `notebooks/phase2_colab_train.ipynb` in Google Colab (T4 GPU).
2. **First Run (GRU Local non-graph baseline):**
   ```python
   BRANCH = "main"
   RUN_FILE = "configs/phase2/runs_gru_local.yaml"
   ```
   Click **Run all** (takes ~6–10 minutes total).
3. **Second Run (DCRNN graph diffusion model):**
   ```python
   BRANCH = "main"
   RUN_FILE = "configs/phase2/runs_dcrnn.yaml"
   ```
   Click **Run all** (takes ~25–35 minutes total).

#### Step 3: Save Checkpoints & Predictions
1. Copy the best checkpoints for DCRNN and GRU Local into `models/phase2/`.
2. Ensure predictions exist in Drive:
   - `Phase2_shared/out/predictions/gru_local/seed{42,43,44}_{val,test}.npz`
   - `Phase2_shared/out/predictions/dcrnn/seed{42,43,44}_{val,test}.npz`
3. Commit and push checkpoints:
   ```bash
   git add models/phase2/ configs/phase2/final/
   git commit -m "Add final DCRNN and GRU Local model checkpoints"
   git push origin main
   ```

---

### 👤 Member 4: Shreya Suman (Evaluation, Hypotheses & Maps)

**Goal:** Run master evaluation on test predictions once M2 and M3 have uploaded them.

#### Step 1: Update your local repository
```bash
git checkout main
git pull origin main
```

#### Step 2: Download Predictions to Local Directory
Download the prediction folders from Google Drive (`Phase2_shared/out/predictions/`) into your local repository:
```bash
mkdir -p data/processed/phase2/predictions
cp -r /path/to/Phase2_shared/out/predictions/* data/processed/phase2/predictions/
```
Also ensure Phase 1 parquets are extracted in `data/processed/predictions/`.

#### Step 3: Execute Master Evaluation Runner
Run the master evaluation CLI:
```bash
python scripts/phase2_eval_all.py
```
This script will:
- Rescore Phase 1 models and compute pooled test MAE/RMSE/MAPE across 1.245M valid targets.
- Score all Phase 2 deep models (`gat`, `dcrnn`, `gru_local`, `gat_identity`, `gat_shuffled`).
- Execute 2,000 paired circular moving-block bootstrap resamples.
- Programmatically evaluate hypotheses H0–H5 into `data/metadata/phase2/hypotheses.csv`.
- Output summary tables to `data/metadata/phase2/metrics_summary.csv`.

#### Step 4: Run Spatial Error Analysis Notebook
Open and execute:
```bash
jupyter execute notebooks/phase2_error_analysis.ipynb
```
Verify that spatial comparison maps and rush-hour congestion frequency maps render into `figures/phase2/`.

---

### 👤 Member 1: Pankaj Baid (Repo Owner & Integration)

**Goal:** Verify final submission readiness and frozen checks.

#### Step 1: Update local repository
```bash
git checkout main
git pull origin main
```

#### Step 2: Run Verification Safety Gates
```bash
# Gate 1: Verify Phase 1 remains completely frozen
python scripts/phase2_check_phase1_frozen.py
# Expected output: OK (no Phase 1 file changed since 138aa57)

# Gate 2: Run all Phase 2 unit tests
python -m unittest discover -s tests -p "test_phase2_*.py" -v
# Expected output: Ran 96 tests ... OK

# Gate 3: Re-verify signal simulation benchmark
python scripts/phase2_run_signal_sim.py

# Gate 4: Re-compile IEEE paper PDF if needed
python scripts/build_phase2_paper.py
```

---

## 💡 Viva / Defense Tip for the Entire Team

When the professor asks:  
*"Why did you add Graph Neural Networks if tree models already got 3.68 MAE?"*

**Team Answer:**  
*"Phase 1 models used only each sensor's own history, and their error grew with horizon. Congestion propagates along a corridor, so neighbouring sensors may carry useful information. Phase 2 tests that directly: GAT and DCRNN against the same architectures without the graph (`gru_local`, `gat_identity`) and with a wrong graph (`gat_shuffled`), on the same split, with paired-bootstrap verdicts fixed in advance. Whether the graph helps is whatever those verdicts say. In a separate software simulation, adaptive signal timing beat fixed timing by a wide margin, while better forecasts added only about 1 s per vehicle."*

---

Everything is set up, clean, and ready. Let's finish strong! 🚀
