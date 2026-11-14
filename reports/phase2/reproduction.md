# Phase 2 Reproduction Guide

**Target Audience:** Teammates, Examiners, and Peer Reviewers  
**Repository:** `traffic-flow` (Branch: `main` or `phase2/m5-signal-paper`)  
**Hardware Tested:** Apple Silicon M3 (8 GB RAM) and Google Colab Free Tier (Nvidia T4 GPU)  
**Python Version:** Python 3.10+ (Colab) / Python 3.12 (Local)

---

## 1. Quickstart: 5-Minute Evaluation & Reproduction

To reproduce all Phase 2 evaluation metrics, simulated signal benchmarks, figures, and IEEE paper from a clean clone:

### Step 1: Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/pankajbaid567/traffic-flow.git
cd traffic-flow
git checkout phase2/m5-signal-paper

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-phase2.txt
pip install reportlab
```

### Step 2: Verify Phase 1 Freezing Invariant
Phase 1 code and artifacts must remain 100% frozen:
```bash
python scripts/phase2_check_phase1_frozen.py
# Expected output: OK (no Phase 1 file changed since 138aa57)
```

### Step 3: Run Full Unit Test Suite (96 Tests)
```bash
python -m unittest discover -s tests -p "test_phase2_*.py" -v
# Expected output: Ran 96 tests ... OK (skipped=17 if raw HDF5 not downloaded)
```

### Step 4: Execute Simulated Adaptive Traffic Signal Benchmark
Runs the discrete-event queue simulation across 80 runs (5 seeds, 4 regimes, 4 control policies):
```bash
python scripts/phase2_run_signal_sim.py
# Outputs:
# - data/metadata/phase2/signal_sim_results.csv
# - data/metadata/phase2/signal_sim_summary.json
# - figures/phase2/fig_signal_delay_comparison.png
# - figures/phase2/fig_signal_queue_dynamics.png
# - figures/phase2/fig_signal_spillback_reduction.png
# - figures/phase2/fig_case_study.png
```

### Step 5: Build Phase 2 IEEE Research Paper & Camera-Ready PDF
Compiles both the official IEEEtran LaTeX source and the ReportLab camera-ready PDF:
```bash
python scripts/build_phase2_paper.py
# Outputs:
# - reports/paper_phase2/phase2_ieee_paper.pdf
# - reports/paper_phase2/phase2_ieee_paper.tex
# - reports/paper_phase2/paper_content.json
# - reports/paper_phase2/qa_report.json
```

---

## 2. Full Pipeline Reproduction (With Raw Data & Colab Training)

If training the deep learning models from scratch or reproducing the full evaluation harness on test tensors:

### Step A: Download Shared Data Assets (Google Drive)
From the team's shared Drive folder `Phase2_shared/`:
1. Copy `inputs/metr-la.h5` and `inputs/adj_mx.pkl` to `data/raw/`.
   - Verify SHA-256 hashes:
     ```bash
     shasum -a 256 data/raw/metr-la.h5
     # Expected: 64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05
     shasum -a 256 data/raw/adj_mx.pkl
     # Expected: a35687c6e15aa228dc45027b0ed2a0ea0f4ec78f573deb992c595092d12f61b3
     ```
2. Unzip `inputs/phase1_predictions.zip` into `data/processed/predictions/`.
3. If using pre-trained Phase 2 models, copy `out/predictions/*` into `data/processed/phase2/predictions/`.

### Step B: Train Deep Models on Google Colab (Free T4 GPU)
1. Open `notebooks/phase2_colab_train.ipynb` in Google Colab.
2. Select Runtime -> Change runtime type -> **T4 GPU**.
3. Set configuration:
   ```python
   BRANCH = "phase2/m5-signal-paper"
   RUN_FILE = "configs/phase2/runs_gat.yaml"  # or runs_dcrnn.yaml, runs_gru_local.yaml
   ```
4. Click **Run All**. Checkpoints and predictions are automatically synchronized to your mounted Google Drive.
   - Search phase: ~15–20 minutes across 4 candidates.
   - Final evaluation: ~30–45 minutes across 3 random seeds.

### Step C: Execute Master Evaluation & Hypothesis Verification
```bash
python scripts/phase2_eval_all.py
# Computes pooled MAE/RMSE/MAPE across 1.245M valid test targets.
# Evaluates hypotheses H0–H5 via 2,000 paired circular moving-block bootstrap resamples.
```

---

## 3. Directory Layout & Artifact Registry

```
AML&DL_Project/
├── configs/phase2/             # Run configurations for GAT, DCRNN, GRU Local
├── data/
│   ├── metadata/phase2/        # Tracked CSV/JSON metrics and signal results
│   │   ├── signal_sim_results.csv
│   │   └── signal_sim_summary.json
│   └── raw/                    # Raw METR-LA HDF5 and adjacency matrix
├── figures/phase2/             # 300 DPI publication figures
│   ├── fig_case_study.png
│   ├── fig_signal_delay_comparison.png
│   ├── fig_signal_queue_dynamics.png
│   └── fig_signal_spillback_reduction.png
├── reports/
│   ├── paper_phase2/           # IEEE camera-ready PDF, LaTeX source, and figures
│   │   ├── phase2_ieee_paper.pdf
│   │   ├── phase2_ieee_paper.tex
│   │   └── paper_content.json
│   └── phase2/                 # Architecture fact sheets, QA checklist, pipeline docs
├── scripts/
│   ├── build_phase2_paper.py   # Paper compilation script
│   ├── phase2_check_phase1_frozen.py # Safety check
│   ├── phase2_eval_all.py      # Master evaluation harness
│   └── phase2_run_signal_sim.py# Signal simulation runner
├── src/phase2/
│   ├── models/                 # GAT, DCRNN, GRU Local architectures
│   └── signal_sim/             # Simulated Adaptive Signal Control Engine
└── tests/
    └── test_phase2_*.py        # Complete unit testing suite (96 tests)
```

