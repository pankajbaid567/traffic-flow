# Phase 2 reproduction guide

Every number in the Phase 2 paper, slides and notebook is read from files written by the
scripts below. Nothing is typed in by hand.

- Repository: `traffic-flow`, branch `main`
- Python 3.12 (Mac) / 3.10+ (Colab)
- Hardware used for the submitted results: Apple M3 (MPS) for every search candidate and
  every final run (`run_meta.csv`, `device` column). Earlier Colab attempts were discarded,
  so all candidates in a family had the same device and the same wall-clock cap.

## 1. From a clean clone, without retraining (about 5 minutes)

The tracked result tables in `data/metadata/phase2/` and figures in `figures/phase2/` are
enough to rebuild the paper, slides and notebook.

```bash
git clone https://github.com/pankajbaid567/traffic-flow.git
cd traffic-flow
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-phase2.txt

python scripts/phase2_check_phase1_frozen.py          # Phase 1 files unchanged
python -m unittest discover -s tests -p "test_phase2_*.py"
python scripts/build_phase2_paper.py                  # PDF, .tex, paper_content.json, qa_report.json
python scripts/build_phase2_slides.py                 # reports/paper_phase2/phase2_slides.pptx
python scripts/build_phase2_error_analysis_notebook.py --execute
```

Tests that need the raw HDF5 file or the Phase 1 prediction files skip with a message when
those files are absent.

## 2. Full pipeline (raw data, training, evaluation)

### 2.1 Data

Copy `metr-la.h5` and `adj_mx.pkl` into `data/raw/` and check their hashes:

```bash
shasum -a 256 data/raw/metr-la.h5   # 64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05
shasum -a 256 data/raw/adj_mx.pkl   # a35687c6e15aa228dc45027b0ed2a0ea0f4ec78f573deb992c595092d12f61b3
```

Unzip the Phase 1 prediction files into `data/processed/predictions/`. The scorer reproduces
Phase 1 from them, and the run stops if it does not.

### 2.2 Training

One command per model family runs search (4 candidates, validation only), pick, and the
final seeds. Re-running it skips finished work and resumes interrupted runs from `last.pt`;
the wall-clock cap counts time across resumes.

```bash
python -m src.phase2.cli all --runs configs/phase2/runs_gru_local.yaml --device mps   # or cuda
python -m src.phase2.cli all --runs configs/phase2/runs_gat.yaml       --device mps
python -m src.phase2.cli all --runs configs/phase2/runs_dcrnn.yaml     --device mps
```

On Colab, use `notebooks/phase2_colab_train.ipynb` (set `RUN_FILE`). Outputs go to
`data/processed/phase2/` (git-ignored):

- `predictions/<model_id>/seed<S>_{val,test}.npz` and `seed<S>_meta.json`
- `attention/gat*_seed<S>.npz`
- `final/<family>.yaml` (the picked candidate)
- `best/<model_id>_seed<S>.pt`

The picked configurations are copied to `configs/phase2/final/`, the best-validation
checkpoints to `models/phase2/` (GRU/GAT 77-96 KB, DCRNN 528 KB each) and the search tables
to `data/metadata/phase2/search_<family>.csv`.

Measured training time for the submitted runs (Apple M3, MPS, sum of `train_seconds`):

| Family | Search (4 candidates) | Final runs | Total |
|---|---|---|---|
| GRU Local | 56 min (cap 15 min each) | 53 min (3 runs, cap 30 min) | 1.8 h |
| GAT (+ identity, shuffled) | 74 min (cap 20 min each) | 170 min (7 runs, cap 40 min) | 4.1 h |
| DCRNN (+ identity) | 121 min (cap 30 min each) | 201 min (4 runs, cap 50 min) | 5.4 h |
| **All** | | | **11.2 h** |

Approximate seconds per epoch on the M3: GRU Local 19 s, GAT 26 s, DCRNN about 94 s
(hidden 32) to about 215 s (hidden 64, the picked candidate). All DCRNN runs stopped at
their wall-clock cap (6-19 search epochs, 14 final epochs) and the GAT finals at the
60-epoch limit, so both may be under-trained; the paper states this as a limitation.

### 2.3 Evaluation, signal simulation, paper

```bash
python scripts/phase2_eval_all.py              # metrics, bootstrap, hypotheses, maps, events, figures
python scripts/phase2_gat_attention_stats.py   # attention_stats.csv
python scripts/phase2_run_signal_sim.py        # synthetic + real-forecast signal experiments
python scripts/build_phase2_paper.py
python scripts/build_phase2_slides.py
python scripts/build_phase2_error_analysis_notebook.py --execute
```

The Overleaf bundle `reports/paper_phase2/phase2_ieee_source.zip` holds the `.tex`,
`authors.json`, `README.md` and the figures the `.tex` includes. Rebuild it from
`reports/paper_phase2/` after the paper:

```bash
cd reports/paper_phase2
(printf '%s\n' phase2_ieee_paper.tex authors.json README.md;
 grep -o 'figures/[A-Za-z0-9_]*\.png' phase2_ieee_paper.tex | sort -u) | zip -X -@ phase2_ieee_source.zip.new
mv phase2_ieee_source.zip.new phase2_ieee_source.zip
```

`phase2_eval_all.py` takes about a minute and under 1 GB of memory. It first checks that it
reproduces the Phase 1 test metrics and the Phase 1 `paired_bootstrap.csv` intervals exactly,
then writes to `data/metadata/phase2/`:

| File | Contents |
|---|---|
| `metrics.csv`, `metrics_summary.csv` | MAE / RMSE / MAPE per model, seed, horizon, period (val and test) |
| `run_meta.csv` | parameters, epochs, stop reason, training time, device per run |
| `phase1_parity.json` | the Phase 1 reproduction check |
| `bootstrap.csv` | paired day-block bootstrap intervals (positive = candidate has lower MAE) |
| `hypotheses.csv` | H0-H5 verdicts, computed from the intervals |
| `sensor_metrics.csv`, `region_metrics.csv` | per-sensor and per-coordinate-region errors |
| `events.csv`, `onset_errors.csv` | proxy congestion onset recall and onset MAE |
| `case_study_717468.csv` | the Phase 1 case-study window with every model's forecast |

## 3. Conventions

Speeds are in source units and times are on the stored clock; neither has been verified.
Regions are coordinate regions, not named freeways. Congestion is a proxy (speed below
0.6 x the sensor's training 85th percentile). Peak means weekday 07-10 and 16-19 forecast
origins. Signal-control results come from a software simulation, not a field trial.
