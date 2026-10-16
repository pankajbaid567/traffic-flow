# Traffic flow capstone — Phase 1, Persons A and B

This independent project contains Person A's METR-LA dataset engineering, quality investigation, exploratory analysis and handoff. The user's speed file was supplied in `DCRNN/metr-la.h5` within this project; graph assets came from the original sibling `../DCRNN` repository. Both source locations remain unchanged.

**Person A deliverables are complete with documented data-quality warnings.** The real speed dataset is validated, all ten EDA figures are reviewed, and both notebooks have executed successfully. Execution details and acceptance evidence are recorded in [verification](reports/verification.md).

## Measured dataset

| Item | Verified result |
| --- | --- |
| Speed table | 34,272 timestamps × 207 sensors |
| Stored date range | 2012-03-01 00:00 to 2012-06-27 23:55 |
| Sampling | Exactly 5 minutes; no duplicate or missing timestamps |
| NaN / infinity / negative readings | 0 / 0 / 0 |
| Exact zeros | 575,302 readings, 8.109% of all cells |
| All-sensor zero timestamps | 2,148, grouped into 97 contiguous blocks |
| Graph alignment | All speed sensor IDs and their order match the stored 207×207 adjacency |
| Constant / unusually low-variance sensors | None under the declared valid-speed rule |

The [quality report](reports/data_quality_report.md) contains the evidence and interpretation. Raw files are unchanged. Exact zeros are treated as benchmark missing/suspect, while positive near-zero speeds remain valid. No imputation is performed by Person A. Speed units and timezone are unverified; plots use source units and the stored clock.

## Deliverables

| Path | Contents |
| --- | --- |
| `data/raw/metr-la.h5` | Canonical, byte-preserved speed measurements |
| `data/raw/adj_mx.pkl` | Graph stored for future phases; excluded from AML inputs |
| `data/raw/manifest.json` | Source paths, distribution reference, sizes and SHA-256 checksums |
| `data/metadata/sensors.csv` | All 207 sensors, locations and measured quality statistics |
| `data/metadata/missingness_summary.csv` | Actual NaN/nonfinite/zero/invalid counts and fractions |
| `data/metadata/split_contract.json` | Shared chronological timestamp boundaries |
| `notebooks/01_data_understanding.ipynb` | Real-data overview and all ten EDA topics |
| `notebooks/02_data_quality.ipynb` | Validation, zero investigation, event examples and rules |
| `figures/` | Ten exported PNG figures and descriptive sensor-selection evidence |
| `reports/data_quality_report.md` | Measured findings and data-quality rules |
| `reports/person_b_handoff.md` | Accepted inputs, shared contracts and limitations |

Other metadata tables contain exact timestamp-gap evidence, zero runs, positive flatlines, large jumps, temporal profiles and speed quantiles. Full-period EDA describes the dataset; it must not become fitted preprocessing or sensor selection based on held-out statistics.

## Run in the current workspace

A complete pre-existing scientific environment is available at `/opt/anaconda3/bin/python` and was used without modification. The separate `.venv` contains a newer, partial cached environment and is not the environment used for real-data execution. Exact direct runtime versions are in `requirements-executed.txt`.

From this directory:

```bash
MPLCONFIGDIR="$PWD/.mplconfig" XDG_CACHE_HOME="$PWD/.cache" /opt/anaconda3/bin/python -m src.data.run_person_a
MPLCONFIGDIR="$PWD/.mplconfig" XDG_CACHE_HOME="$PWD/.cache" /opt/anaconda3/bin/python scripts/execute_notebooks.py --in-process
/opt/anaconda3/bin/python -m unittest discover -s tests -v
```

The notebook helper normally uses a Jupyter kernel. `--in-process` executes real cells through IPython and captures their outputs without starting a listening kernel, because this session blocks both TCP and Unix socket binding. The notebook metadata records the execution method. Each notebook is saved only after every cell succeeds.

## Reproduce on another machine

Use Python 3.12 and create a fresh environment. `requirements.txt` lists supported dependency ranges; `requirements-executed.txt` records exact direct versions used here.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-executed.txt
.venv/bin/python -m src.data.run_person_a
.venv/bin/python scripts/execute_notebooks.py
```

Raw data is not committed by the supplied `.gitignore`. Preserve `data/raw/manifest.json` when sharing the actual data separately. To import an original speed file into a fresh copy with the upstream graph repository available:

```bash
.venv/bin/python scripts/acquire_data.py --dcrnn ../DCRNN --metr-la /absolute/path/metr-la.h5
```

The [official DCRNN distribution](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) is linked from the [upstream README](https://github.com/liyaguang/DCRNN#data-preparation). The supplied local file's original download URL is not independently known; its measured structure and local checksum are verified. See [source evidence](reports/data_sources.md).

## Team scope and handoff

Read the [plan](reports/person_a_plan.md), [Person B handoff](reports/person_b_handoff.md) and [verification record](reports/verification.md). The split is chronological, based on the complete 5-minute grid: 23,990 train, 3,427 validation and 6,855 test timestamps. Person B owns feature construction and causal input imputation, with all fitted parameters learned on training timestamps only. All future targets of a sample must remain in one partition.

The input history is 12 observations; target offsets are +1, +3, +6 and +12. Only the predicted sensor's own historical speed and temporal information may enter AML features. No flow/occupancy variables, neighboring speeds or graph-derived inputs are fabricated.

Person A's work does not complete the team's Phase 1. Feature engineering, persistence/historical baselines, XGBoost, Random Forest and model evaluation belong to Persons B–E. Graph learning, SUMO and signal control are outside the current scope.

## Person B — features and processed datasets

Person B built the one feature pipeline that Persons C and D train on. Code is in `src/data/load_data.py`, `split.py`, `features.py` and `dataset_builder.py`; feature lists are also written down in `configs/features.yaml`. Read the [Person C/D handoff](reports/person_c_d_handoff.md) before training anything.

```bash
/opt/anaconda3/bin/python -m src.data.dataset_builder
```

This writes `data/processed/{train,val,test}.parquet` and `dataset_summary.json` (git-ignored, about 330 MB, ~12 s to rebuild). One row is one sensor at one prediction origin: 32 features, 4 raw targets (+5/+15/+30/+60 min) and 4 `target_valid_*` flags.

| Split | Origins | Rows (×207 sensors) | First origin | Last origin |
| --- | ---: | ---: | --- | --- |
| Train | 23,967 | 4,961,169 | 2012-03-01 00:55 | 2012-05-23 06:05 |
| Validation | 3,416 | 707,112 | 2012-05-23 07:05 | 2012-06-04 03:40 |
| Test | 6,844 | 1,416,708 | 2012-06-04 04:40 | 2012-06-27 22:55 |

- Zero / NaN / negative speeds are missing. Inputs are forward-filled (past only) and leading gaps get the sensor's training median; the medians are saved in `dataset_summary.json`.
- Targets are raw and never filled. Filter on `target_valid_<h>` before training or scoring.
- Origins whose targets cross a split boundary are dropped; history may come from the previous split.
- Only the sensor's own speed plus time features are used. No neighbours or graph inputs.
