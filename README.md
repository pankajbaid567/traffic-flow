# Traffic Flow Forecasting — Phase 1

A 5-person team project that predicts **traffic speed** on Los Angeles highways **5, 15, 30 and 60 minutes into the future**, using past speed readings from road sensors.

> This is an academic study (Phase 1), not a live traffic-control system.
> Full numbers: [reports/person_e_report.md](reports/person_e_report.md) · IEEE-style paper: [reports/paper/](reports/paper/)

---

## 1. Problem statement

Traffic jams waste time and fuel. If we can **predict how fast traffic will move in the next few minutes to an hour**, drivers can pick better routes and traffic managers can act earlier.

**Our question:** given the last 1 hour of speed readings from a road sensor, how well can machine learning predict that sensor's speed 5, 15, 30 and 60 minutes later? And which model does it best?

**What we did not do in Phase 1:** we did not use graph models (models that also look at nearby sensors on the road network, e.g. GAT, DCRNN). Each forecast uses **only that sensor's own history**. Graph models are planned for Phase 2.

---

## 2. The dataset — METR-LA

| Item | Value |
|---|---|
| What | Speed readings from **207 road sensors** (loop detectors, which are devices built into the road that measure passing cars) on Los Angeles highways |
| How often | One reading every **5 minutes** |
| Period | 1 March 2012 → 27 June 2012 (**34,272 time points**) |
| Special issue | **8.1% of readings are exactly 0**. We treat a 0 as "sensor gave no reading", not as "traffic stopped" |

---

## 3. How we did it — step by step, person by person

The work flowed like a relay: each person handed their output to the next.

```
Person A  ──►  Person B  ──►  Person C  ─┐
(clean &      (build         (baselines  ├──►  Person E
 explore      features &      + XGBoost) │     (fair comparison,
 data)        split data)    Person D  ──┘      maps, final report)
                             (Random Forest)
```

### 👤 Person A — Get the data and check it can be trusted

**Goal:** make sure the raw data is correct before anyone builds a model on it.

1. **Got the raw file** (`metr-la.h5`, an HDF5 file, which is a format for storing big tables) and saved a **checksum** (SHA-256, a unique "fingerprint" of the file, so we can prove it was never changed).
2. **Checked the structure:** 207 sensors, 34,272 timestamps, no duplicate or missing times, all readings 5 minutes apart, no negative or NaN (empty) values.
3. **Investigated zeros:** found 575,302 zero readings. We decided a zero means **missing data**, the same rule used by the official DCRNN benchmark (a well-known traffic forecasting paper).
4. **Looked for strange sensors:** flatlines (a sensor stuck on the same value), low variance (a sensor that barely changes) and big jumps (sudden unrealistic speed changes).
5. **Made 10 EDA charts** (Exploratory Data Analysis, meaning graphs that show what the data looks like), for example average speed by hour, weekday vs weekend, and missing data over time.

📓 Notebooks: [01_data_understanding.ipynb](notebooks/01_data_understanding.ipynb), [02_data_quality.ipynb](notebooks/02_data_quality.ipynb)
📄 Handoff: [reports/person_b_handoff.md](reports/person_b_handoff.md)

---

### 👤 Person B — Turn raw speeds into model inputs (features)

**Goal:** build a clean, ready-to-train table without cheating (no future information).

1. **Split the data by time** (chronological split, meaning older data trains and newer data tests, like real life):

   | Part | Share | Used for |
   |---|---|---|
   | Train | 70% | Teaching the model |
   | Validation | 10% | Tuning the model's settings |
   | Test | 20% | Final exam, used once at the end |

2. **Prevented data leakage** (accidentally letting the model see the answer). A sample is kept only if all its future targets fall inside the same split.
3. **Built 32 features** (input columns the model learns from). All of them use only the sensor's own past:

   | Group | Count | What it means |
   |---|---|---|
   | Lags | 12 | Speed now, 5 min ago, 10 min ago … 55 min ago |
   | Rolling | 8 | Average / spread / min / max of speed over the last 15, 30 or 60 min |
   | Trend | 3 | How much speed changed in the last 5, 15 or 30 min |
   | Time | 7 | Hour of day, day of week, weekend flag, morning/evening rush-hour flags |
   | Missingness | 2 | Is the current reading missing? How many readings were missing in the last hour? |

4. **Built 4 targets** (the answers to predict): speed at +5, +15, +30 and +60 minutes. Missing targets were **never filled in**. They are just flagged and skipped.
5. Saved everything as **parquet files** (a fast, compressed table format): ~5.0M train rows, 0.7M validation rows and 1.4M test rows.

💻 Code: [src/data/dataset_builder.py](src/data/dataset_builder.py), [src/data/features.py](src/data/features.py)
📄 Handoff: [reports/person_c_d_handoff.md](reports/person_c_d_handoff.md)

---

### 👤 Person C — Simple baselines and XGBoost

**Goal:** set a bar to beat, then beat it with a strong model.

1. **Baseline 1: Persistence** (assume speed in the future = speed right now). It is simple but hard to beat for short horizons.
2. **Baseline 2: Historical Average (HA)** (the usual speed for that sensor at that weekday and time, taken from training data).
3. **XGBoost-32** (Extreme Gradient Boosting, which builds many small decision trees where each new tree fixes the mistakes of the previous ones). It uses Person B's 32 features. One model was trained per horizon.
4. **XGBoost + HA (36 features)**: the same model plus 4 extra features built from the historical average, e.g. "how different is the speed right now from what is normal for this time?". This is our **best model**.

💻 Code: [src/models/baselines.py](src/models/baselines.py), [src/models/xgboost_model.py](src/models/xgboost_model.py) · 📄 [reports/person_c_report.md](reports/person_c_report.md)

---

### 👤 Person D — Random Forest and feature ablation

**Goal:** try a second model type and find out which feature groups actually help.

1. **Random Forest-32** (many independent decision trees whose answers are averaged). It uses the same 32 features as XGBoost, so the comparison is fair.
2. **Ablation study** (removing feature groups to see how much each one matters). The model was trained 4 times with more and more features:

   | Experiment | Features | Count |
   |---|---|---|
   | A | Lags only | 12 |
   | B | + Time | 19 |
   | C | + Rolling | 27 |
   | D | + Trend + Missingness (all) | 32 |

   **Finding:** adding **time** and **rolling** features helped the most, especially for 30–60 min forecasts.

💻 Code: [src/models/random_forest.py](src/models/random_forest.py) · 📄 [reports/person_d_report.md](reports/person_d_report.md)

---

### 👤 Person E — Fair comparison, maps and final report

**Goal:** score every model the same way and explain where and when the models fail.

1. **Re-scored all 5 models on the same test set**, the same 1.24 million valid targets, so the comparison is fair.
2. **Rush hour vs normal time:** checked errors separately for the morning peak (7–10 am), the evening peak (4–7 pm) and off-peak times.
3. **Error maps:** placed each sensor on a map of LA, colored by its error, to see *where* forecasts go wrong.
4. **Congestion maps:** compared predicted vs actual congestion. "Congested" means speed below 60% of that sensor's normal fast speed.
5. **Bootstrap test** (resampling the test days 2,000 times to check that one model really beats another and the win is not luck). This gives a 95% confidence interval (a range we are 95% sure the true improvement lies in).
6. Wrote the final report, the notebook and the IEEE paper.

📓 Notebook: [03_model_evaluation_and_congestion_maps.ipynb](notebooks/03_model_evaluation_and_congestion_maps.ipynb)
📄 [reports/person_e_report.md](reports/person_e_report.md)

---

## 4. Results

**How to read the scores (lower is better for all three):**
- **MAE** (Mean Absolute Error): on average, how far off the prediction is, in speed units (likely mph).
- **RMSE** (Root Mean Squared Error): like MAE, but punishes big mistakes more.
- **MAPE** (Mean Absolute Percentage Error): average error as a percentage.

### MAE on the test set

| Model | 5 min | 15 min | 30 min | 60 min |
|---|---:|---:|---:|---:|
| Persistence | 2.81 | 3.50 | 4.21 | 5.38 |
| Historical Average | 4.19 | 4.19 | 4.19 | 4.19 |
| Random Forest-32 | 2.45 | 3.17 | 3.87 | 4.85 |
| XGBoost-32 | 2.36 | 2.97 | 3.56 | 4.39 |
| **XGBoost + HA-36** | **2.31** | **2.84** | **3.28** | **3.69** |

<details>
<summary>Full table with RMSE and MAPE</summary>

| Horizon | Model | MAE | RMSE | MAPE % | Valid targets |
| --- | --- | --- | --- | --- | --- |
| 5min | Persistence | 2.814 | 4.627 | 6.49 | 1,245,052 |
| 5min | Historical average | 4.188 | 7.855 | 13.04 | 1,245,052 |
| 5min | Random Forest 32 | 2.452 | 4.156 | 6.02 | 1,245,052 |
| 5min | XGBoost 32 | 2.362 | 4.151 | 5.76 | 1,245,052 |
| 5min | XGBoost plus HA 36 | 2.312 | 4.065 | 5.66 | 1,245,052 |
| 15min | Persistence | 3.499 | 6.418 | 8.61 | 1,245,040 |
| 15min | Historical average | 4.188 | 7.855 | 13.04 | 1,245,040 |
| 15min | Random Forest 32 | 3.167 | 5.763 | 8.56 | 1,245,040 |
| 15min | XGBoost 32 | 2.970 | 5.806 | 7.92 | 1,245,040 |
| 15min | XGBoost plus HA 36 | 2.844 | 5.540 | 7.63 | 1,245,040 |
| 30min | Persistence | 4.206 | 8.052 | 10.86 | 1,245,022 |
| 30min | Historical average | 4.188 | 7.855 | 13.04 | 1,245,022 |
| 30min | Random Forest 32 | 3.871 | 7.002 | 11.10 | 1,245,022 |
| 30min | XGBoost 32 | 3.564 | 7.129 | 10.27 | 1,245,022 |
| 30min | XGBoost plus HA 36 | 3.275 | 6.546 | 9.49 | 1,245,022 |
| 60min | Persistence | 5.380 | 10.344 | 14.53 | 1,244,986 |
| 60min | Historical average | 4.189 | 7.855 | 13.04 | 1,244,986 |
| 60min | Random Forest 32 | 4.854 | 8.410 | 14.48 | 1,244,986 |
| 60min | XGBoost 32 | 4.393 | 8.720 | 13.74 | 1,244,986 |
| 60min | XGBoost plus HA 36 | 3.688 | 7.470 | 11.34 | 1,244,986 |

</details>

### Key findings

1. **XGBoost + HA is the best model at every horizon.** At 60 min it cuts error by **31%** compared with Persistence (5.38 → 3.69).
2. **The further ahead we predict, the harder it gets.** Error rises steadily from 5 min to 60 min for every model.
3. **XGBoost beats Random Forest** when both use the same 32 features.
4. **Rush hour is hardest.** At 15 min, evening-peak error is **75% higher** than off-peak error.
5. **These wins are real, not luck.** All bootstrap confidence intervals stay above zero.
6. **Why Phase 2 needs graphs:** errors grow at long horizons because each sensor cannot "see" traffic coming from neighboring sensors. That is the gap graph models (GAT/DCRNN) aim to fill.

### Honest limits
- The speed units and timezone of the data are not officially confirmed.
- Missing (zero) readings are skipped, never guessed.
- Map regions are simple latitude/longitude groups, not verified named highways.
- There is one test period (June 2012), so the results may differ in other cities or seasons.

---

## 5. Project structure

```
data/raw/           original METR-LA file (never modified)
data/processed/     Person B's train/val/test parquet files (rebuild, not in git)
data/metadata/      sensor list, split rules, metric CSVs
notebooks/          01, 02 (Person A) and 03 (Person E)
src/data/           loading, quality checks, EDA, features, splitting
src/models/         baselines, XGBoost, Random Forest, metrics
scripts/            one runner script per person + verification
models/             saved trained models
figures/            all charts and maps
reports/            each person's report, handoffs, final paper
tests/              unit tests
```

---

## 6. How to run it

Use **Python 3.12**. For the exact package versions:

```bash
pip install -r requirements-phase1-executed.txt
```

Then run each step in order:

```bash
python -m src.data.dataset_builder                          # Person B: build features + splits
NUMBA_NUM_THREADS=4 python scripts/run_person_c.py --predict-only   # Person C: score saved XGBoost models
python scripts/run_person_d.py --save-models                # Person D: train Random Forests
python scripts/predict_person_d.py                          # Person D: predictions
python scripts/run_person_e.py                              # Person E: evaluation, maps, report
python -m unittest discover -s tests -v                     # run tests
python scripts/build_notebook_03.py                         # rebuild notebook 03
python scripts/verify_phase1.py                             # check everything matches
python scripts/build_phase1_paper.py                        # build the IEEE paper
```

Or run everything at once with `python scripts/complete_phase1.py`.

**Notes**
- `run_person_d.py --resume --save-models` continues a Random Forest run that was interrupted.
- The processed data, predictions and large model files are not stored in git. Regenerate them with the commands above.
- On machines with restricted folders, set `MPLCONFIGDIR` and `XDG_CACHE_HOME` to writable folders.

Full completion audit: [reports/phase1_completion_verification.md](reports/phase1_completion_verification.md)
