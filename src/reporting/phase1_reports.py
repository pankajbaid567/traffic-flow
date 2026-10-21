"""Single-source narrative and tables for Phase 1 completion."""

from pathlib import Path
import json

LABELS = {
    "persistence": "Persistence",
    "historical_average": "Historical average",
    "random_forest_32": "Random Forest 32",
    "xgboost_32": "XGBoost 32",
    "xgboost_ha": "XGBoost plus HA 36",
}
HORIZONS = ["5min", "15min", "30min", "60min"]


def markdown_table(columns, rows):
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    return "\n".join(lines)


def write_reports(root, result, metrics, sensor_metrics, region_metrics, ci, corridors):
    root = Path(root)
    report = root / "reports"
    # Use the independently scored, unrounded values for displayed headline
    # tables. Rounding a four-decimal model summary again can change the last
    # printed digit at a three-decimal boundary.
    from scripts.run_person_c import write_report as write_person_c_report

    c_results = json.loads((root / "data/metadata/person_c_results.json").read_text())
    d_results = json.loads((root / "data/metadata/person_d_results.json").read_text())
    for _, row in metrics.iterrows():
        if row.model == "random_forest_32":
            continue
        model = {"xgboost_32": "xgboost"}.get(row.model, row.model)
        c_results["horizons"][row.horizon][row.split][model][row.period].update(
            {
                "mae": float(row.mae),
                "rmse": float(row.rmse),
                "mape": float(row.mape),
                "n": int(row.n),
            }
        )
    write_person_c_report(c_results, d_results, report / "person_c_report.md")
    bench = metrics[(metrics.split == "test") & (metrics.period == "all")]
    lines = []
    for h in HORIZONS:
        for model, label in LABELS.items():
            r = bench[(bench.horizon == h) & (bench.model == model)].iloc[0]
            lines.append(
                [
                    h,
                    label,
                    f"{r.mae:.3f}",
                    f"{r.rmse:.3f}",
                    f"{r.mape:.2f}",
                    f"{int(r.n):,}",
                ]
            )
    benchmark = markdown_table(
        ["Horizon", "Model", "MAE", "RMSE", "MAPE %", "Valid targets"], lines
    )
    peak_rows = []
    for period in ["off_peak", "morning_peak", "evening_peak"]:
        row = ["15min", period]
        for model in LABELS:
            r = metrics[
                (metrics.split == "test")
                & (metrics.horizon == "15min")
                & (metrics.period == period)
                & (metrics.model == model)
            ].iloc[0]
            row.append(f"{r.mae:.3f}")
        peak_rows.append(row)
    peak_table = markdown_table(
        ["Horizon", "Origin period"] + list(LABELS.values()), peak_rows
    )
    interval_table = markdown_table(
        ["Horizon", "Baseline", "Candidate", "MAE reduction", "95% interval"],
        [
            [
                r.horizon,
                LABELS[r.baseline],
                LABELS[r.candidate],
                f"{r.mae_difference:.3f}",
                f"[{r.ci95_low:.3f}, {r.ci95_high:.3f}]",
            ]
            for _, r in ci.iterrows()
        ],
    )
    geographic_table = markdown_table(
        [
            "Coordinate region",
            "Sensors",
            "Morning observed speed",
            "Evening observed speed",
            "Evening drop from reference %",
        ],
        [
            [
                r.region,
                int(r.sensor_count),
                f"{r.mean_morning_speed:.2f}",
                f"{r.mean_evening_speed:.2f}",
                f"{r.evening_speed_drop_pct:.2f}",
            ]
            for _, r in corridors.iterrows()
        ],
    )
    e15 = metrics[
        (metrics.split == "test")
        & (metrics.horizon == "15min")
        & (metrics.model == "xgboost_ha")
    ].set_index("period")
    penalty = 100 * (e15.loc["evening_peak", "mae"] / e15.loc["off_peak", "mae"] - 1)
    generated = result["metadata"]["generated_at_utc"]
    protocol = """## Evaluation protocol

The preserved METR-LA file has 34,272 five-minute timestamps and 207 sensors, from 2012-03-01 00:00 through 2012-06-27 23:55 on the stored clock. Its 575,302 exact-zero cells (8.109%) are treated as unavailable benchmark readings. Physical speed units and timestamp timezone remain unverified. We do not identify the causes of individual zero events.

The canonical grid contains 23,990 training, 3,427 validation and 6,855 test timestamps. Forecasts use 12 observations ending at origin t and targets t+1, t+3, t+6 and t+12. All four targets must stay in one partition. This yields 4,961,169 training, 707,112 validation and 1,416,708 test rows. Earlier observed history may cross a partition boundary. Parameters and historical averages are fitted on training timestamps only; forward filling uses past inputs, with a training-median fallback for leading gaps. Ground truth is never imputed.

Five models use the same target-validity mask, sensor order and forecast origins. The 32-feature RF and XGBoost comparison is like for like. XGBoost plus HA is a separate 36-feature extension using four features derived from the same sensor's training history; training rows exclude their own lookup observation. No graph, coordinates, sensor ID or neighboring speeds enter Phase 1 predictive inputs. Pooled fitting across sensors does not provide neighboring observations to an individual forecast.

All metrics are recomputed from aligned prediction parquets. MAE and RMSE use finite positive observed targets. MAPE uses max(y, 1e-5) as its denominator. Positive near-zero targets remain included. Empty slices have no score, and nonfinite predictions on valid targets cause an error. Peak flags describe weekday forecast origins at [07:00,10:00) or [16:00,19:00), with no unverified timezone conversion.
"""
    spatial = """## Spatial evaluation and interpretation

The forecast maps compare observed speed, predicted speed and per-sensor forecast MAE on identical valid test rows, for both morning and evening origin periods. Separate maps cover the full-feature RF and XGBoost plus HA at 15 minutes. A congestion-frequency map uses the declared proxy speed < 60% of each sensor's training 85th-percentile speed; this is an operational diagnostic, not a validated physical congestion label.

Regions are deterministic latitude/longitude bins. Verified freeway IDs and traffic directions are not supplied, so these bins must not be presented as confirmed named freeway corridors. Spatial summaries use held-out weekday measurements rather than full-period hourly means including weekends. Speed-drop references are training 85th percentiles, weighted by each sensor's valid peak observations. They are reference-speed proxies, not measured free-flow speeds. Descriptive speed summaries slice raw observation times; forecast-error maps slice forecast-origin times. These are explicitly different row sets.

`data/metadata/sensor_forecast_metrics.csv` and `sensor_error_rankings.csv` contain actual prediction errors, counts, observed/predicted means and congestion frequencies. `region_forecast_metrics.csv` aggregates additive errors and valid counts, avoiding unweighted averaging of sensor MAEs. The historical `corridor_performance.csv` filename now holds explicitly labelled coordinate regions.
"""
    uncertainty = """## Paired uncertainty and limitations

The paired comparison retains all sensors and observations within each stored calendar day and resamples circular blocks of two consecutive days 2,000 times with seed 42. Each resample recomputes the pooled valid-target MAE difference; positive values favor the candidate. The 95% percentile intervals describe sampling variation within this test period. They do not account for every longer dependence pattern, model-selection uncertainty or generalization to other cities/seasons.

The test interval and validation interval have different unavailable-target rates. All reported metrics describe observed targets only; missing-not-at-random behavior can bias the represented traffic conditions. This single test split, fixed random seed and selected model family do not establish a universal performance ceiling. Descriptive worst-error examples are selected retrospectively on the test set for inspection and never for model tuning. Tree impurity/gain importances are model-specific diagnostics, not causal effects.

Phase 1 does not train or test GAT, DCRNN or traffic-signal control. Increasing long-horizon errors and local-input restrictions motivate a future graph comparison; they do not prove measured shockwave propagation or guaranteed graph-model improvement. Any schematic is conceptual.
"""
    artifacts = """## Generated evidence and reproduction

- `data/metadata/phase1_metrics.csv`: all five models, two partitions, four horizons and five temporal groups.
- `data/metadata/sensor_forecast_metrics.csv`: per-sensor errors and observed/predicted congestion rates.
- `data/metadata/region_forecast_metrics.csv`: weighted errors by coordinate region.
- `data/metadata/daily_forecast_errors.csv` and `paired_bootstrap.csv`: paired comparison evidence.
- `data/metadata/person_e_results.json`: generated summaries, reproduction checks, runtime versions and input SHA-256 hashes.
- `data/processed/predictions/person_{c,d}_{val,test}_predictions.parquet`: complete forecast rows (derived, git-ignored).
- `models/person_c`: eight preserved XGBoost boosters; `models/person_d`: regenerated RF ablation models.
- `figures/person_e_*.png`: regenerated benchmark, temporal, spatial forecast, congestion, ablation and conceptual figures.

Use the commands in the README. Frozen XGBoost boosters are re-scored without test-based tuning. RF uses the original fixed 100 trees, depth 15, max_samples 0.1 and seed 42 for all 16 fits. Saved-model inference and a fixed-parameter regeneration are distinct from searching for new settings on the test set. Raw files remain unchanged.
"""
    text = f"""# Phase 1 verified evaluation report

Generated from prediction rows at {generated}. Persons A through E's data engineering, features, local forecasting and evaluation are integrated. The missing forecast maps, RF temporal slices, paired uncertainty and numerical report consistency have been completed.

{protocol}
## Test benchmark

{benchmark}

## Temporal comparison

{peak_table}

For XGBoost plus HA at 15 minutes, evening-origin MAE is {penalty:.1f}% higher than off-peak MAE. This is an observed stored-clock difference, not evidence of a verified physical cause.

{spatial}
{geographic_table}

{uncertainty}
{interval_table}

{artifacts}
"""
    (report / "person_e_report.md").write_text(text.rstrip() + "\n")
    (report / "phase1_synthesis_report.md").write_text(
        text.replace(
            "# Phase 1 verified evaluation report",
            "# Phase 1 synthesis and completion report",
            1,
        ).rstrip()
        + "\n"
    )
    (report / "person_c_e_handoff.md").write_text(
        "# Person C to Person E verified forecast handoff\n\n"
        "The eight frozen XGBoost boosters have been re-scored with official XGBoost 3.4.1. "
        "Validation and test predictions are saved in `data/processed/predictions/person_c_{val,test}_predictions.parquet`. "
        "Each row carries the origin timestamp, sensor ID, peak flags, current speed, raw targets, validity flags and all four Person C forecasts at all four horizons.\n\n"
        "Join Person D forecasts only after verifying identical origin/sensor keys, targets, masks and row order. "
        "Use `scripts/run_person_e.py` to recompute metrics, rather than copying numerical summaries. "
        "Current all-model scores follow. The 32-feature RF/XGBoost comparison is like for like; the 36-feature HA extension is separate.\n\n"
        + benchmark
        + "\n\nThe boosters predict a change from current speed, which must be added back. "
        "HA lookups are fitted on training timestamps only, queried at the forecast target time, and leave-one-out for training features. "
        "Targets remain unimputed; score finite positive observations only. Peak flags refer to weekday origin times on the unverified stored clock. "
        "Use source speed units and verified coordinates without assigning unverified freeway identities. "
        "Frozen-model regeneration is for reproducibility and does not introduce test-based tuning. "
        "Full evidence and current limitations are in `person_e_report.md`.\n"
    )
    (report / "phase2_graph_gap_analysis.md").write_text(
        """# Evidence and hypotheses for a future graph comparison

Phase 1 compares persistence, historical averages, RF and XGBoost using only each target sensor's own history and origin-time variables. MAE generally increases with forecast horizon and differs by stored-clock period. These observations motivate testing whether neighboring observations improve prediction on the same splits and masks.

No measured result in Phase 1 establishes that graph learning is necessary, that traffic shockwaves caused a particular error, or that a local model has an insurmountable limit. The graph diagram is conceptual. Geographic proximity and coordinate regions are not a substitute for verified graph edges or road direction.

A future experiment should freeze the Phase 1 split and horizons, fit preprocessing on train only, compare graph and equally capable non-graph temporal models, and repeat seeds. Use identical observed targets, time slices and paired day-block comparisons. Check whether additional spatial observations improve performance rather than presuming improvement. GAT and DCRNN remain unimplemented future work in this phase. Traffic-signal experiments, if later performed, must be labelled simulated.
"""
    )
    (report / "person_e_phase2_handoff.md").write_text(
        """# Phase 1 completion handoff

The completion source is `person_e_results.json`, independently recomputed from C and D forecast parquets. Full tables and methodological limits are in `person_e_report.md` and `phase1_synthesis_report.md`. Phase 1 includes all five team roles; Phase 2 is future work.

Retain the raw-file hashes, canonical timestamp partitions, own-sensor feature definitions, positive-target masks and forecast offsets. Compare RF and XGBoost on the same 32 features; report the 36-feature HA extension separately. Reuse per-sensor, regional, temporal and paired-day evidence. Geographic groups have no verified freeway identity; units and timezone remain unverified. The conceptual graph motivation is not an empirical graph result.
"""
    )
    (report / "person_e_viva_guide.md").write_text(
        """# Phase 1 defense notes

What did we complete? Data quality and EDA, 32 causal features, chronological datasets, persistence and historical-average baselines, eight XGBoost boosters, four RF feature ablations at four horizons, and common evaluation from prediction rows.

What is the fair model comparison? RF and plain XGBoost use the same 32 local features. The 36-feature XGBoost plus HA extension changes both the model information and the learner relative to RF; its advantage cannot be attributed to boosting alone.

How is leakage limited? Parameters, medians, reference speeds and historical lookups are fitted on training timestamps. Windows look backward; targets remain raw; all four targets stay in one split. HA training features use leave-one-out lookup values. Test predictions are regenerated with fixed models/settings for reproducibility, not hyperparameter selection.

How do we handle unavailable readings? Exact zeros and nonfinite/negative targets are not scored. Positive targets are retained; MAPE has a declared 1e-5 denominator floor. All models have identical valid-target counts. Empty slices are not perfect forecasts.

What do maps establish? Observed and predicted speeds, per-sensor errors and a declared congestion-frequency proxy over the held-out period. They do not establish named freeway membership, traffic direction, physical queue propagation, or causal sensor relationships.

What is the uncertainty calculation? A paired circular moving-day-block bootstrap with blocks of two days, all sensors kept together, 2,000 resamples and seed 42. It measures variation within this held-out interval, not universal or selection-adjusted significance.

Can we claim mph or UTC? No. Use source speed units and the stored clock until authoritative metadata verifies units and timezone.

Did Phase 1 prove that graph learning wins? No graph model is evaluated. Graph methods are a testable next experiment. Consult `person_e_report.md` for the generated numerical tables rather than memorizing manually copied values.
"""
    )
    readme = f"""# Traffic Flow Capstone Phase 1

Phase 1 integrates all five roles: A data quality/EDA, B causal features, C baselines and XGBoost, D Random Forest/ablations, E independently recomputed temporal/spatial evaluation and synthesis.

Results and limits are generated in [the verified report](reports/person_e_report.md). The IEEE-style Phase 1 paper is in `reports/paper/`. This is an academic Phase 1 study, not a deployed traffic-control system.

## Data and protocol

METR-LA: 34,272 five-minute timestamps by 207 sensors; 2012-03-01 through 2012-06-27 on the stored clock. Speed units and timezone are unverified. The 8.109% exact-zero readings are treated as unavailable, with no imputed ground truth. Raw files and checksums are preserved. The chronological grid split is 70/10/20; 12 history observations predict 5, 15, 30 and 60 minutes. All four targets stay in the same partition. Only a sensor's own history enters its forecasts.

## Verified test results

{benchmark}

RF and XGBoost 32 share exactly 32 features. XGBoost plus HA uses 36 and is labelled separately. Actual per-sensor forecast errors and predicted-versus-observed maps replace the previous speed-volatility proxy. Geographic regions are coordinate bins, not verified named highways. Claims of proven shockwave propagation or inevitable graph-model gains are excluded.

## Reproduction

Use Python 3.12. Install `requirements-phase1-executed.txt` in a fresh environment for the exact completion versions; `requirements.txt` provides broader supported ranges. The verified local environment is `.phase1-venv/bin/python`, which inherits the existing scientific packages and adds official XGBoost 3.4.1. The optional numeric JSON reader also matches a scalar walk and the official runtime on 96 validation rows for each saved model. New XGBoost training requires the official package.

```bash
python -m src.data.dataset_builder
# Re-score the eight saved XGBoost boosters without retraining
NUMBA_NUM_THREADS=4 python scripts/run_person_c.py --predict-only
# Regenerate all 16 fixed-parameter RF fits and save their models
python scripts/run_person_d.py --save-models
python scripts/predict_person_d.py
python scripts/run_person_e.py
python -m unittest discover -s tests -v
python scripts/build_notebook_03.py
python scripts/verify_phase1.py
python scripts/build_phase1_paper.py
```

For a full orchestration with logs and checkpoints use `python scripts/complete_phase1.py`. Set `MPLCONFIGDIR` and `XDG_CACHE_HOME` to writable project cache directories in restricted environments. `run_person_d.py --resume --save-models` resumes an interrupted fixed-parameter run. Datasets, predictions and large RF model files are derived artifacts; transfer or regenerate them separately.

## Roles and evidence

| Person | Delivery | Evidence |
|---|---|---|
| A | Raw validation, quality, EDA | `reports/verification.md`, notebooks 01/02 |
| B | 32 features and chronological partitions | `src/data/`, `reports/person_c_d_handoff.md` |
| C | Baselines, XGBoost 32 and HA 36 | `models/person_c/`, `person_c_results.json` |
| D | RF, 12/19/27/32-feature ablations | `person_d_results.json`, saved RF models |
| E | Common metrics, all-model temporal slices, spatial forecast errors, paired uncertainty | `phase1_metrics.csv`, `sensor_forecast_metrics.csv`, `paired_bootstrap.csv` |

The detailed completion audit is `reports/phase1_completion_verification.md`. No GAT/DCRNN training or signal simulation belongs to Phase 1.
"""
    (root / "README.md").write_text(readme)
