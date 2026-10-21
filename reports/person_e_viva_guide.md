# Phase 1 defense notes

What did we complete? Data quality and EDA, 32 causal features, chronological datasets, persistence and historical-average baselines, eight XGBoost boosters, four RF feature ablations at four horizons, and common evaluation from prediction rows.

What is the fair model comparison? RF and plain XGBoost use the same 32 local features. The 36-feature XGBoost plus HA extension changes both the model information and the learner relative to RF; its advantage cannot be attributed to boosting alone.

How is leakage limited? Parameters, medians, reference speeds and historical lookups are fitted on training timestamps. Windows look backward; targets remain raw; all four targets stay in one split. HA training features use leave-one-out lookup values. Test predictions are regenerated with fixed models/settings for reproducibility, not hyperparameter selection.

How do we handle unavailable readings? Exact zeros and nonfinite/negative targets are not scored. Positive targets are retained; MAPE has a declared 1e-5 denominator floor. All models have identical valid-target counts. Empty slices are not perfect forecasts.

What do maps establish? Observed and predicted speeds, per-sensor errors and a declared congestion-frequency proxy over the held-out period. They do not establish named freeway membership, traffic direction, physical queue propagation, or causal sensor relationships.

What is the uncertainty calculation? A paired circular moving-day-block bootstrap with blocks of two days, all sensors kept together, 2,000 resamples and seed 42. It measures variation within this held-out interval, not universal or selection-adjusted significance.

Can we claim mph or UTC? No. Use source speed units and the stored clock until authoritative metadata verifies units and timezone.

Did Phase 1 prove that graph learning wins? No graph model is evaluated. Graph methods are a testable next experiment. Consult `person_e_report.md` for the generated numerical tables rather than memorizing manually copied values.
