# Phase 1 completion verification

Verified at 2026-10-21T09:22:56.794158+00:00. All 311 deterministic acceptance checks passed. The full software suite passed 82 tests with no failures, errors or skips.

Raw/provenance hashes are unchanged; model and forecast hashes match the evaluation manifest. Two partitions by four horizons by five models by five periods produce 200 scored groups. Valid-target counts match the raw-derived shared dataset, and sensor error sums reconstruct each pooled MAE. All 8 frozen XGBoost artifacts and 16 regenerated RF artifacts are present. All three notebooks have executed cells without saved errors. All 14 evaluation PNGs decode successfully.

The numerical sources are `phase1_metrics.csv`, `sensor_forecast_metrics.csv`, `region_forecast_metrics.csv`, `paired_bootstrap.csv` and `person_e_results.json`. The full machine-readable audit is `phase1_verification.json`. Test execution evidence is recorded separately in `phase1_test_results.json`. Visual inspection of the IEEE PDF is a separate final authoring gate.

Speed units, timezone and named freeway identities remain unverified and are labelled accordingly. The original booster files are re-evaluated with official XGBoost 3.4.1, not refitted or tuned on test. RF ablations use the original fixed settings. The additional offline numeric JSON reader exactly matches the official runtime on 96 deterministic validation rows per model and also matches a direct scalar tree walk.
