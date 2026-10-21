# Evidence and hypotheses for a future graph comparison

Phase 1 compares persistence, historical averages, RF and XGBoost using only each target sensor's own history and origin-time variables. MAE generally increases with forecast horizon and differs by stored-clock period. These observations motivate testing whether neighboring observations improve prediction on the same splits and masks.

No measured result in Phase 1 establishes that graph learning is necessary, that traffic shockwaves caused a particular error, or that a local model has an insurmountable limit. The graph diagram is conceptual. Geographic proximity and coordinate regions are not a substitute for verified graph edges or road direction.

A future experiment should freeze the Phase 1 split and horizons, fit preprocessing on train only, compare graph and equally capable non-graph temporal models, and repeat seeds. Use identical observed targets, time slices and paired day-block comparisons. Check whether additional spatial observations improve performance rather than presuming improvement. GAT and DCRNN remain unimplemented future work in this phase. Traffic-signal experiments, if later performed, must be labelled simulated.
