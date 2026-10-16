# Person A → Person B handoff

**Handoff status: real speed data validated with documented quality warnings.** The supplied `DCRNN/metr-la.h5` has been copied unchanged to `data/raw/metr-la.h5`, with SHA-256 `64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05`. It contains 34,272 ordered, unique 5-minute timestamps and 207 sensors, spanning 2012-03-01 00:00 to 2012-06-27 23:55. No timestamps are missing. Speed sensor IDs and their order match the graph exactly. All ten EDA figures are available; notebook execution is recorded in `verification.md`.

Warnings are data-quality findings, not structural failures: 575,302 exact-zero cells (8.109%), 135 positive flatline runs and 7,990 large consecutive changes under predeclared thresholds. No positive flatline or jump is automatically removed. There are no NaN/infinite/negative readings and no constant or low-variance sensors under the declared rule.

## Persisted timestamp partitions

| Partition | Start inclusive | End inclusive | Timestamps |
| --- | --- | --- | ---: |
| Train | 2012-03-01 00:00 | 2012-05-23 07:05 | 23,990 |
| Validation | 2012-05-23 07:10 | 2012-06-04 04:40 | 3,427 |
| Test | 2012-06-04 04:45 | 2012-06-27 23:55 | 6,855 |

Use `data/metadata/split_contract.json` as the machine-readable source. These are timestamp partitions, not counts of eligible supervised samples. History availability, target masks and boundary rejection will reduce sample counts.


## Canonical inputs and acceptance

The canonical inputs are `data/raw/metr-la.h5` and `data/raw/adj_mx.pkl`. Preserve their bytes and record source paths/URLs, file sizes and SHA-256 digests. Any masks, repaired grids or imputed features are derived artifacts; never write them back into the original HDF5 file.

Before accepting the handoff, verify that:

- The HDF5 speed table loads and has a chronological, unique datetime index and numeric sensor columns.
- The measured sensor/timestamp counts, date coverage, observed intervals, exact time gaps, duplicate timestamps and value-quality counts are present in `data_quality_report.md`.
- `sensors.csv` covers exactly the ordered HDF5 sensor columns. Graph sensor IDs and adjacency mappings agree with that order or have an explicit verified mapping.
- `missingness_summary.csv` is derived from the actual speed file, and both notebooks complete against that file.

These checks have passed for the supplied file; the measured CSV inventory follows the original speed columns. Recheck checksums when transferring data. Never replace HDF5 columns with graph IDs solely because their lengths match.

## Measurement and missingness contract

Only speed is available under this Phase 1 contract. Do not fabricate flow or occupancy. Speed units and timestamp timezone are not verified by the inspected upstream metadata; preserve values and stored timestamps, and label source units/time accordingly.

Use the following predeclared rules consistently:

| Condition | Person A treatment | Person B obligation |
|---|---|---|
| Finite speed greater than zero | Retain, including near-zero speeds | Use as observed local-sensor data |
| Exact zero | Flag as benchmark missing/suspect; preserve raw zero | Represent as missing before lag/rolling calculations; retain an observation-missing flag |
| NaN, infinity or negative speed | Flag as missing/invalid; preserve original raw value | Exclude from valid targets and handle inputs under a documented causal rule |
| Absent 5-minute timestamp | Report exact gap; preserve raw index | Reindex to the canonical grid with explicit absent-timestamp and missing-value flags before constructing windows |
| Duplicate timestamp or sensor ID | Report structural failure | Resolve explicitly before feature generation; do not silently average or discard |
| Constant/low-variance or candidate-outage flag | Report diagnostic only | Do not drop sensors using full-series or held-out statistics |

Exact-zero masking follows the [upstream baseline implementation](https://github.com/liyaguang/DCRNN/blob/602afd9d767d3aa1c9b3eac51710d6aeee12c227/scripts/eval_baseline_methods.py#L89-L106), which excludes zero targets from metrics and zero observations from historical averages. This convention is evidence for handling zeros as unavailable benchmark readings, not proof of the cause of each zero.

Person A performs no imputation. Person B must select and document one shared causal input-imputation rule for all models. Fit any fallback medians, means, scalers or other learned parameters using training timestamps only. Forward filling must use observations strictly available at the prediction origin; backfilling and interpolation from future observations are prohibited. Keep missing targets missing and exclude them from the corresponding model training/evaluation rows. Do not impute ground truth.

## Quality diagnostics

The initial diagnostic thresholds are fixed for inspection:

- Low variance: standard deviation below `1` source-speed unit, calculated on valid positive observations.
- Possible outage: a contiguous exact-zero run of at least `12` 5-minute slots, equivalent to one hour.
- Large discontinuity: an absolute change of at least `30` source-speed units between consecutive, valid observations exactly five minutes apart.

These flags identify cases to inspect; they are not established fault labels. A long zero run may be an outage, an abrupt positive-speed change may be real congestion, and low variance alone does not prove a failed sensor. Missing timestamps interrupt consecutive-observation comparisons and must not be bridged as if they were five minutes apart.

Do not tune diagnostic thresholds using model test errors. Any later sensor exclusion or learned preprocessing decision must use training-only evidence, be documented, and be applied identically for all Phase 1 models and future graph comparisons.

## Chronological partitions and sample construction

Persist one complete canonical 5-minute chronological timestamp grid and its split boundaries for the whole team. On that grid of length `N`, including flagged rows for absent timestamps, use:

```text
n_train = floor(0.70 * N)
n_val   = floor(0.10 * N)
n_test  = N - n_train - n_val

train = timestamps[:n_train]
val   = timestamps[n_train:n_train + n_val]
test  = timestamps[n_train + n_val:]
```

Record the complete grid length, index start/end and exact partition boundaries. Use the same persisted definition everywhere; splitting the original observed rows would give a different partition if gaps exist. Changing indexing conventions requires regenerating and documenting the shared split. Never shuffle time or derive a separate split for each model.

For prediction origin `t`, history consists of `t, t−1, …, t−11` on the 5-minute grid. Targets are `t+1`, `t+3`, `t+6`, `t+12`. Check elapsed timestamp differences explicitly; adjacent file rows alone do not establish five minutes when timestamps are missing.

All four target timestamps for a sample must belong to the same partition, and the origin must precede the earliest target. Discard origins whose targets cross a partition boundary. Earlier observed history may come from a preceding partition because it was already available at prediction time. Such history must use the shared training-fitted transforms; it does not permit refitting on validation/test values or using future targets in features. Validation/test input observations may be used when they are available at the current prediction origin under the same rolling forecasting protocol.

## Feature restrictions and evaluation handoff

Person B owns the only shared feature pipeline used by Persons C and D. Features may use the predicted sensor's own past speed, causal rolling/trend statistics, temporal variables and missingness indicators. The adjacency matrix, coordinates, graph embeddings and neighboring sensor speeds are excluded from AML inputs. Pooling training examples across sensors is compatible with this local-input restriction if each row still contains only the target sensor's allowed information.

All rolling windows look backward and end at `t`. Persist feature definitions, sensor order, split boundaries, preprocessing parameters, validity masks and forecast offsets. Carry those contracts into later graph-model comparisons.

Person E owns the common MAE/RMSE/MAPE implementation. Exclude unavailable/nonfinite targets consistently and explicitly document any additional near-zero MAPE threshold or denominator floor. Report valid-target counts per horizon. Positive near-zero speeds remain valid data even if the MAPE definition uses a declared denominator rule.

The shared diagnostic peak definition is weekday mornings `[07:00, 10:00)` and evenings `[16:00, 19:00)` on the stored clock; other times are off-peak. Its real-world local-time interpretation remains subject to the unresolved timezone evidence.

No Person B feature implementation, model training or evaluation results are included in Person A's delivery. The quality warnings and unresolved units/timezone must travel with this dataset. Phase 1 additionally requires the work of Persons B–E.
