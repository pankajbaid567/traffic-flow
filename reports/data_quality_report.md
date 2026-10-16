# METR-LA data quality report — Person A, Phase 1

**Validation status: WARNING**

## Measured descriptive findings

The source covers **2012-03-01T00:00:00 through 2012-06-27T23:55:00**, with 34,272 timestamps and 207 sensors. Speeds use unverified source units. Calendar groups use the stored clock; no timezone is inferred or converted.

Across **6,519,002 valid positive sensor cells**, mean speed is **58.46**, median **63.22**, and the 5th–95th percentile range is **25.62–68.89**.

Excluded readings total **575,302 / 7,094,304 (8.11%)**; exact zeros account for 575,302. Positive near-zero speeds remain observed. No imputation or trained preprocessing is performed.

Hourly pooled means are lowest at stored hour **17:00 (49.85)** and highest at **22:00 (62.96)**. Daily pooled means are lowest on **Friday (56.32)** and highest on **Sunday (63.21)**.

All temporal means pool valid sensor cells using their counts. They are descriptive associations; unavailable readings may be nonrandom, so these profiles can be biased. Counts and exclusion rates accompany the means.

### Weekday and weekend observations

| day type | Valid cells | Mean speed | Excluded (%) |
| --- | --- | --- | --- |
| Weekday | 4690209 | 57.20 | 7.44 |
| Weekend | 1828793 | 61.68 | 9.78 |

### Traffic-period observations

| period | Valid cells | Mean speed | Excluded (%) |
| --- | --- | --- | --- |
| Weekday morning peak | 574372 | 52.02 | 9.32 |
| Weekday evening peak | 597567 | 48.25 | 5.66 |
| Weekday off-peak | 3518270 | 59.57 | 7.43 |
| Weekend | 1828793 | 61.68 | 9.78 |

Peak windows are weekdays [07:00, 10:00) and [16:00, 19:00); weekday off-peak excludes both windows. Weekend observations are reported separately. Full hourly and day-of-week tables are exported alongside these tables.

### Strongest zero-value evidence

All 207 sensors are zero together in **2,148 timestamp rows (6.27%)**, forming **97** consecutive five-minute blocks. Gaps and duplicate timestamps interrupt blocks.

These simultaneous-zero rows account for **77.2874%** of all exact-zero sensor cells.

The longest simultaneous-zero block has **381 slots**, from **2012-03-26T08:25:00** through **2012-03-27T16:05:00** (1,905 minutes of five-minute slots; the elapsed first-to-last span is one slot shorter).

Sensors with the highest zero rates:

| Sensor | Zero cells | Zero (%) |
| --- | --- | --- |
| 718076 | 6890 | 20.10 |
| 716939 | 6464 | 18.86 |
| 760987 | 5532 | 16.14 |

Longest individual zero runs (slot duration includes all observed five-minute slots):

| sensor_id | position | start_timestamp | end_timestamp | length | duration_minutes | possible_outage |
| --- | --- | --- | --- | --- | --- | --- |
| 717490 | 70 | 2012-06-05 12:25:00 | 2012-06-11 08:35:00 | 1683 | 8,415.00 | True |
| 717491 | 71 | 2012-06-05 12:25:00 | 2012-06-11 08:35:00 | 1683 | 8,415.00 | True |
| 769444 | 60 | 2012-06-21 17:35:00 | 2012-06-26 08:20:00 | 1330 | 6,650.00 | True |

Simultaneous zeros and long zero runs are evidence for investigating unavailable measurements. They do not establish a physical cause or prove an outage. Raw values are preserved, exact zeros are flagged as benchmark missing/suspect, and no sensor is automatically removed.


## Measured dataset validation

All numeric findings below are computed from the checksum-recorded HDF5 file. Source values and timestamp timezone are preserved; speed units are unverified.

### Validation status

```json
"warning"
```

### Issues

```json
[
  "575302 suspect/invalid speed values are masked in valid_speed; raw values are preserved.",
  "135 positive constant runs warrant inspection; readings are retained.",
  "7990 large consecutive speed changes warrant inspection; readings are retained."
]
```

### Dimensions

```json
{
  "timestamps": 34272,
  "unique_timestamps": 34272,
  "sensors": 207,
  "observed_cells": 7094304,
  "expected_timestamps": 34272,
  "expected_grid_cells": 7094304,
  "missing_timestamp_cells": 0
}
```

### Timestamp validation

```json
{
  "start": "2012-03-01T00:00:00",
  "end": "2012-06-27T23:55:00",
  "timezone": null,
  "timezone_note": "Naive source timestamps; no timezone inferred.",
  "expected_sampling_minutes": 5.0,
  "modal_sampling_minutes": 5.0,
  "duplicate_rows": 0,
  "chronologically_ordered": true,
  "missing_timestamp_count": 0,
  "missing_timestamps": [],
  "off_grid_timestamp_count": 0,
  "off_grid_timestamps": [],
  "gap_count": 0
}
```

### Value counts

```json
{
  "missing_nan": 0,
  "infinite": 0,
  "zero": 575302,
  "negative_finite": 0,
  "valid_positive_finite": 6519002,
  "invalid_total": 575302
}
```

### Zero evidence

```json
{
  "all_zero_timestamp_rows": 2148,
  "at_least_90pct_zero_timestamp_rows": 2148,
  "all_invalid_timestamp_rows": 2148,
  "total_zero_runs": 25963,
  "possible_outage_runs": 6942,
  "maximum_zero_run_steps": 1683,
  "interpretation": "Zero is suspect, not proof of an outage or genuinely stopped traffic."
}
```

### Sensor quality

```json
{
  "no_valid_observations": 0,
  "constant_sensors": 0,
  "raw_constant_sensors": 0,
  "all_zero_sensors": 0,
  "low_variance_sensors": 0,
  "positive_flatline_runs": 135,
  "large_discontinuities": 7990
}
```

### Metadata validation

```json
{
  "provided": true,
  "sensor_count": 207,
  "duplicate_sensor_ids": 0,
  "missing_sensor_ids": [],
  "extra_sensor_ids": [],
  "all_sensors_covered": true
}
```

### Graph validation

```json
{
  "sensor_ids_provided": true,
  "adjacency_provided": true,
  "used_as_features": false,
  "sensor_count": 207,
  "duplicate_sensor_ids": 0,
  "same_sensor_set": true,
  "same_sensor_order": true,
  "missing_sensor_ids": [],
  "extra_sensor_ids": [],
  "adjacency_shape": [
    207,
    207
  ],
  "adjacency_shape_valid": true,
  "adjacency_finite": true
}
```

### Policy

```json
{
  "expected_sensors": 207,
  "frequency": "0 days 00:05:00",
  "low_variance_std_threshold": 1.0,
  "zero_run_min_steps": 12,
  "flatline_min_steps": 12,
  "discontinuity_threshold": 30.0,
  "invalid_values": [
    "NaN",
    "infinity",
    "negative",
    "exact zero"
  ],
  "positive_near_zero": "retained",
  "imputation": "none",
  "raw_modified": false,
  "valid_std_ddof": 1
}
```

### Generated at utc

```json
"2026-10-16T16:48:51.671283+00:00"
```

### Speed sha256

```json
"64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05"
```

## Zero and missing-value handling

The [upstream baseline](https://github.com/liyaguang/DCRNN/blob/602afd9d767d3aa1c9b3eac51710d6aeee12c227/scripts/eval_baseline_methods.py#L89-L106) masks zero targets and excludes zero observations from historical averages. We preserve raw bytes, treat exact zero as benchmark missing/suspect, mask NaN/nonfinite/negative values and retain positive near-zero observations. The data alone cannot establish whether any particular zero is stopped traffic, an outage or corruption.

Contiguous zero runs, simultaneous sensor failures, flatlines and large jumps provide diagnostic evidence. Run thresholds are predeclared in config. No sensor is automatically removed, and no full-period statistic is used to fit imputation or model selection. Imputation belongs to Person B and must use only past observations and parameters fitted on training timestamps. Invalid targets must remain masked rather than imputed.

## Diagnostic evidence tables (first 12 rows; full CSVs in metadata)

### Timestamp gaps

None detected.

### Zero runs

| sensor_id | position | start_timestamp | end_timestamp | length | duration_minutes | possible_outage |
| --- | --- | --- | --- | --- | --- | --- |
| 773869 | 0 | 2012-03-01 00:15:00 | 2012-03-01 00:20:00 | 2 | 10.0 | False |
| 773869 | 0 | 2012-03-03 00:15:00 | 2012-03-03 00:15:00 | 1 | 5.0 | False |
| 773869 | 0 | 2012-03-03 10:30:00 | 2012-03-03 10:30:00 | 1 | 5.0 | False |
| 773869 | 0 | 2012-03-03 18:15:00 | 2012-03-03 18:35:00 | 5 | 25.0 | False |
| 773869 | 0 | 2012-03-04 11:05:00 | 2012-03-04 11:10:00 | 2 | 10.0 | False |
| 773869 | 0 | 2012-03-04 18:35:00 | 2012-03-04 18:45:00 | 3 | 15.0 | False |
| 773869 | 0 | 2012-03-04 19:00:00 | 2012-03-04 19:00:00 | 1 | 5.0 | False |
| 773869 | 0 | 2012-03-04 19:10:00 | 2012-03-04 19:10:00 | 1 | 5.0 | False |
| 773869 | 0 | 2012-03-05 13:30:00 | 2012-03-05 14:50:00 | 17 | 85.0 | True |
| 773869 | 0 | 2012-03-05 19:20:00 | 2012-03-05 19:50:00 | 7 | 35.0 | False |
| 773869 | 0 | 2012-03-06 11:55:00 | 2012-03-06 12:00:00 | 2 | 10.0 | False |
| 773869 | 0 | 2012-03-06 14:10:00 | 2012-03-06 14:15:00 | 2 | 10.0 | False |

### Flatline runs

| sensor_id | position | start_timestamp | end_timestamp | length | duration_minutes | speed |
| --- | --- | --- | --- | --- | --- | --- |
| 717804 | 26 | 2012-03-06 12:05:00 | 2012-03-06 14:05:00 | 25 | 125.0 | 46.0 |
| 717804 | 26 | 2012-03-06 14:20:00 | 2012-03-06 15:55:00 | 20 | 100.0 | 46.0 |
| 717804 | 26 | 2012-03-06 16:10:00 | 2012-03-06 17:45:00 | 20 | 100.0 | 46.0 |
| 717804 | 26 | 2012-03-07 20:10:00 | 2012-03-07 22:50:00 | 33 | 165.0 | 52.0 |
| 774067 | 62 | 2012-04-19 02:00:00 | 2012-04-19 03:05:00 | 14 | 70.0 | 70.0 |
| 774067 | 62 | 2012-04-20 02:00:00 | 2012-04-20 03:00:00 | 13 | 65.0 | 70.0 |
| 774067 | 62 | 2012-04-27 02:15:00 | 2012-04-27 03:45:00 | 19 | 95.0 | 70.0 |
| 774067 | 62 | 2012-04-28 03:25:00 | 2012-04-28 04:40:00 | 16 | 80.0 | 70.0 |
| 718076 | 105 | 2012-03-07 09:25:00 | 2012-03-07 16:10:00 | 82 | 410.0 | 64.0 |
| 718496 | 140 | 2012-03-08 16:00:00 | 2012-03-08 18:45:00 | 34 | 170.0 | 65.0 |
| 764853 | 144 | 2012-05-02 17:05:00 | 2012-05-02 18:15:00 | 15 | 75.0 | 70.0 |
| 763995 | 149 | 2012-03-14 02:10:00 | 2012-03-14 03:20:00 | 15 | 75.0 | 70.0 |

### Discontinuities

| sensor_id | position | previous_timestamp | timestamp | previous_speed | speed | speed_change | absolute_change |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 773869 | 0 | 2012-03-01 18:05:00 | 2012-03-01 18:10:00 | 47.111111111111114 | 13.0 | -34.111111111111114 | 34.111111111111114 |
| 773869 | 0 | 2012-03-01 19:35:00 | 2012-03-01 19:40:00 | 30.875 | 66.0 | 35.125 | 35.125 |
| 773869 | 0 | 2012-03-04 10:45:00 | 2012-03-04 10:50:00 | 68.66666666666667 | 25.0 | -43.66666666666667 | 43.66666666666667 |
| 773869 | 0 | 2012-03-04 11:35:00 | 2012-03-04 11:40:00 | 28.0 | 65.0 | 37.0 | 37.0 |
| 773869 | 0 | 2012-03-07 19:30:00 | 2012-03-07 19:35:00 | 24.25 | 62.666666666666664 | 38.416666666666664 | 38.416666666666664 |
| 773869 | 0 | 2012-03-08 15:00:00 | 2012-03-08 15:05:00 | 62.75 | 32.75 | -30.0 | 30.0 |
| 773869 | 0 | 2012-03-09 20:05:00 | 2012-03-09 20:10:00 | 24.125 | 60.22222222222222 | 36.09722222222222 | 36.09722222222222 |
| 773869 | 0 | 2012-03-12 17:45:00 | 2012-03-12 17:50:00 | 60.375 | 27.333333333333332 | -33.04166666666667 | 33.04166666666667 |
| 773869 | 0 | 2012-03-12 18:00:00 | 2012-03-12 18:05:00 | 65.625 | 34.875 | -30.75 | 30.75 |
| 773869 | 0 | 2012-03-12 18:45:00 | 2012-03-12 18:50:00 | 21.5 | 63.888888888888886 | 42.388888888888886 | 42.388888888888886 |
| 773869 | 0 | 2012-03-13 17:10:00 | 2012-03-13 17:15:00 | 64.55555555555556 | 27.0 | -37.55555555555556 | 37.55555555555556 |
| 773869 | 0 | 2012-03-14 15:00:00 | 2012-03-14 15:05:00 | 37.111111111111114 | 5.625 | -31.486111111111114 | 31.486111111111114 |

## EDA figures

Descriptive full-period EDA is not an estimator fitted for forecasting. Missing/suspect speed values are excluded; observed-speed patterns can be biased by nonrandom missingness. Hours use the stored clock; peak definitions are weekdays 07:00–10:00 and 16:00–19:00, half-open intervals.

- [overall speed distribution](../figures/overall_speed_distribution.png)

- [sensor speed distribution](../figures/sensor_speed_distribution.png)

- [average speed by hour](../figures/average_speed_by_hour.png)

- [average speed by day of week](../figures/average_speed_by_day_of_week.png)

- [weekday weekend profile](../figures/weekday_weekend_profile.png)

- [morning evening congestion](../figures/morning_evening_congestion.png)

- [missingness by sensor](../figures/missingness_by_sensor.png)

- [missingness over time](../figures/missingness_over_time.png)

- [sensor variance distribution](../figures/sensor_variance_distribution.png)

- [sample sensor timeseries](../figures/sample_sensor_timeseries.png)

## Handoff and limitations

See [Person B handoff](person_b_handoff.md). Structural failures block the canonical handoff. Warning-level diagnostics require awareness, not automatic sensor deletion. `data/metadata/split_contract.json` defines timestamp boundaries only; Person B owns feature samples. No models, performance findings, graph features, SUMO or control were implemented.
