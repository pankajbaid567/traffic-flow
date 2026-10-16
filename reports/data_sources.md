# Data sources, provenance and evidence limits

## Local source

The existing checkout at `/Users/pankajbaid/projects/AML/DCRNN` has:

```text
origin: https://github.com/liyaguang/DCRNN.git
commit: 602afd9d767d3aa1c9b3eac51710d6aeee12c227
commit timestamp: 2024-12-09T09:40:10-08:00
```

The independent project copies only relevant METR-LA assets; it does not modify that checkout. Acquisition must record the copied files' source locations, sizes and SHA-256 checksums. A repository commit identifies the inspected code/assets, not the provenance of an independently obtained speed file.

Available local inputs include `data/sensor_graph/adj_mx.pkl`, `graph_sensor_ids.txt` and `graph_sensor_locations.csv`. The repository also contains Bay Area files; those are not METR-LA inputs. The initial sibling-repository inventory found no speed file. The user later supplied `traffic-flow-capstone/DCRNN/metr-la.h5`; it was copied unchanged to `data/raw/metr-la.h5`. The accompanying `pems-bay.h5` was not used.

## Official speed-data distribution

The [DCRNN README at the inspected commit](https://github.com/liyaguang/DCRNN/blob/602afd9d767d3aa1c9b3eac51710d6aeee12c227/README.md#data-preparation) identifies `metr-la.h5` as the Los Angeles traffic file and states that the HDF5 file stores a pandas DataFrame. Its distribution links are:

- [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX)
- [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g)

The terminal acquisition attempt could not resolve the official Google Drive host. PyPI requests also failed DNS resolution, preventing installation of missing analysis dependencies. No configured browser was available for acquisition. These are environment-access failures; they do not establish that the remote dataset has disappeared. That initial download attempt failed. Acquisition was subsequently completed from the user-supplied local file using the pre-existing `/opt/anaconda3/bin/python` scientific runtime. The supplied source file and canonical copy have identical SHA-256 `64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05` and size 57,038,056 bytes. No remote checksum comparison or independently verified original download URL is claimed.

The manifest records the actual local source path and checksum, with the official distribution URL retained as a reference rather than a verified local-file download origin. A compatible shape alone is insufficient proof that a third-party file is identical to the official distribution. Do not substitute synthesized speed measurements.

## Dataset characteristics: documented versus measured

The [original DCRNN paper](https://arxiv.org/html/1707.01926), §4 and Appendix E.1, describes METR-LA as highway loop-detector traffic information from Los Angeles County, selects 207 sensors, aggregates speed into 5-minute windows and uses a 70% training / 10% validation / 20% test allocation. Its nominal collection period is March 1–June 30, 2012.

The acquired artifact measures 34,272 timestamps × 207 sensors, starting 2012-03-01 00:00 and ending 2012-06-27 23:55. Every elapsed interval is five minutes; there are no duplicate or missing timestamps. Its end date therefore differs from the paper's nominal June 30 coverage. All measured quality findings come from the HDF5 speed table, independently checked against exported metadata; adjacency alone supplies no speed findings.

The [official location file](https://github.com/liyaguang/DCRNN/blob/602afd9d767d3aa1c9b3eac51710d6aeee12c227/data/sensor_graph/graph_sensor_locations.csv) supplies sensor-location metadata. The [adjacency generator](https://github.com/liyaguang/DCRNN/blob/602afd9d767d3aa1c9b3eac51710d6aeee12c227/scripts/gen_adj_mx.py) stores the ordered IDs, ID-to-index mapping and a thresholded distance-derived adjacency matrix. Their correspondence to the acquired speed columns has been verified by sensor ID and exact ordering.

## Zero and missing-value evidence

The paper's §4.2 states that missing values are excluded when calculating MAE, RMSE and MAPE. The inspected implementation makes the zero convention explicit:

- [Historical-average baseline](https://github.com/liyaguang/DCRNN/blob/602afd9d767d3aa1c9b3eac51710d6aeee12c227/scripts/eval_baseline_methods.py#L12-L30) defaults `null_val` to zero and excludes those readings from historical means.
- [Baseline evaluation](https://github.com/liyaguang/DCRNN/blob/602afd9d767d3aa1c9b3eac51710d6aeee12c227/scripts/eval_baseline_methods.py#L89-L106) passes `null_val=0` to all three metrics.
- [Metric code](https://github.com/liyaguang/DCRNN/blob/602afd9d767d3aa1c9b3eac51710d6aeee12c227/lib/metrics.py#L62-L98) constructs its mask from labels unequal to the supplied null value.
- [DCRNN supervisor](https://github.com/liyaguang/DCRNN/blob/602afd9d767d3aa1c9b3eac51710d6aeee12c227/model/dcrnn_supervisor.py#L78-L79) also uses zero as the training-loss null value.

The project therefore predeclares exact zeros as benchmark missing/suspect while preserving the original data. This is a documented operational rule. It does not determine whether a particular zero represents a sensor outage, corruption or real stopped traffic. Actual inspection found 97 simultaneous all-sensor zero blocks and long individual runs; these are consistent with unavailable benchmark readings but do not prove a physical cause. See the measured quality report and executed quality notebook. Positive near-zero values are retained.

## Units and timezone

The inspected upstream README, scripts and searchable paper text do not explicitly establish the speed file's unit or timezone. MPH is a common METR-LA reporting convention, but it has not been verified as artifact metadata here. Figures and thresholds must use **source-speed units** until units are established; do not silently convert to km/h.

Likewise, a Los Angeles sensor location does not prove how timestamps were encoded. Inspect whether the acquired pandas index has timezone information. If it is naive, preserve its stored clock and label hour/weekday/peak analysis as based on that clock with timezone unverified. Do not silently localize it to `America/Los_Angeles`, UTC or the user's local timezone. Any later timezone decision must be documented and reused by the team.

## Reference to cite

Li, Yaguang; Yu, Rose; Shahabi, Cyrus; Liu, Yan. *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting*. ICLR 2018. [Paper](https://arxiv.org/abs/1707.01926), [official implementation](https://github.com/liyaguang/DCRNN).
