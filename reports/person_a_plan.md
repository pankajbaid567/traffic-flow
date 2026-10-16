# Person A — Phase 1 plan

Project directory: `AML/traffic-flow-capstone`. Source repository: sibling `DCRNN` (read-only).

## Scope

Own acquisition, canonical raw data, validation, zero/missing investigation, EDA, sensor metadata, and a documented handoff. No Person B feature pipeline, models, graph learning, simulation, or control is implemented here.

## Execution sequence

1. Inventory DCRNN; copy only METR-LA adjacency, sensor IDs, locations and provenance into this independent project.
2. Acquire the original `metr-la.h5` if absent; record source URL, file sizes and SHA-256 checksums. Never synthesize substitute speed data.
3. Validate dimensions, numeric speeds, ordered unique sensor IDs and timestamps, observed cadence, exact gaps, date coverage, nonfinite/negative/zero values, constant/low-variance sensors and candidate outage/discontinuity events.
4. Preserve raw files byte-for-byte. Define zero as a suspect/missing benchmark value, keep positive near-zero values and distinguish diagnostic event flags from proven outages. Keep model imputation and all fitted transforms training-only in Person B's pipeline.
5. Generate required metadata, descriptive plots and two reproducible executed notebooks from the real dataset; report any acquisition or validation blockers explicitly.
6. Verify outputs, checksums, metadata alignment, notebook execution and handoff constraints. Canonical 5-minute chronological partitions must preserve actual forecast elapsed time and keep every target of a sample within its partition.

## Acceptance criteria

- `data/raw/metr-la.h5` and `data/raw/adj_mx.pkl` exist with provenance.
- Required metadata CSVs cover exactly the speed dataset's ordered sensor IDs.
- All ten EDA topics requested are covered; tables accompany figures.
- `notebooks/01_data_understanding.ipynb` and `notebooks/02_data_quality.ipynb` execute without errors.
- `reports/data_quality_report.md` contains measured results and explicit quality rules; `reports/person_b_handoff.md` identifies limitations and shared contracts.
- Acquisition failure leaves a clearly marked blocked report, never invented measurements or success claims.

## Initial findings

- DCRNN contains adjacency and sensor locations, but no `.h5`, `.npz` or `.npy` speed dataset.
- DCRNN source commit: `602afd9d767d3aa1c9b3eac51710d6aeee12c227`.
- Terminal download attempt on the official Google Drive host failed DNS resolution. Browser inventory is empty. Preparing the reproducible workflow while locating a usable source copy.

## Execution outcome

- Created the independent project and copied graph/provenance assets without changing source files.
- Acquired the real user-supplied `DCRNN/metr-la.h5` into canonical raw storage and recorded an identical source/copy checksum.
- Verified 34,272 chronological, unique five-minute timestamps and all 207 sensors; no missing timestamps, NaN, infinite or negative readings. Sensor IDs/order and locations align exactly with the graph.
- Measured zero values, synchronized zero blocks, long sensor runs, positive flatlines, variance and large jumps. Preserved all raw values; documented the conservative zero-handling rule and diagnostic limitations.
- Generated ten reviewed EDA figures, measured sensor/missingness metadata, descriptive evidence tables, quality report and persisted chronological boundaries.
- Successfully executed both notebooks against the real dataset through IPython, recording the execution method and actual outputs. Socket restrictions prevent a listening Jupyter kernel but do not prevent this execution method.
- Passed 36 software tests under both installed scientific and cached modern environments; independent raw-data audit agrees with exported results.
- Person A deliverables are complete with documented quality warnings. The full Phase 1 still requires Persons B–E. See `reports/verification.md` for acceptance evidence and runtime details.
