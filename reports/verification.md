> Historical Person A delivery record from October 16. For the integrated Phase 1 completion and current results, see [the completion audit](phase1_completion_verification.md) and [the verified evaluation](person_e_report.md).

# Person A verification record

Verified on 2026-10-16. **Person A's Phase 1 dataset engineering, quality analysis, EDA and handoff deliverables are complete.** The dataset has documented quality warnings; there are no structural validation failures. This does not complete Persons B–E's work or the team's full Phase 1.

## Real data and provenance

The user supplied `DCRNN/metr-la.h5` within this project. It and the canonical copy `data/raw/metr-la.h5` have identical size (57,038,056 bytes) and SHA-256 `64784b76d6fb8ec9bff4b6decafb354da2bb37840468fdccee5044e511277c05`. All eight raw/provenance manifest entries match their recorded hashes. No remote artifact checksum or original local-file download URL is claimed.

The original sibling DCRNN working tree remains unchanged. PEMS-BAY was not used.

## Data validation and outputs

| Check | Verified result |
| --- | --- |
| Actual speed dimensions | 34,272 timestamps × 207 sensors |
| Stored coverage | 2012-03-01 00:00 to 2012-06-27 23:55 |
| Elapsed cadence | Every interval exactly five minutes |
| Duplicate / missing / off-grid timestamps | 0 / 0 / 0 |
| Graph order and coordinates | Exact ID/order match; complete locations |
| NaN / infinity / negative readings | 0 / 0 / 0 |
| Exact zeros | 575,302, or 8.10935% of all sensor cells |
| Simultaneous all-sensor zeros | 2,148 timestamp rows in 97 blocks |
| Longest simultaneous zero block | 381 sampling slots |
| Individual zero runs / possible-outage runs | 25,963 / 6,942 under the declared diagnostic rule |
| Positive flatline runs / large consecutive jumps | 135 / 7,990, retained as observations |
| Constant / low-variance sensors | 0 / 0 under the declared rule |
| Per-sensor metadata | 207 measured rows in each required sensor/missingness CSV |
| Split grid timestamps | 23,990 train / 3,427 validation / 6,855 test |
| EDA | Ten real-data PNGs; every image visually inspected |
| Exported figure links | All quality-report figure targets exist |

Independent raw-data recomputation agrees with the exported CSV totals, event counts, split boundaries and reported descriptive means. Speed units and timezone remain unverified. Exact zeros are benchmark missing/suspect; physical causes are not assigned. Raw measurements are preserved and no imputation occurs in Person A's code.

## Executed notebooks

| Notebook | Executed code cells | Saved outputs | Error outputs |
| --- | ---: | ---: | ---: |
| `01_data_understanding.ipynb` | 8 | 34 | 0 |
| `02_data_quality.ipynb` | 9 | 33 | 0 |

Both notebook schemas validate after execution. The first notebook embeds the ten EDA figures plus a raw-versus-masked comparison. The second saves actual validation tables, zero-run context and the regenerated report. JSON, HTML, Markdown, stream and PNG outputs were captured from real cell execution as applicable. No outputs were synthesized.

The notebooks were executed with `/opt/anaconda3/bin/python scripts/execute_notebooks.py --in-process`. This explicitly selected IPython execution because the session denies both TCP and Unix socket binding for Jupyter kernels. Cells executed in fresh IPython namespaces with in-memory history and a 300-second per-cell limit. The method, interpreter, package versions and execution time are recorded in notebook metadata and `data/metadata/notebook_execution_summary.json`. This is actual in-process execution; a listening Jupyter kernel was not started. Static plotting uses the Agg backend.

## Software and reproduction

All **36** quality, workflow and descriptive-arithmetic tests pass under the scientific execution environment; the same suite also passes in the newer cached `.venv`. Tests cover raw preservation, masks, positive near-zero retention, elapsed gaps, duplicate handling, graph alignment, chronological boundaries, stale-output invalidation, weighted means and diagnostic block definitions. Fixture outputs remain in temporary directories and are not dataset findings.

The pre-existing scientific installation was used without modification: Python 3.12.2, NumPy 1.26.4, pandas 2.2.2, PyTables 3.9.2, Matplotlib 3.8.4, nbformat 5.9.2, nbclient 0.8.0 and IPython 8.25.0. `requirements-executed.txt` records exact direct runtime versions. `requirements.txt` states supported ranges. No network installation is needed to rerun with this existing local interpreter.

Reproduction commands and team scope are in [README](../README.md); the data-quality rules and next team member's obligations are in [Person B handoff](person_b_handoff.md).
