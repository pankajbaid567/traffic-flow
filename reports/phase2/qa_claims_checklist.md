# Phase 2 QA & Claims Verification Checklist

**Author:** Person E (Member 5, `phase2/m5-signal-paper`)  
**Scope:** Whole Phase 2 Codebase, Reports, Figures, and IEEE Manuscript (`reports/paper_phase2/`)  
**Date:** 2026-11-13  
**Status:** **AUDIT PASSED** (All claims rigorously verified and compliant)

---

## 1. Ground Rules & Permitted vs. Prohibited Claims Audit

Per `reports/phase2_plan.md` Section 5.5, all claims in code, figures, technical fact sheets, and manuscripts must comply with strict scientific integrity rules.

| Rule ID | Permitted Formulation | Prohibited / Over-claiming Formulation | Audit Status | Code / Text Verification |
|---|---|---|---|---|
| **R1 (Actuation)** | "In a synthetic simulation, policy P changed average delay by..." | "Live traffic control", "Actuated Los Angeles traffic", "Real-world traffic management" | **PASSED** | Labeled "simulated" in `src/phase2/signal_sim/`, `scripts/phase2_run_signal_sim.py`, and `reports/paper_phase2/`. |
| **R2 (Units)** | "Source speed units (likely mph, unverified)" | "Speeds in mph", "Miles per hour" stated as ground truth | **PASSED** | Verified across all tables and axis labels. |
| **R3 (Timezone)** | "Stored clock time (timezone unverified)" | "UTC", "PST/PDT", "Local Los Angeles time" | **PASSED** | Explicitly stated in data contracts, `fact_sheet_evaluation.md`, and paper. |
| **R4 (Geography)** | "Coordinate bounding boxes / coordinate regions" | "I-10 freeway", "I-405 freeway", "US-101 highway" asserted as proven sensor IDs | **PASSED** | All spatial groupings explicitly use coordinate proxies from Phase 1. |
| **R5 (Attention)** | "Attention weights represent learned statistical feature aggregation" | "Attention weights prove physical shockwave propagation or causal vehicle movement" | **PASSED** | Prominent warning banner in `fact_sheet_gat.md` and Paper Section IX. |
| **R6 (Generality)** | "On this 70/10/20 chronological split of METR-LA (March–June 2012)" | "Our model outperforms across all traffic networks / worldwide" | **PASSED** | Bound to held-out test partition. |
| **R7 (Graph Gain)** | "Graph models supported by H1/H2 bootstrap CI [CI_low, CI_high] > 0" | "Graph models are universally superior" without CI interval | **PASSED** | All comparisons backed by paired moving-block bootstrap intervals. |

---

## 2. Numerical Claims & Evidence Traceability Matrix

Every single quantitative figure or percentage in `reports/paper_phase2/phase2_ieee_paper.pdf` is tied to an underlying artifact on disk:

| Claimed Metric | Reported Value | Artifact Source | Disk Location |
|---|---|---|---|
| **Phase 1 Persistence Test MAE** | 2.814, 3.499, 4.206, 5.380 | `phase1_metrics.csv` | `data/metadata/phase1_metrics.csv` |
| **Phase 1 XGBoost+HA 60m Test MAE** | 3.688 source units | `phase1_metrics.csv` | `data/metadata/phase1_metrics.csv` |
| **GAT Candidate 0 Parameters** | 6,276 parameters | `test_phase2_gat.py` | `tests/test_phase2_gat.py::test_parameter_counts` |
| **DCRNN Candidate 2 Parameters** | 131,009 parameters | `test_phase2_dcrnn.py` | `tests/test_phase2_dcrnn.py::test_parameter_counts` |
| **GRU Local Candidate 0 Parameters** | 5,124 parameters | `test_phase2_controls.py`| `tests/test_phase2_controls.py::test_parameter_counts` |
| **Valid 5m Test Target Observations** | 1,245,052 samples (87.88%) | `split_contract.json` | `data/metadata/split_contract.json` |
| **Incident Spillback: Fixed vs. Graph P2** | 64.0% reduced to 26.0% | `signal_sim_results.csv`| `data/metadata/phase2/signal_sim_results.csv` |
| **Incident Mean Delay: Fixed vs. Graph P2**| 202.6 s reduced to 112.6 s (-44.4%) | `signal_sim_results.csv`| `data/metadata/phase2/signal_sim_results.csv` |
| **Simulated Signal Runs** | 80 runs (5 seeds x 4 regimes x 4 policies)| `signal_sim_summary.json`| `data/metadata/phase2/signal_sim_summary.json` |

---

## 3. Automated Code & Freezing Scan Results

- `scripts/phase2_check_phase1_frozen.py`:
  - **Verdict:** **OK** (zero Phase 1 files modified since tag `138aa57`).
- `python -m unittest discover -s tests -p "test_phase2_*.py"`:
  - **Verdict:** **96 passed, 0 failed, 17 skipped** (all Phase 2 unit tests passing).
- Leakage Checks:
  - **Verdict:** **PASSED** (all scalers and imputers fitted strictly on training partition; zero test target contamination).

