# Phase 2 QA & Claims Verification Checklist

**Original author:** Person E (Member 5, `phase2/m5-signal-paper`), 2026-11-13
**Re-audited:** 2026-11-17, after the Phase 2 branches were merged
**Scope:** Phase 2 code, reports, figures and the IEEE manuscript (`reports/paper_phase2/`)
**Final check:** 2026-11-18, after all training and evaluation finished
**Status:** **CLOSED.** Every Phase 2 number in the paper, slides and notebook is read from evaluation outputs; wording rules pass (Section 1); automated checks pass (Section 3).

> **Correction (2026-11-17).** The 2026-11-13 version of this file said "AUDIT PASSED (all claims rigorously verified)". That was wrong:
> - The GAT / DCRNN / GRU Local / control test MAEs and the H1–H4 "SUPPORTED" verdicts in the paper were typed into `scripts/build_phase2_paper.py`. No model had been trained and no evaluation output existed.
> - `src/phase2/signal_sim/simulator.py` was committed empty (commit `2034792`), so the signal numbers it cited (incident delay 202.6 s → 112.6 s, "−44.4%", spillback 64.0% → 26.0%) could not be reproduced. The simulator was rebuilt in `0d1db93`, and its results do not support those claims.
> - The Sensor 717468 "case study" figure was drawn from hand-made curves, not from data or model output. It has been removed.
> - GRU Local candidate 0 has 5,124 parameters (as listed), but the paper's Table IV latency, epoch-time and tree-model size figures had no source.
>
> The paper builder now reads every Phase 2 forecasting number from `data/metadata/phase2/`. All three model families were trained on 2026-11-17/18 and evaluated, so nothing in the paper is pending.

---

## 1. Wording rules (plan Section 5.5)

| Rule | Allowed | Not allowed | Status | Where checked |
|---|---|---|---|---|
| R1 Actuation | "In a synthetic simulation, policy P changed average delay by ..." | live / real-world traffic control | PASS | paper Section VIII, both signal summary disclaimers |
| R2 Units | "source speed units (likely mph, unverified)" | mph stated as fact | PASS | paper tables and axes |
| R3 Timezone | "stored clock (timezone unverified)" | UTC / PST stated as fact | PASS | paper Sections IV and VIII |
| R4 Geography | "coordinate regions" | named freeways | PASS | paper Section VIII |
| R5 Attention | attention = what the model used | attention proves shockwave propagation | PASS | paper Sections II and VIII, `fact_sheet_gat.md` |
| R6 Generality | "on this chronological split of METR-LA" | generalisation beyond the split | PASS | paper Section VIII |
| R7 Graph gain | claim only what the H1/H2 verdicts support | "graph is better" without verdicts | PASS (graph gains are quoted with intervals; H2 is reported as contradicted with both parts, and the shuffled-graph gain is stated) | paper Sections VI and XI |

## 2. Numerical claims and their sources

| Claim in the paper | Value | Source |
|---|---|---|
| Phase 1 test MAE (all five models) | e.g. persistence 2.814 / 3.499 / 4.206 / 5.380; XGBoost+HA 60 min 3.688 | `data/metadata/phase1_metrics.csv`; harness reproduces them exactly (`tests/test_phase2_evaluation.py`) |
| Valid 5-min test targets | 1,245,052 | `tests/test_phase2_core.py` parity test |
| Phase 2 test MAE (gru_local, gat, gat_identity, gat_shuffled, dcrnn, dcrnn_identity) | e.g. 60 min: GAT 4.175 ± 0.007, DCRNN 3.734 ± 0.036, GRU Local 4.379 ± 0.006 | `data/metadata/phase2/metrics_summary.csv` |
| Bootstrap intervals | e.g. GAT vs GAT identity at 60 min +0.189 [+0.154, +0.221] | `data/metadata/phase2/bootstrap.csv` |
| H0–H5 verdicts | H1 supported, H2 contradicted (a supported, b contradicted), H3 supported, H4 supported, H5 descriptive, H0 contradicted | `data/metadata/phase2/hypotheses.csv` |
| Training runs (epochs, stop reason, minutes, device) | Table I | `data/metadata/phase2/run_meta.csv` (from each run's `meta.json`) |
| Search candidate parameter counts | e.g. GAT c0 6,276; DCRNN c2 131,009 | computed from the run files at build time; also asserted in `tests/test_phase2_gat.py`, `tests/test_phase2_dcrnn.py`; matches `search_<family>.csv` |
| Attention summaries | self 0.096-0.098, neighbours 0.902-0.904 | `data/metadata/phase2/attention_stats.csv` |
| Signal experiment A (synthetic, 80 runs) | e.g. incident: fixed 297.5 s, local predictive 26.3 s, graph predictive 25.7 s | `data/metadata/phase2/signal_sim_summary.json`; forecast noise sized from measured test MAEs (XGBoost 32 2.97, GAT 2.905 at 15 min) |
| Signal experiment B (real speeds and forecasts, 3,780 runs) | e.g. DCRNN forecast +0.01 s [-0.02, +0.04] vs XGBoost + HA; perfect forecast +0.40 s | `data/metadata/phase2/signal_sim_real_summary.json` |

Spot checks done by hand on 2026-11-18: abstract, Tables I-V and VIII, the H0/H2 paragraphs and the conclusion against the CSV/JSON files above; no mismatches.

## 3. Automated checks (2026-11-18)

- `python scripts/phase2_check_phase1_frozen.py` → `OK (no Phase 1 file changed since phase1-final)`
- `python -m unittest discover -s tests -p "test_phase2_*.py"` → `Ran 107 tests ... OK`
- `python scripts/phase2_eval_all.py` → Phase 1 parity passed (max |ΔMAE| 0.0), Phase 1 bootstrap intervals reproduced
- Banned-wording scan of `paper_content.json` and the slides (mph, UTC/PST, freeway names, propagation, real-world control, "pending"): only allowed negations found
- LaTeX source compiles (tectonic) with no overfull boxes
- Leakage: the filler and scaler are fitted on training data only (`tests/test_phase2_core.py`).

## 4. Known limitations stated in the paper

1. GAT finals stopped at the 60-epoch limit and DCRNN at its 50-minute wall-clock cap (14 epochs); both may be under-trained.
2. GAT shuffled and DCRNN identity have one seed each.
3. One chronological split; intervals not adjusted for multiple comparisons.
4. Signal results are simulated with an assumed demand rule.
