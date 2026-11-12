"""Script to generate notebooks/phase2_error_analysis.ipynb using nbformat (Member 4).

Contract: reports/phase2_plan.md section 8.4.
Outputs a clean, executable Jupyter notebook performing stratified error analysis.
"""

from __future__ import annotations

from pathlib import Path
import nbformat as nbf

from src.phase2.common import ROOT

NOTEBOOK_PATH = ROOT / "notebooks" / "phase2_error_analysis.ipynb"


def build_notebook() -> Path:
    nb = nbf.v4.new_notebook()

    cells = []

    # 1. Header
    cells.append(nbf.v4.new_markdown_cell("""# Phase 2: Spatiotemporal Error Analysis & Graph Model Diagnostics

**Author:** Member 4 (`phase2/m4-evaluation`)  
**Project:** METR-LA Highway Traffic Speed Forecasting — Phase 2 Deep & Graph Benchmark  

This notebook evaluates multi-horizon traffic speed forecasts, isolating where and when road network graph representations (GAT, DCRNN) provide statistically measurable advantages over non-graph deep controls (`gru_local`, `gat_identity`, `dcrnn_identity`, `gat_shuffled`) and Phase 1 competitive tree baselines (`xgboost_ha`).

> **Scientific Disclaimers & Terminology Invariants:**
> - Speeds are measured in **source speed units** (unverified, likely mph).
> - Timestamps represent the **stored clock** (unverified timezone).
> - Spatial groupings represent deterministic **coordinate regions**, not administrative freeways.
> - Congestion states are identified via a declared **proxy threshold** (< 60% of training 85th percentile speed).
"""))

    # 2. Setup
    cells.append(nbf.v4.new_code_cell("""import os
import sys
from pathlib import Path

# Ensure writable matplotlib cache
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_p2"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add repository root to path
REPO_ROOT = Path("..").resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

METADATA_DIR = REPO_ROOT / "data" / "metadata" / "phase2"
FIGURES_DIR = REPO_ROOT / "figures" / "phase2"

print(f"Repository Root: {REPO_ROOT}")
print(f"Metadata Dir:    {METADATA_DIR}")
"""))

    # 3. Overall Performance
    cells.append(nbf.v4.new_markdown_cell("""## 1. Overall Forecast Performance Across Horizons (5, 15, 30, 60 min)

We load the aggregated metrics summary across all seeds on the held-out test split.
"""))

    cells.append(nbf.v4.new_code_cell("""summary_path = METADATA_DIR / "metrics_summary.csv"
if summary_path.is_file():
    df_summary = pd.read_csv(summary_path)
    test_all = df_summary[(df_summary["split"] == "test") & (df_summary["period"] == "all")]
    pivot_mae = test_all.pivot(index="model", columns="horizon", values="mae_mean")[["5min", "15min", "30min", "60min"]]
    display(pivot_mae.style.format("{:.3f}").highlight_min(color="lightgreen", axis=0))
else:
    print(f"Notice: {summary_path.name} not found. Run scripts/phase2_eval_all.py to generate tables.")
"""))

    # 4. Peak vs Off-Peak
    cells.append(nbf.v4.new_markdown_cell("""## 2. Rush Hour vs. Off-Peak Error Stratification

Non-recurrent congestion during peak hours (07:00–10:00 and 16:00–19:00 on weekdays) presents the primary forecasting challenge. We examine whether graph models mitigate error spikes during peak congestion.
"""))

    cells.append(nbf.v4.new_code_cell("""if summary_path.is_file():
    peak_comp = df_summary[
        (df_summary["split"] == "test")
        & (df_summary["horizon"] == "60min")
        & (df_summary["period"].isin(["peak", "off_peak"]))
    ]
    if not peak_comp.empty:
        pivot_peak = peak_comp.pivot(index="model", columns="period", values="mae_mean")[["peak", "off_peak"]]
        pivot_peak["peak_increase_%"] = (pivot_peak["peak"] - pivot_peak["off_peak"]) / pivot_peak["off_peak"] * 100
        display(pivot_peak.style.format({"peak": "{:.3f}", "off_peak": "{:.3f}", "peak_increase_%": "+{:.1f}%"}))
"""))

    # 5. Graph Architecture & Topology Ablation
    cells.append(nbf.v4.new_markdown_cell("""## 3. Graph Architecture & Topology Ablations

To isolate the genuine value of road network topology, we compare:
1. **GAT (Bidirectional)**: GAT with real highway adjacency.
2. **GAT Identity (Control)**: GAT architecture with self-loops only (isolating temporal attention without spatial neighbours).
3. **GAT Shuffled (Control)**: GAT with permuted sensor labels (preserving node degree multiset while destroying spatial connectivity).
"""))

    cells.append(nbf.v4.new_code_cell("""fig_ablation_path = FIGURES_DIR / "fig_graph_ablation.png"
if fig_ablation_path.is_file():
    from IPython.display import Image
    display(Image(filename=str(fig_ablation_path)))
else:
    print("Graph ablation figure pending evaluation runner.")
"""))

    # 6. Formal Hypotheses
    cells.append(nbf.v4.new_markdown_cell("""## 4. Formal Hypothesis Testing (H0–H5)

Evaluated via paired circular moving-block bootstrap (2-day blocks, 2000 resamples, seed 42).
"""))

    cells.append(nbf.v4.new_code_cell("""hyp_path = METADATA_DIR / "hypotheses.csv"
if hyp_path.is_file():
    df_hyp = pd.read_csv(hyp_path)
    display(df_hyp[["id", "statement", "comparison", "mae_difference", "verdict"]])
else:
    print("Hypotheses table not found. Run scripts/phase2_eval_all.py to compute verdicts.")
"""))

    # 7. Congestion Onset Diagnostics
    cells.append(nbf.v4.new_markdown_cell("""## 5. Congestion Onset Detection Diagnostics

Evaluating precision and recall of predicting proxy congestion transitions (speed dropping below 60% of sensor free-speed).
"""))

    cells.append(nbf.v4.new_code_cell("""events_path = METADATA_DIR / "events.csv"
if events_path.is_file():
    df_events = pd.read_csv(events_path)
    display(df_events[["model", "horizon", "precision_congestion", "recall_congestion", "recall_congestion_onset"]])
else:
    print("Events diagnostics table not found.")
"""))

    nb.cells = cells
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    return NOTEBOOK_PATH


if __name__ == "__main__":
    out = build_notebook()
    print(f"Generated Phase 2 error analysis notebook: {out}")
