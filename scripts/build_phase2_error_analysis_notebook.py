"""Build (and optionally execute) notebooks/phase2_error_analysis.ipynb (Member 4).

    python scripts/build_phase2_error_analysis_notebook.py [--execute]

The notebook only reads tracked outputs of scripts/phase2_eval_all.py and
scripts/phase2_run_signal_sim.py (data/metadata/phase2/, figures/phase2/), so it runs
on a clean clone without the git-ignored prediction files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nbformat as nbf

NB_PATH = ROOT / "notebooks" / "phase2_error_analysis.ipynb"

CELLS = [
    ("md", """# Phase 2 error analysis

Reads the tracked result tables written by `scripts/phase2_eval_all.py` (all numbers come from scored
test/validation predictions). Units are **source speed units**, times are the **stored clock**, regions are
**coordinate regions**, congestion is a **proxy**, and signal results are **simulated**."""),
    ("code", """from pathlib import Path
import json
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd().resolve()
while not (ROOT / "data" / "metadata" / "phase2").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
M = ROOT / "data" / "metadata" / "phase2"
F = ROOT / "figures" / "phase2"
pd.set_option("display.precision", 3)
pd.set_option("display.width", 160)
print(json.loads((M / "phase1_parity.json").read_text()))"""),
    ("md", "## 1. Training runs (from each run's `meta.json`)"),
    ("code", """meta = pd.read_csv(M / "run_meta.csv")
meta[["model_id", "seed", "graph", "params", "epochs_run", "best_epoch", "best_val_mae",
      "train_seconds", "stop_reason", "device"]].sort_values(["model_id", "seed"])"""),
    ("md", "## 2. Test MAE by horizon (Phase 2 = mean ± sd over seeds)"),
    ("code", """s = pd.read_csv(M / "metrics_summary.csv")
t = s[(s.split == "test") & (s.period == "all")]
tab = t.pivot(index="model", columns="horizon", values="mae_mean")[["5min", "15min", "30min", "60min"]]
tab["sd_60min"] = t[t.horizon == "60min"].set_index("model")["mae_sd"]
tab.sort_values("60min")"""),
    ("code", """display(Image(filename=str(F / "fig_horizon_mae.png"), width=650))"""),
    ("md", "## 3. Peak vs off-peak, RMSE and MAPE at 60 minutes"),
    ("code", """t60 = s[(s.split == "test") & (s.horizon == "60min")]
t60.pivot(index="model", columns="period", values="mae_mean")[["all", "peak", "off_peak", "morning_peak", "evening_peak"]].sort_values("all")"""),
    ("code", """t60[t60.period == "all"].set_index("model")[["rmse_mean", "mape_mean"]].sort_values("rmse_mean")"""),
    ("md", "## 4. Paired bootstrap (positive = candidate has lower MAE)"),
    ("code", """b = pd.read_csv(M / "bootstrap.csv")
b["pair"] = b.candidate + " vs " + b.baseline
b[b.period == "all"].pivot(index="pair", columns="horizon", values="mae_difference")[["5min", "15min", "30min", "60min"]]"""),
    ("code", """b[(b.horizon == "60min")][["pair", "period", "mae_difference", "ci95_low", "ci95_high"]]"""),
    ("code", """display(Image(filename=str(F / "fig_graph_ablation.png"), width=800))"""),
    ("md", "## 5. Pre-registered hypotheses (verdicts computed by code)"),
    ("code", """h = pd.read_csv(M / "hypotheses.csv")
h.groupby("id", sort=False).first()[["statement", "overall_verdict"]]"""),
    ("code", """h[["id", "horizon", "period", "comparison", "mae_difference", "ci95_low", "ci95_high", "verdict"]]"""),
    ("code", """print(h[h.id == "H5"].note.iloc[0])"""),
    ("md", "## 6. Where are the errors? Sensors, degree, regions"),
    ("code", """sm = pd.read_csv(M / "sensor_metrics.csv", dtype={"sensor_id": str})
x = sm[(sm.split == "test") & (sm.period == "all") & (sm.horizon == "60min")]
w = x.pivot(index="sensor_id", columns="model", values="mae")
cols = [c for c in ["xgboost_ha", "gru_local", "gat_identity", "gat", "dcrnn"] if c in w]
w[cols].sort_values(cols[0], ascending=False).head(10)"""),
    ("code", """if {"gat", "gat_identity"} <= set(w.columns):
    g = (w.gat_identity - w.gat).rename("gain_vs_identity").to_frame()
    g = g.join(x.drop_duplicates("sensor_id").set_index("sensor_id")[["graph_degree", "region"]])
    display(g.groupby(pd.cut(g.graph_degree, [0, 4, 8, 12, 16, 100], right=False), observed=True).gain_vs_identity.describe())
    display(g.groupby("region").gain_vs_identity.mean().sort_values())
display(Image(filename=str(F / "fig_error_by_degree.png"), width=600))"""),
    ("code", """r = pd.read_csv(M / "region_metrics.csv")
r[(r.split == "test") & (r.period == "all") & (r.horizon == "60min")].pivot(index="region", columns="model", values="mae")[cols]"""),
    ("code", """for f in ["fig_map_morning_60min.png", "fig_map_evening_60min.png", "fig_congestion_map_evening_15min.png"]:
    display(Image(filename=str(F / f), width=800))"""),
    ("md", "## 7. Proxy congestion onsets"),
    ("code", """ev = pd.read_csv(M / "events.csv")
ev.pivot(index="model", columns="horizon", values="recall_onset")"""),
    ("code", """on = pd.read_csv(M / "onset_errors.csv")
on.pivot(index="model", columns="horizon", values="mae_onset")"""),
    ("code", """display(Image(filename=str(F / "fig_case_study.png"), width=800))"""),
    ("md", "## 8. Seeds and attention"),
    ("code", """display(Image(filename=str(F / "fig_seed_spread.png"), width=800))
if (M / "attention_stats.csv").is_file():
    a = pd.read_csv(M / "attention_stats.csv")
    display(a.groupby("period")[["entropy", "mass_self", "mass_neighbours", "mass_downstream", "mass_upstream"]].mean())
if (F / "fig_attention_example.png").is_file():
    display(Image(filename=str(F / "fig_attention_example.png"), width=800))"""),
    ("md", "## 9. Simulated signal control"),
    ("code", """syn = json.loads((M / "signal_sim_summary.json").read_text())
pd.DataFrame({sc: {p: v["mean_delay_s"] for p, v in d.items()} for sc, d in syn["scenarios"].items()})"""),
    ("code", """p = M / "signal_sim_real_summary.json"
if p.is_file():
    real = json.loads(p.read_text())
    print(real["metadata"]["disclaimer"])
    display(pd.DataFrame(real["policies"]).T)
    display(pd.DataFrame(real["paired_vs_xgboost_ha"]).T)
    display(Image(filename=str(F / "fig_signal_real_forecasts.png"), width=600))"""),
]


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [nbf.v4.new_markdown_cell(t) if k == "md" else nbf.v4.new_code_cell(t) for k, t in CELLS]
    return nb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    nb = build_notebook()
    if args.execute:
        from nbclient import NotebookClient

        NotebookClient(nb, timeout=600, kernel_name="python3",
                       resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
    NB_PATH.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"Saved: {NB_PATH}{' (executed)' if args.execute else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
