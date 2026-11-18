"""Build the Phase 2 presentation (reports/paper_phase2/phase2_slides.pptx) (Member 5).

    python scripts/build_phase2_slides.py

Every number and figure comes from data/metadata/phase2/ and figures/phase2/
(written by scripts/phase2_eval_all.py and scripts/phase2_run_signal_sim.py).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Inches, Pt

M = Path(os.environ.get("PHASE2_METRICS_DIR", ROOT / "data" / "metadata" / "phase2"))
F = Path(os.environ.get("PHASE2_FIG_DIR", ROOT / "figures" / "phase2"))
OUT = Path(os.environ.get("PHASE2_PAPER_OUT", ROOT / "reports" / "paper_phase2")) / "phase2_slides.pptx"
INK = RGBColor(0x0B, 0x0B, 0x0B)
INK2 = RGBColor(0x52, 0x51, 0x4E)
ACCENT = RGBColor(0x2A, 0x78, 0xD6)
NAME = {"persistence": "Persistence", "historical_average": "Historical avg.", "random_forest_32": "Random Forest 32",
        "xgboost_32": "XGBoost 32", "xgboost_ha": "XGBoost + HA 36", "gru_local": "GRU Local",
        "gat_identity": "GAT identity", "gat_shuffled": "GAT shuffled", "dcrnn_identity": "DCRNN identity",
        "gat": "GAT", "dcrnn": "DCRNN"}


def add_title(slide, text, sub=None):
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.8))
    p = tb.text_frame.paragraphs[0]
    p.text = text
    p.font.size, p.font.bold, p.font.color.rgb = Pt(28), True, INK
    if sub:
        p2 = tb.text_frame.add_paragraph()
        p2.text = sub
        p2.font.size, p2.font.color.rgb = Pt(14), INK2


def add_bullets(slide, items, left=0.6, top=1.4, width=6.2, height=5.5, size=16):
    items = [it for it in items if it]
    height = min(height, 7.3 - top)
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = "• " + it
        p.font.size, p.font.color.rgb = Pt(size), INK
        p.space_after = Pt(8)


def add_image(slide, name, left, top, width=None, height=None):
    """Place an image inside the box (width x height, default: to the slide edge), keeping its aspect."""
    path = F / name
    if not path.is_file():
        return
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        aspect = im.width / im.height
    max_w = width or (13.333 - left - 0.2)
    max_h = min(height or 99.0, 7.4 - top)
    w = min(max_w, max_h * aspect)
    slide.shapes.add_picture(str(path), Inches(left), Inches(top), width=Inches(w), height=Inches(w / aspect))


def add_table(slide, header, rows, left=0.6, top=1.4, width=12.0, size=12):
    shape = slide.shapes.add_table(len(rows) + 1, len(header), Inches(left), Inches(top), Inches(width),
                                   Inches(0.35 * (len(rows) + 1)))
    tbl = shape.table
    for j, h in enumerate(header):
        c = tbl.cell(0, j)
        c.text = h
        c.text_frame.paragraphs[0].font.size = Pt(size)
        c.text_frame.paragraphs[0].font.bold = True
    for i, row in enumerate(rows, start=1):
        for j, v in enumerate(row):
            c = tbl.cell(i, j)
            c.text = str(v)
            c.text_frame.paragraphs[0].font.size = Pt(size)


def main() -> int:
    s = pd.read_csv(M / "metrics_summary.csv")
    h = pd.read_csv(M / "hypotheses.csv")
    b = pd.read_csv(M / "bootstrap.csv")
    meta = pd.read_csv(M / "run_meta.csv")
    parity = json.loads((M / "phase1_parity.json").read_text())
    real_p = M / "signal_sim_real_summary.json"
    real = json.loads(real_p.read_text()) if real_p.is_file() else None
    authors = json.loads((ROOT / "reports" / "paper_phase2" / "authors.json").read_text())
    t = s[(s.split == "test") & (s.period == "all")].set_index(["model", "horizon"])
    V = h.groupby("id", sort=False)["overall_verdict"].first()
    from src.phase2.evaluation.hypotheses import combine

    h2 = h[h.id == "H2"]
    h2a = combine(h2[h2.statement.str.contains("part a")].verdict.tolist())
    h2b = combine(h2[h2.statement.str.contains("part b")].verdict.tolist())

    def mae(m, hz="60min"):
        return t.loc[(m, hz), "mae_mean"]

    def bt(base, cand, hz="60min"):
        r = b[(b.baseline == base) & (b.candidate == cand) & (b.horizon == hz) & (b.period == "all")]
        return None if r.empty else r.iloc[0]

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    blank = prs.slide_layouts[6]

    sl = prs.slides.add_slide(blank)
    add_title(sl, "Does the road graph help traffic forecasting?",
              "Phase 2: GAT and DCRNN vs equally capable non-graph models on METR-LA (+ a simulated signal experiment)")
    add_bullets(sl, [", ".join(authors.get("names", [])), authors.get("affiliation", "").strip(),
                     "All results: chronological test split, source speed units, stored clock; signal results simulated."],
                top=3.0, width=12, size=18)

    sl = prs.slides.add_slide(blank)
    add_title(sl, "Question and protocol", "Fixed before any Phase 2 result existed")
    add_bullets(sl, [
        "Same data, split (70/10/20 chronological), 12-step history, horizons 5/15/30/60 min and valid-target mask as Phase 1",
        "Graph models: GAT (attention over neighbours), DCRNN (diffusion on the directed graph)",
        "Controls: GRU Local (no neighbours), GAT/DCRNN on identity graph, GAT on shuffled graph",
        "4 candidates per family chosen on validation only; finals with seeds 42/43/44",
        "Paired day-block bootstrap (2-day blocks, 2000 resamples); verdicts computed by code",
        f"Scorer reproduces Phase 1 exactly (max |ΔMAE| {parity['max_abs_mae_diff']:.1e}, bootstrap identical)",
    ], width=12.2)

    sl = prs.slides.add_slide(blank)
    add_title(sl, "Training runs", "From each run's meta.json")
    rows = []
    for mid, g in meta.groupby("model_id"):
        rows.append([NAME.get(mid, mid), g["graph"].iloc[0], f"{int(g['params'].iloc[0]):,}", len(g),
                     f"{g['epochs_run'].mean():.0f}", ", ".join(g["stop_reason"].unique()),
                     f"{g['train_seconds'].mean() / 60:.0f}", g["device"].iloc[0]])
    add_table(sl, ["Model", "Graph", "Params", "Seeds", "Epochs", "Stop", "Min/run", "Device"], rows)

    sl = prs.slides.add_slide(blank)
    add_title(sl, "Test MAE by horizon")
    add_image(sl, "fig_horizon_mae.png", 0.4, 1.3, width=8.0, height=5.9)
    models = [m for m in ("persistence", "xgboost_ha", "gru_local", "gat", "dcrnn") if (m, "60min") in t.index]
    add_bullets(sl, [f"{NAME[m]}: {mae(m, '15min'):.3f} @15m, {mae(m):.3f} @60m" for m in models],
                left=8.6, top=1.6, width=4.4, size=14)

    sl = prs.slides.add_slide(blank)
    add_title(sl, "Does the graph help? Paired bootstrap", "Positive = first model has lower test MAE")
    add_image(sl, "fig_graph_ablation.png", 0.4, 1.4, width=8.4)
    lines = []
    for base, cand in (("gat_identity", "gat"), ("gru_local", "gat"), ("dcrnn_identity", "dcrnn"), ("xgboost_ha", "gat")):
        r = bt(base, cand)
        if r is not None:
            lines.append(f"{NAME[cand]} vs {NAME[base]} @60m: {r.mae_difference:+.3f} [{r.ci95_low:+.3f}, {r.ci95_high:+.3f}]")
    add_bullets(sl, lines, left=8.9, top=1.6, width=4.2, size=13)

    sl = prs.slides.add_slide(blank)
    add_title(sl, "Pre-registered hypotheses", "Verdicts computed by code from the bootstrap intervals")
    rows = [[hid, h[h.id == hid].statement.iloc[0].split(" [")[0],
             V[hid] + (f" (part a {h2a}, part b {h2b})" if hid == "H2" else "")]
            for hid in ("H1", "H2", "H3", "H4", "H5", "H0")]
    add_table(sl, ["ID", "Hypothesis", "Verdict"], rows, size=14)

    sl = prs.slides.add_slide(blank)
    add_title(sl, "Peak vs off-peak and per-sensor gain")
    add_image(sl, "fig_peak_offpeak.png", 0.3, 1.3, width=6.3)
    add_image(sl, "fig_error_by_degree.png", 6.8, 1.3, width=6.2)

    sl = prs.slides.add_slide(blank)
    add_title(sl, "Where are the errors? (test morning peak, 60 min)", "Sensor coordinates; not named freeways")
    add_image(sl, "fig_map_morning_60min.png", 2.2, 1.2, height=6.1)

    sl = prs.slides.add_slide(blank)
    add_title(sl, "Phase 1's hardest example, revisited", "Sensor 717468, 2012-06-11 (selected after evaluation; illustrative)")
    add_image(sl, "fig_case_study.png", 0.5, 1.5, width=12.2)

    if (F / "fig_attention_example.png").is_file():
        sl = prs.slides.add_slide(blank)
        add_title(sl, "What GAT attends to", "Shows what the model weighted, not how traffic moves")
        add_image(sl, "fig_attention_example.png", 0.5, 1.5, width=12.2)

    sl = prs.slides.add_slide(blank)
    add_title(sl, "Simulated signal control (software only)")
    add_image(sl, "fig_signal_delay_comparison.png", 0.3, 1.3, width=6.5)
    if real:
        add_image(sl, "fig_signal_real_forecasts.png", 7.0, 1.3, width=6.0, height=3.6)
        c = real["paired_vs_xgboost_ha"]
        add_bullets(sl, [f"Real test speeds + real forecasts, {real['metadata']['runs']} runs",
                         f"Perfect forecast saves only {c['forecast_perfect']['delay_reduction_vs_xgboost_ha_s']:+.2f} s/veh vs XGBoost forecast",
                         "Adaptive vs fixed timing is the large effect"], left=7.0, top=5.0, width=6, size=13)

    sl = prs.slides.add_slide(blank)
    add_title(sl, "Conclusions and limits")
    g60 = bt("gat_identity", "gat")
    s60 = bt("gat_identity", "gat_shuffled")
    gs60 = bt("gat_shuffled", "gat")

    def ci(r):
        return f"{r.mae_difference:+.3f} [{r.ci95_low:+.3f}, {r.ci95_high:+.3f}]"

    graph_lines = [f"Real road graph vs no neighbours (GAT, 60 min): {ci(g60)}" if g60 is not None else "Graph effect: see hypotheses"]
    if s60 is not None and gs60 is not None:
        graph_lines.append(f"Shuffled graph vs no neighbours: {ci(s60)}; real vs shuffled: {ci(gs60)} "
                           f"-> H2 {V['H2']} (part a {h2a}, part b {h2b})")
    add_bullets(sl, graph_lines + [
        f"H1 (GAT beats GRU Local at 30/60 min): {V['H1']};  H0 vs XGBoost + HA: {V['H0']}",
        "XGBoost + HA has historical-average features the deep models lack: not a graph comparison",
        "Limits: one split, one city/season; source units and stored clock unverified; proxy congestion; simulated signals",
        "Everything regenerates from the repo: phase2_eval_all.py → phase2_run_signal_sim.py → build_phase2_paper.py",
    ], width=12.2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f"Saved: {OUT} ({len(prs.slides)} slides)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
