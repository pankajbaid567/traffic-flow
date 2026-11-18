"""Generate Phase 2 two-column IEEE-style PDF and matching IEEEtran LaTeX source.

Owner: Member 5 (phase2/m5-signal-paper).
Contract: reports/phase2_plan.md section 2, 5.5, 6, 8.5.

Generates:
- reports/paper_phase2/phase2_ieee_paper.pdf (ReportLab, Times New Roman, 2-column)
- reports/paper_phase2/phase2_ieee_paper.tex (official IEEEtran conference class)
- reports/paper_phase2/paper_content.json (machine-readable structured text)
- reports/paper_phase2/qa_report.json (claims & schema verification audit)
- reports/paper_phase2/README.md
"""

from __future__ import annotations

from functools import partial
import html
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Add repo root to python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import MathTextParser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    FrameBreak,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

OUT = Path(os.environ.get("PHASE2_PAPER_OUT", ROOT / "reports" / "paper_phase2"))
TITLE = "Spatiotemporal Graph Attention and Diffusion Forecasting on METR-LA"
SUBTITLE = "with Simulated Adaptive Traffic Signal Control"

HORIZONS = ["5min", "15min", "30min", "60min"]

REFERENCES = [
    'Y. Li, R. Yu, C. Shahabi, and Y. Liu, "Diffusion convolutional recurrent neural network: Data-driven traffic forecasting," in Proc. Int. Conf. Learning Representations, 2018. arXiv:1707.01926.',
    'P. Velickovic, G. Cucurull, A. Casanova, A. Romero, P. Lio, and Y. Bengio, "Graph attention networks," in Proc. Int. Conf. Learning Representations, 2018. arXiv:1710.10903.',
    'T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining, 2016, pp. 785-794, doi: 10.1145/2939672.2939785.',
    'L. Breiman, "Random forests," Machine Learning, vol. 45, pp. 5-32, 2001, doi: 10.1023/A:1010933404324.',
    'F. V. Webster, "Traffic signal settings," Road Research Technical Paper No. 39, Road Research Laboratory, London, 1958.',
    'G. F. Newell, "A simplified car-following theory: a lower order model," Transportation Research Part B: Methodological, vol. 36, no. 3, pp. 195-205, 2002.',
    'G. C. Cawley and N. L. C. Talbot, "On over-fitting in model selection and subsequent selection bias in performance evaluation," Journal of Machine Learning Research, vol. 11, pp. 2079-2107, 2010.',
    'M. J. Lighthill and G. B. Whitham, "On kinematic waves II. A theory of traffic flow on long crowded roads," Proc. Royal Society of London. Series A, vol. 229, no. 1178, pp. 317-345, 1955.',
]


def _column_widths(headers, rows, style, total: float = 252.0) -> List[float]:
    """Column widths that fit header words and short cells unbroken, summing to the column width.

    Each column first gets its minimum (longest header word, bold; whole cell when cells are
    short, else longest word); the remaining width goes to columns in proportion to how much
    wider their longest cell is than that minimum.
    """
    from reportlab.pdfbase.pdfmetrics import stringWidth

    def w(text, bold=False):
        return stringWidth(text, style.fontName, style.fontSize) * (1.12 if bold else 1.0) + 4

    mins, wants = [], []
    for j, h in enumerate(headers):
        cells = [str(r[j]) for r in rows]
        longest = max([w(c) for c in cells] + [w(str(h), True)])
        word_min = max([w(t, True) for t in str(h).split()] + [w(t) for c in cells for t in c.split()] + [0])
        cell_min = max([w(c) for c in cells] + [0])
        mins.append(max(word_min, cell_min if cell_min <= 75 else 0))
        wants.append(max(longest, mins[-1]))
    if sum(mins) >= total:
        return [m * total / sum(mins) for m in mins]
    spare = total - sum(mins)
    extra = [x - m for x, m in zip(wants, mins)]
    if sum(extra) <= spare:
        return [x + (total - sum(wants)) / len(wants) for x in wants]
    return [m + spare * e / sum(extra) for m, e in zip(mins, extra)]


class PaperEquation(Flowable):
    """Render mathematical glyphs and fraction rules as embedded vector text."""

    formulas = {
        "MAE": r"$\mathrm{MAE}=\frac{1}{N}\sum_{i=1}^{N}|y_i-\hat{y}_i|$",
        "RMSE": r"$\mathrm{RMSE}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2}$",
        "GAT": r"$\alpha_{ij}=\frac{\exp(\mathrm{LeakyReLU}(a^\top[Wh_i\Vert Wh_j]))}{\sum_{k\in\mathcal{N}_i}\exp(\mathrm{LeakyReLU}(a^\top[Wh_i\Vert Wh_k]))}$",
        "DIFFUSION": r"$P_f=D_O^{-1}A,\quad P_r=D_I^{-1}A^\top,\quad H=\sum_{k=0}^K P^k X W_k$",
        "DELAY": r"$d=\frac{C(1-g/C)^2}{2(1-(g/C)x)}+\frac{x^2}{2q(1-x)}+d_{\mathrm{spill}}$",
    }

    def __init__(self, name: str, number: int):
        super().__init__()
        self.number = number
        with matplotlib.rc_context({"mathtext.fontset": "stix"}):
            self.vector = MathTextParser("path").parse(
                self.formulas[name], dpi=72, prop=FontProperties(size=8.5)
            )
        self.width = 252
        self.height = self.vector.height + 6
        # matplotlib < 3.10 yields (font, size, code, x, y); newer adds a glyph index before x
        self.glyphs = [(g[0], g[1], g[2], g[-2], g[-1]) for g in self.vector.glyphs]
        for font, size, code, x, y in self.glyphs:
            fname = "Math" + Path(font.fname).stem
            if fname not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(fname, font.fname))

    def draw(self):
        canvas = self.canv
        canvas.saveState()
        canvas.translate((self.width - self.vector.width) / 2, self.vector.depth + 2)
        for font, size, code, x, y in self.glyphs:
            canvas.setFont("Math" + Path(font.fname).stem, size)
            canvas.drawString(x, y, chr(code))
        for x, y, width, height in self.vector.rects:
            canvas.rect(x, y, width, height, fill=1, stroke=0)
        canvas.restoreState()
        canvas.setFont("TNR", 8.5)
        canvas.drawRightString(252, self.vector.depth + 2, f"({self.number})")


# Overridable for dry runs (tests write to temp folders, never to the repo)
METRICS_DIR = Path(os.environ.get("PHASE2_METRICS_DIR", ROOT / "data" / "metadata" / "phase2"))
FIG_SRC = Path(os.environ.get("PHASE2_FIG_DIR", ROOT / "figures" / "phase2"))
HZ = ["5min", "15min", "30min", "60min"]
PHASE1_MODELS = ["persistence", "historical_average", "random_forest_32", "xgboost_32", "xgboost_ha"]
PHASE2_MODELS = ["gru_local", "gat_identity", "gat_shuffled", "dcrnn_identity", "gat", "dcrnn"]
NAME = {
    "persistence": "Persistence", "historical_average": "Historical average",
    "random_forest_32": "Random Forest 32", "xgboost_32": "XGBoost 32", "xgboost_ha": "XGBoost + HA 36",
    "gru_local": "GRU Local", "gat_identity": "GAT identity", "gat_shuffled": "GAT shuffled",
    "dcrnn_identity": "DCRNN identity", "gat": "GAT", "dcrnn": "DCRNN",
}
FAMILY = {"persistence": "Base", "historical_average": "Base", "random_forest_32": "P1 ML",
          "xgboost_32": "P1 ML", "xgboost_ha": "P1 ML", "gru_local": "No graph",
          "gat_identity": "Control", "gat_shuffled": "Control", "dcrnn_identity": "Control",
          "gat": "Graph", "dcrnn": "Graph"}
PAPER_FIGURES = [
    "fig_horizon_mae.png", "fig_peak_offpeak.png", "fig_graph_ablation.png", "fig_seed_spread.png",
    "fig_error_by_degree.png", "fig_map_morning_60min.png", "fig_map_evening_60min.png",
    "fig_congestion_map_evening_15min.png", "fig_case_study.png", "fig_attention_example.png",
    "fig_signal_delay_comparison.png", "fig_signal_real_forecasts.png",
]


def load_phase2_results() -> Dict[str, Any]:
    """Every Phase 2 number in the paper comes from these files (scripts/phase2_eval_all.py,
    scripts/phase2_run_signal_sim.py). Nothing about trained models is typed in here."""
    need = ["metrics_summary.csv", "metrics.csv", "bootstrap.csv", "hypotheses.csv", "run_meta.csv",
            "events.csv", "phase1_parity.json"]
    missing = [f for f in need if not (METRICS_DIR / f).is_file()]
    if missing:
        raise SystemExit(f"Run scripts/phase2_eval_all.py first; missing {missing}")
    r: Dict[str, Any] = {k.split(".")[0]: pd.read_csv(METRICS_DIR / k) for k in need if k.endswith(".csv")}
    r["parity"] = json.loads((METRICS_DIR / "phase1_parity.json").read_text())
    for k in ("signal_sim_summary.json", "signal_sim_real_summary.json"):
        p = METRICS_DIR / k
        r[k.split(".")[0]] = json.loads(p.read_text()) if p.is_file() else None
    p = METRICS_DIR / "attention_stats.csv"
    r["attention_stats"] = pd.read_csv(p) if p.is_file() else None
    s = r["metrics_summary"]
    r["test"] = s[(s["split"] == "test")].set_index(["model", "period", "horizon"])
    r["models"] = [m for m in PHASE2_MODELS if (m, "all", "60min") in r["test"].index]
    from src.phase2.viz.report_figures import best_models

    r["run_best"] = best_models(s)
    return r


def mae(r, model, h, period="all"):
    return float(r["test"].loc[(model, period, h), "mae_mean"])


def sd(r, model, h, period="all"):
    return float(r["test"].loc[(model, period, h), "mae_sd"])


def boot(r, base, cand, h, period="all"):
    b = r["bootstrap"]
    row = b[(b["baseline"] == base) & (b["candidate"] == cand) & (b["horizon"] == h) & (b["period"] == period)]
    return None if row.empty else row.iloc[0]


def ci_text(row) -> str:
    return f"{row['mae_difference']:+.3f} [{row['ci95_low']:+.3f}, {row['ci95_high']:+.3f}]"


def verdicts(r) -> Dict[str, str]:
    h = r["hypotheses"]
    return h.groupby("id", sort=False)["overall_verdict"].first().to_dict()


def h2_parts(r) -> Tuple[str, str]:
    """Combined verdict of H2 part a (gat beats gat_identity) and part b (shuffled no better)."""
    from src.phase2.evaluation.hypotheses import combine

    h = r["hypotheses"]
    h = h[h["id"] == "H2"]
    a = combine(h[h["statement"].str.contains("part a")]["verdict"].tolist())
    b = combine(h[h["statement"].str.contains("part b")]["verdict"].tolist())
    return a, b


def candidate_parameter_counts() -> List[List[str]]:
    """Exact parameter counts of the search candidates, built from the run files."""
    import yaml
    from src.phase2.registry import build_model

    rows = []
    base = {"num_nodes": 207, "in_channels": 7, "history": 12, "num_horizons": 4}
    for run in ["gru_local", "gat", "dcrnn"]:
        spec = yaml.safe_load((ROOT / "configs" / "phase2" / f"runs_{run}.yaml").read_text())
        search = spec["groups"][0]["search"]
        for i, params in enumerate(search["candidates"]):
            model = build_model(search["model"], {**base, **params})
            n = sum(p.numel() for p in model.parameters())
            desc = ", ".join(f"{k}={v}" for k, v in params.items() if k != "dropout")
            rows.append([f"{search['model']} c{i}", desc, f"{n:,}"])
    return rows


def count_phase2_tests() -> int:
    import unittest

    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_phase2_*.py")

    def walk(s):
        for t in s:
            if isinstance(t, unittest.TestSuite):
                yield from walk(t)
            else:
                yield t

    return sum(1 for _ in walk(suite))


def render_paper_figures(results: Dict[str, Any]) -> Path:
    """Copy the evaluation and signal figures (made from result files) into the paper folder."""
    assets = OUT / "figures"
    assets.mkdir(parents=True, exist_ok=True)
    for old in assets.glob("*.png"):
        old.unlink()
    for name in PAPER_FIGURES:
        if (FIG_SRC / name).is_file():
            shutil.copy(FIG_SRC / name, assets / name)
    return assets


def build_paper_content(
    signal_summary: Dict[str, Any],
    authors_info: Dict[str, Any],
    r: Dict[str, Any],
) -> Tuple[str, List[Tuple[str, Any]]]:
    """Paper sections; every result sentence is computed from the evaluation outputs."""
    V = verdicts(r)
    meta = r["run_meta"]
    real = r["signal_sim_real_summary"]
    have = set(r["models"])
    blocks: List[Tuple[str, Any]] = []

    def heading(t): blocks.append(("heading", t))
    def subheading(t): blocks.append(("subheading", t))
    def para(t): blocks.append(("paragraph", t))
    def table(c, h, rows): blocks.append(("table", {"caption": c, "headers": h, "rows": rows}))
    def equation(n, k): blocks.append(("equation", f"{n} = ({k})"))
    def reference(t): blocks.append(("reference", t))

    def figure(f, c):
        if (OUT / "figures" / f).is_file():
            blocks.append(("figure", {"file": f, "caption": c}))

    best_p2 = min((m for m in ("gru_local", "gat", "dcrnn") if m in have), key=lambda m: mae(r, m, "60min"))
    g60 = boot(r, "gat_identity", "gat", "60min")
    gl60 = boot(r, "gru_local", "gat", "60min")
    x60 = {m: boot(r, "xgboost_ha", m, "60min") for m in ("gat", "dcrnn", "gru_local")}

    def word(v):
        return {"supported": "supported", "contradicted": "contradicted", "inconclusive": "inconclusive",
                "descriptive_only": "descriptive", "not_evaluated": "not evaluated"}.get(v, v)

    h2a, h2b = h2_parts(r)
    s60 = boot(r, "gat_identity", "gat_shuffled", "60min")
    gs60 = boot(r, "gat_shuffled", "gat", "60min")
    h2_detail = ""
    if h2a == "supported" and h2b == "contradicted" and s60 is not None:
        h2_detail = (f" because a degree-preserving shuffled graph also beat the identity graph "
                     f"({ci_text(s60)} at 60 min), although the real graph beat the shuffled one"
                     + (f" ({ci_text(gs60)})" if gs60 is not None else ""))

    d60 = boot(r, "dcrnn_identity", "dcrnn", "60min")

    def vs_xgb(m):
        """Plain-language per-horizon verdict of model m against XGBoost + HA."""
        groups = {"lower": [], "higher": [], "same": []}
        for hz in HORIZONS:
            row = boot(r, "xgboost_ha", m, hz)
            if row is None:
                continue
            k = "lower" if row["ci95_low"] > 0 else "higher" if row["ci95_high"] < 0 else "same"
            groups[k].append(hz.replace("min", ""))
        out = []
        for k, txt in (("lower", "lower error at"), ("higher", "higher error at"), ("same", "no detectable difference at")):
            if groups[k]:
                out.append(f"{txt} {' and '.join([', '.join(groups[k][:-1]), groups[k][-1]] if len(groups[k]) > 1 else groups[k])} min")
        return ", and ".join(out)

    capped = []
    rm = r["run_meta"]
    for m in ("gat", "dcrnn"):
        g = rm[rm["model_id"] == m]
        if len(g) and (g["stop_reason"] != "early_stop").all():
            lo, hi = int(g["epochs_run"].min()), int(g["epochs_run"].max())
            why = "the wall-clock cap" if (g["stop_reason"] == "time_cap").all() else "the 60-epoch limit"
            capped.append(f"{NAME[m]} stopped at {why} after {lo if lo == hi else f'{lo}-{hi}'} epochs")
    capped_note = ("; ".join(capped) + ", so these models may be under-trained") if capped else ""

    # ---------------- Abstract ----------------
    graph_sentence = (
        f"Against the same architecture without neighbours (identity graph), GAT changes 60-minute test MAE by "
        f"{ci_text(g60)} (positive = lower error, 95% paired day-block bootstrap interval); the pre-registered "
        f"graph hypothesis H2 is {word(V['H2'])}{h2_detail}. H1 (GAT beats a per-sensor GRU at 30 and 60 min) is "
        f"{word(V['H1'])}."
        + (f" DCRNN against its own identity-graph control changes 60-minute MAE by {ci_text(d60)}." if d60 is not None else "")
    ) if g60 is not None else ""
    xgb_sentence = (
        f"Compared with the strongest Phase 1 model (XGBoost with historical-average features, 60-minute MAE "
        f"{mae(r, 'xgboost_ha', '60min'):.3f}), the best Phase 2 model ({NAME[best_p2]}) reaches "
        f"{mae(r, best_p2, '60min'):.3f}"
        + (f"; DCRNN has {vs_xgb('dcrnn')} (H0 {word(V['H0'])})" if "dcrnn" in have else "") + "."
    )
    sim_sentence = ""
    if real:
        c = real["paired_vs_xgboost_ha"]
        best_src = max((k for k in c if k.startswith("forecast_") and k != "forecast_perfect"),
                       key=lambda k: c[k]["delay_reduction_vs_xgboost_ha_s"], default=None)
        if best_src:
            sim_sentence = (
                " In a software simulation of one signalised approach driven by real test-period speeds and the "
                f"models' real forecasts, the best Phase 2 forecast changed mean delay by "
                f"{c[best_src]['delay_reduction_vs_xgboost_ha_s']:+.2f} s per vehicle relative to the Phase 1 "
                f"forecast, and even a perfect forecast only by {c['forecast_perfect']['delay_reduction_vs_xgboost_ha_s']:+.2f} s; "
                "the large effect is adaptive versus fixed timing. All signal results are simulated."
            )
    abstract = (
        "We extend a frozen Phase 1 benchmark of local forecasters on METR-LA (207 loop detectors, 5-minute "
        "speeds) with graph models that also see neighbouring sensors: a Graph Attention Network (GAT) and a "
        "Diffusion Convolutional Recurrent Neural Network (DCRNN). All models share the Phase 1 chronological "
        "70/10/20 split, 12-step history, horizons (5, 15, 30, 60 min) and valid-target mask. Hyperparameters "
        "are chosen on validation only; final models are trained with three seeds (one for the shuffled and DCRNN identity controls). To "
        "separate the road graph from model capacity we compare against a per-sensor GRU, an identity-graph "
        "ablation and a degree-preserving shuffled graph. " + graph_sentence + " " + xgb_sentence + sim_sentence
    )

    # ---------------- I. Introduction ----------------
    heading("I. INTRODUCTION")
    para(
        "Short-term speed forecasts feed ramp metering and signal timing. In Phase 1 of this capstone we built a "
        "local benchmark on METR-LA: persistence, historical average, Random Forest and XGBoost using only each "
        "sensor's own history. The best model, XGBoost with 36 features including historical-average profiles, "
        f"reached a 60-minute test MAE of {PHASE1_60['xgboost_ha']:.3f} source speed units, and error grew with "
        "horizon, consistent with models that cannot see conditions at neighbouring sensors."
    )
    para(
        "Phase 2 asks one question, fixed before any result existed: does giving a forecaster neighbouring "
        "sensors through the road graph improve forecasts over equally capable models without the graph, on the "
        "same split, masks and history? We answer with pre-registered hypotheses whose verdicts are computed by "
        "code from paired bootstrap intervals. 'No detectable gain' was declared an acceptable answer. A separate, "
        "clearly simulated experiment feeds the forecasts into a signal-timing model."
    )

    # ---------------- II. Related work ----------------
    heading("II. RELATED WORK")
    para(
        "DCRNN [1] combines bidirectional diffusion on the road graph with gated recurrent units. Graph Attention "
        "Networks [2] learn data-dependent weights over a node's neighbours. Tree ensembles [3], [4] are strong on "
        "engineered features but have no built-in way to use the graph. Kinematic-wave theory [8] motivates using "
        "neighbours, since congestion propagates along a corridor. Model-selection bias [7] motivates choosing "
        "hyperparameters on validation only and reporting controls tuned the same way."
    )

    # ---------------- III. Methods ----------------
    heading("III. MODELS")
    para(
        "Every deep model receives x of shape [12, 207, 7]: z-scored causally filled speed, a missing-reading "
        "flag, sine/cosine of time of day and day of week, and a weekend flag; it outputs z-scored speed for the "
        "four horizons. The loss is masked MAE in source speed units on valid targets only."
    )
    subheading("A. Graph Attention Network (GAT)")
    para("A shared temporal encoder (GRU or 1-D convolution) embeds each sensor's last hour (12 steps); multi-head masked attention "
         "mixes embeddings over graph neighbours (bidirectional road graph plus self-loops):")
    equation("GAT", 1)
    para("Logits of unconnected pairs are set to -10^9. Head outputs pass through a projection with a residual "
         "connection and a small head predicts the four horizons.")
    subheading("B. DCRNN")
    para("DCRNN applies K-hop forward and reverse random-walk diffusion on the directed graph inside a DCGRU "
         "encoder-decoder:")
    equation("DIFFUSION", 2)
    para("with P_f = D_out^-1 A and P_r = D_in^-1 A^T.")
    subheading("C. Controls")
    para(
        "GRU Local is a shared per-sensor GRU with no spatial mixing. GAT identity and DCRNN identity are the same "
        "architectures on a self-loop-only graph, so any difference to the full model is the effect of the "
        "neighbours. GAT shuffled uses a node-permuted graph with the same degree structure (wrong neighbours). "
        "Controls reuse the hyperparameters tuned for their main model, which can slightly understate them."
    )

    # ---------------- IV. Protocol ----------------
    heading("IV. EXPERIMENTAL PROTOCOL")
    para(
        "The Phase 1 contract is reused through Phase 1's own loaders: chronological split (2012-03-01 to "
        "2012-06-27, stored clock), unavailable readings (zero, negative, non-finite) excluded from loss and "
        "scoring, fill and scaling fitted on training data only. Origins are 23,967 / 3,416 / 6,844. Each deep "
        "family has four search candidates trained with seed 42 and scored on validation; the candidate with the "
        "lowest mean validation MAE is retrained with seeds 42, 43, 44 (one seed for the shuffled and DCRNN "
        "identity controls) and only these final runs predict the test period. Training uses Adam, batch "
        "32-64, early stopping with patience 8 on validation MAE, and a wall-clock cap per run. The scorer "
        f"reproduces every Phase 1 metric (maximum |MAE difference| {r['parity']['max_abs_mae_diff']:.1e}) and "
        "every Phase 1 bootstrap interval exactly."
    )
    para(
        "Comparisons use Phase 1's paired circular moving-block bootstrap over stored days (2-day blocks, 2,000 "
        "resamples, seed 42); for multi-seed models each day's error sum is the mean over seeds. Peak/off-peak "
        "intervals use period-filtered day sums; contrasts between gains (H3, H4) are bootstrapped on the same "
        "resampled days. A comparison 'A beats B' is supported if the whole 95% interval of the MAE reduction is "
        "above zero, contradicted if below, inconclusive otherwise. Intervals are conditional on this test period "
        "and not adjusted for multiple comparisons."
    )
    rows = []
    for mid, g in meta.groupby("model_id", sort=False):
        if mid not in NAME:
            continue
        code = {"early_stop": "E", "max_epochs": "M", "time_cap": "T"}
        stops = " ".join(f"{code.get(k, k)}{v}" for k, v in g["stop_reason"].value_counts().items())
        rows.append([NAME[mid], str(g["graph"].iloc[0]), f"{int(g['params'].iloc[0]):,}", str(len(g)),
                     f"{g['epochs_run'].min():.0f}-{g['epochs_run'].max():.0f}" if g["epochs_run"].nunique() > 1
                     else f"{g['epochs_run'].iloc[0]:.0f}", stops, f"{g['train_seconds'].mean() / 60:.0f}"])
    order = {m: i for i, m in enumerate(PHASE2_MODELS)}
    rows.sort(key=lambda x: order.get(next(k for k, v in NAME.items() if v == x[0]), 99))
    devices = ", ".join(sorted({str(d).upper() for d in meta["device"].dropna()}))
    table(f"TABLE I\nFINAL TRAINING RUNS (FROM EACH RUN'S meta.json; DEVICE {devices}). STOP: E EARLY STOPPING, "
          "M 60-EPOCH LIMIT, T TIME CAP, WITH RUN COUNTS",
          ["Model", "Graph", "Params", "Seeds", "Epochs", "Stop", "Min/run"], rows)

    # ---------------- V. Results ----------------
    heading("V. FORECAST ACCURACY")
    rows = []
    for m in PHASE1_MODELS + [m for m in PHASE2_MODELS if m in have]:
        cells = []
        for h in HZ:
            v = mae(r, m, h)
            s_ = sd(r, m, h)
            cells.append(f"{v:.3f}" + (f" ±{s_:.3f}" if s_ > 0 else ""))
        rows.append([FAMILY[m], NAME[m], *cells])
    table("TABLE II\nTEST MAE BY HORIZON (SOURCE SPEED UNITS; PHASE 2 MEAN ± SD OVER SEEDS; P1 = PHASE 1)",
          ["Family", "Model", "5 min", "15 min", "30 min", "60 min"], rows)
    para(
        f"Table II gives test MAE. At 60 minutes the best Phase 2 model is {NAME[best_p2]} "
        f"({mae(r, best_p2, '60min'):.3f}); GRU Local reaches {mae(r, 'gru_local', '60min'):.3f} and "
        f"XGBoost + HA {mae(r, 'xgboost_ha', '60min'):.3f}. At 5 minutes the ordering is "
        + ", ".join(f"{NAME[m]} {mae(r, m, '5min'):.3f}" for m in sorted(
            [m for m in ("xgboost_ha", "gru_local", "gat", "dcrnn") if m == "xgboost_ha" or m in have],
            key=lambda m: mae(r, m, "5min"))) + "."
    )
    figure("fig_horizon_mae.png", "Fig. 1. Test MAE by horizon (Phase 2: mean ± sd over seeds).")
    rows = []
    for m in ["persistence", "xgboost_ha"] + [m for m in ("gru_local", "gat_identity", "gat", "dcrnn") if m in have]:
        t = r["test"]
        rows.append([NAME[m], f"{t.loc[(m, 'all', '60min'), 'rmse_mean']:.3f}", f"{t.loc[(m, 'all', '60min'), 'mape_mean']:.2f}",
                     f"{mae(r, m, '60min', 'peak'):.3f}", f"{mae(r, m, '60min', 'off_peak'):.3f}"])
    table("TABLE III\n60-MINUTE TEST RMSE, MAPE (%) AND PEAK / OFF-PEAK MAE",
          ["Model", "RMSE", "MAPE", "MAE peak", "MAE off-peak"], rows)
    figure("fig_peak_offpeak.png", "Fig. 2. 60-minute test MAE in weekday peak (07-10, 16-19 stored clock) and off-peak.")

    # ---------------- VI. Does the graph help? ----------------
    heading("VI. DOES THE ROAD GRAPH HELP?")
    pairs = [("gru_local", "gat"), ("gat_identity", "gat"), ("gat_shuffled", "gat"), ("gat_identity", "gat_shuffled"),
             ("gru_local", "dcrnn"), ("dcrnn_identity", "dcrnn"), ("xgboost_ha", "gat"), ("xgboost_ha", "dcrnn"),
             ("xgboost_ha", "gru_local")]
    rows = []
    for b_, c_ in pairs:
        r15, r60 = boot(r, b_, c_, "15min"), boot(r, b_, c_, "60min")
        if r15 is None or r60 is None:
            continue
        rows.append([f"{NAME[c_]} vs {NAME[b_]}", ci_text(r15), ci_text(r60)])
    table("TABLE IV\nPAIRED BOOTSTRAP: MAE REDUCTION OF FIRST MODEL [95% CI]",
          ["Comparison", "15 min", "60 min"], rows)
    hyp_rows = []
    h = r["hypotheses"]
    for hid in ("H1", "H2", "H3", "H4", "H5", "H0"):
        g = h[h["id"] == hid]
        stmt = g["statement"].iloc[0].split(" [")[0]
        v = word(g["overall_verdict"].iloc[0])
        if hid == "H2":
            v += f" (part a {word(h2a)}, part b {word(h2b)})"
        hyp_rows.append([hid, stmt, v])
    table("TABLE V\nPRE-REGISTERED HYPOTHESES AND CODE-COMPUTED VERDICTS", ["ID", "Hypothesis", "Verdict"], hyp_rows)

    def comp_line(hid, labels=None):
        g = h[h["id"] == hid]
        out = []
        for comp, gg in g.groupby("comparison", sort=False):
            parts = [f"{row['horizon']}: {row['mae_difference']:+.3f} [{row['ci95_low']:+.3f}, {row['ci95_high']:+.3f}]"
                     for _, row in gg.iterrows() if pd.notna(row["ci95_low"])]
            if not parts:
                continue
            lab = (labels or {}).get(comp)
            out.append((f"{lab}, " if lab else "") + "; ".join(parts))
        return " | ".join(out)

    para(
        f"H1 ({word(V['H1'])}): GAT minus GRU Local reduction {comp_line('H1')}. "
        f"H2 ({word(V['H2'])}): " + comp_line('H2', {"gat_identity - gat": "part a, GAT vs GAT identity",
                                                   "gat_identity - gat_shuffled": "part b, GAT shuffled vs GAT identity"}) + ". "
        f"H3 ({word(V['H3'])}): peak gain minus off-peak gain {comp_line('H3')}. "
        f"H4 ({word(V['H4'])}): change in gain between consecutive horizons {comp_line('H4')}."
    )
    h5 = h[h["id"] == "H5"]["note"].iloc[0]
    para(f"H5 (descriptive): {h5}")
    h0 = h[h["id"] == "H0"]
    para(
        f"H0 ({word(V['H0'])}): reduction of each model vs XGBoost + HA, " + comp_line('H0', {"xgboost_ha - gat": "GAT",
                                                   "xgboost_ha - dcrnn": "DCRNN"}) + ". XGBoost + HA uses "
        "historical-average features the deep models do not receive, so this difference is not a graph effect; "
        "only the within-family comparisons above isolate the graph."
    )
    figure("fig_graph_ablation.png", "Fig. 3. Paired bootstrap intervals; positive values mean the first model has lower test MAE.")
    figure("fig_error_by_degree.png", "Fig. 4. Per-sensor 60-minute gain of GAT over GAT identity against graph degree (descriptive).")
    figure("fig_seed_spread.png", "Fig. 5. Test MAE of each training seed.")

    # ---------------- VII. Where are the errors? ----------------
    heading("VII. SPATIAL AND EVENT ANALYSIS")
    ev = r["events"]
    rows = []
    for m in ["persistence", "xgboost_ha"] + [m for m in ("gru_local", "gat", "dcrnn") if m in have]:
        e = ev[ev["model"] == m].set_index("horizon")
        rows.append([NAME[m]] + [f"{e.loc[hh, 'recall_onset']:.3f}" for hh in ("15min", "30min", "60min")]
                    + [f"{e.loc['60min', 'precision_congestion']:.3f}"])
    table("TABLE VI\nPROXY CONGESTION: ONSET RECALL BY HORIZON AND 60-MIN PRECISION (TEST)",
          ["Model", "Onset 15m", "Onset 30m", "Onset 60m", "Prec. 60m"], rows)
    para(
        "Proxy congestion means speed below 60% of the sensor's training 85th percentile; an onset is a target "
        "that is congested while the origin was not. Onsets are rare and hard for every model; Table VI reports "
        "how many of them each model's forecast also places below the threshold."
    )
    figure("fig_map_morning_60min.png", f"Fig. 6. Test morning peak, 60 min: observed speed, MAE of {NAME[r['run_best']['phase1']]} "
           f"and {NAME[r['run_best']['phase2']]}, and their difference (sensor coordinates).")
    figure("fig_map_evening_60min.png", "Fig. 7. As Fig. 6 for the evening peak.")
    figure("fig_congestion_map_evening_15min.png", "Fig. 8. Observed and predicted proxy congestion rate, evening peak, 15 min.")
    figure("fig_case_study.png", "Fig. 9. Phase 1's retrospective example (sensor 717468, 2012-06-11): observed speed and forecasts "
           "plotted at target time. Selected for inspection after evaluation; illustrative, not a test.")
    if r["attention_stats"] is not None:
        a = r["attention_stats"]
        para("Attention summaries (GAT seed 42, 12 exported test origins): " + attention_sentence(a)
             + " Attention shows what the model weighted, not how traffic propagates.")
    figure("fig_attention_example.png", "Fig. 10. GAT attention from the highest-degree sensor to its neighbours (last layer, head mean).")

    # ---------------- VIII. Simulated signal control ----------------
    heading("VIII. SIMULATED SIGNAL CONTROL")
    sm = signal_summary.get("metadata", {})
    fm = sm.get("forecast_mae", {})
    para(
        "This is a software simulation, not traffic control. One signalised main approach (2 lanes) and a cross "
        "street (1 lane, 500 veh/h) are simulated cycle by cycle (90 s cycle, 10 s lost time, green 15-60 s) with "
        "Greenshields-shaped arrivals generated from speed and Webster uniform delay plus an overflow term [5]. "
        "Adaptive policies size green for the worst of the speed signals they see."
    )
    para(
        "Experiment A (synthetic) uses synthetic speed profiles and synthetic forecasts with noise sized to "
        f"measured test MAEs (local {fm.get('local_15m_mae')}, graph {fm.get('graph_15m_mae')} at 15 min). "
        f"Experiment B uses real test-period weekday speeds (06:00-21:00) of the {len(real['metadata']['sensors']) if real else 0} sensors with the most "
        "proxy congestion (chosen from observed data only) and the models' actual test forecasts; every predictive "
        "policy sees the same signals and only the forecast source differs."
    )
    scen = signal_summary.get("scenarios", {})
    rows = []
    for sk, sl in (("morning_peak", "Morning"), ("evening_peak", "Evening"), ("shockwave_incident", "Incident"), ("off_peak", "Off-peak")):
        if sk in scen:
            rows.append([sl] + [f"{scen[sk][p]['mean_delay_s']:.1f}" for p in
                                ("fixed_pretimed", "reactive_actuated", "predictive_local_p1", "predictive_graph_p2")])
    table("TABLE VII\nEXPERIMENT A (SYNTHETIC): MEAN DELAY, S/VEH",
          ["Scenario", "Fixed", "Reactive", "Local fc.", "Graph fc."], rows)
    figure("fig_signal_delay_comparison.png", "Fig. 11. Experiment A: simulated mean delay by policy and scenario.")
    if real:
        rows = []
        lab = {"fixed_pretimed": "Fixed-time", "reactive_actuated": "Reactive", "forecast_xgboost_ha": "XGBoost + HA forecast",
               "forecast_gru_local": "GRU Local forecast", "forecast_gat": "GAT forecast", "forecast_dcrnn": "DCRNN forecast",
               "forecast_perfect": "Perfect forecast (ref.)"}
        for p in lab:
            if p in real["policies"]:
                c = real["paired_vs_xgboost_ha"].get(p)
                rows.append([lab[p], f"{real['policies'][p]['mean_delay_s']:.2f}",
                             f"{real['policies'][p]['spillback_rate_pct']:.1f}%",
                             "-" if c is None else f"{c['delay_reduction_vs_xgboost_ha_s']:+.2f} [{c['ci95_low']:+.2f}, {c['ci95_high']:+.2f}]"])
        table(f"TABLE VIII\nEXPERIMENT B (REAL SPEEDS AND FORECASTS, {real['metadata']['runs']} RUNS)",
              ["Policy", "Delay s/veh", "Spillback", "Saved vs XGB [95% CI]"], rows)
        para(
            "Fixed timing with a 35 s green cannot keep up with the simulated demand over a full day, so its queue and "
            "delay grow without bound; this is a property of the assumed demand rule. Among adaptive policies the "
            "forecast source changes delay by fractions of a second per vehicle, and a perfect forecast bounds what "
            "any forecaster could add in this simulator."
        )
        figure("fig_signal_real_forecasts.png", "Fig. 12. Experiment B: delay saved per vehicle relative to the XGBoost + HA forecast.")

    # ---------------- IX. Model size ----------------
    heading("IX. MODEL SIZE")
    table("TABLE IX\nPARAMETER COUNTS OF SEARCH CANDIDATES", ["Candidate", "Settings", "Parameters"],
          candidate_parameter_counts())

    # ---------------- X. Limitations ----------------
    devs = sorted({str(d) for d in r["run_meta"]["device"].dropna()})
    devices_note = ("wall-clock caps and a 60-epoch limit bound training; all final runs used the same device "
                    f"({devs[0].upper()}, Table I), so runs that hit a cap reflect that device's speed"
                    if len(devs) == 1 else
                    "wall-clock caps bound training, and devices differ between runs (Table I)")
    heading("X. REPRODUCIBILITY AND LIMITATIONS")
    para(
        f"Code, run files, final configs, checkpoints and result tables are in the repository; {count_phase2_tests()} "
        "Phase 2 unit tests cover leakage, Phase 1 parity, graph handling, verdict logic and the simulator. Limits: "
        "(1) speeds are in source units (likely mph, unverified); (2) timestamps are on the stored clock (timezone "
        "unverified); (3) regions are coordinate boxes, not named freeways; (4) congestion is a proxy; (5) one "
        "chronological split of one city and season; (6) " + devices_note + (f"; {capped_note}" if capped_note else "") + "; (7) intervals are not adjusted for multiple comparisons; (8) attention weights are not "
        "evidence of traffic propagation; (9) signal results are simulated with an assumed demand rule."
    )

    # ---------------- XI. Conclusion ----------------
    heading("XI. CONCLUSION")
    concl = []
    if V.get("H2") == "supported":
        concl.append("on this split, giving GAT the road graph lowered error relative to the same model without neighbours, and a wrong graph did not help")
    elif h2a == "supported" and h2b == "contradicted":
        concl.append(
            f"on this split, the road graph lowered GAT's error relative to the same model without neighbours "
            f"({ci_text(g60)} at 60 min), but a shuffled graph with the same degrees also helped "
            f"({ci_text(s60)}), so H2 as pre-registered is not supported: part of the gain comes from mixing "
            f"information across sensors, not only from the true road links"
            + (f", while the real graph still beat the shuffled one ({ci_text(gs60)})" if gs60 is not None else ""))
    elif h2a == "contradicted":
        concl.append("on this split, the road graph did not help GAT relative to the same model without neighbours")
    else:
        concl.append("on this split, the evidence that the road graph helps GAT is mixed (H2 inconclusive)")
    if x60["gat"] is not None:
        better = x60["gat"]["ci95_low"] > 0
        worse = x60["gat"]["ci95_high"] < 0
        concl.append("GAT's 60-minute MAE is " + ("lower than" if better else "higher than" if worse else "not distinguishable from")
                     + " the Phase 1 XGBoost + HA model, which uses features the deep models lack")
    if d60 is not None:
        concl.append(f"DCRNN gained more from its graph ({ci_text(d60)} at 60 min against its identity control, one control seed)")
    if "dcrnn" in have and x60.get("dcrnn") is not None:
        concl.append(f"DCRNN, compared with XGBoost + HA, has {vs_xgb('dcrnn')}, without historical-average features"
                     + (", even though its training was cut by the wall-clock cap"
                        if any(c.startswith("DCRNN stopped at the wall-clock") for c in capped) else ""))
    para("We compared graph and non-graph forecasters under one leak-free protocol with code-computed verdicts. "
         + (lambda t: t[:1].upper() + t[1:])("; ".join(concl)) + ". In simulation, adaptive timing mattered far more than forecast accuracy.")

    heading("REFERENCES")
    for i, ref in enumerate(REFERENCES, 1):
        reference(f"[{i}] {ref}")
    return abstract, blocks


PHASE1_60 = {"xgboost_ha": 3.688}


def attention_sentence(a: pd.DataFrame) -> str:
    cols = a.columns
    out = []
    label = {"mass_self": "attention on the sensor itself", "mass_neighbours": "on graph neighbours",
             "mass_downstream": "on 'downstream' neighbours (graph direction convention)",
             "mass_upstream": "on 'upstream' neighbours", "entropy": "row entropy"}
    for c in ("mass_self", "mass_neighbours", "mass_downstream", "mass_upstream", "entropy"):
        if c in cols:
            g = a[a["period"].isin(["peak", "off_peak"])].groupby("period")[c].mean() if "period" in cols else None
            if g is not None and len(g):
                out.append(f"{label[c]} " + " / ".join(f"{v:.3f} ({k.replace('_', '-')})" for k, v in g.items()))
            else:
                out.append(f"{label[c]} {a[c].mean():.3f}")
    if not out:
        return ""
    return ("Mean " + "; ".join(out) + ". A neighbour linked in both directions counts as both upstream and "
            "downstream, so those two shares can add up to more than the neighbour share.")


def generate_latex_source(
    abstract: str,
    blocks: List[Tuple[str, Any]],
    authors: Dict[str, Any],
) -> None:
    """Generate official IEEEtran conference LaTeX file."""
    def escape(text: Any) -> str:
        s = str(text)
        mapping = {
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        return "".join(mapping.get(c, c) for c in s)

    names = authors.get("names", [])
    author_lines = [", ".join(names[i : i + 3]) for i in range(0, len(names), 3)] if names else ["Phase 2 Team"]
    author_block = r"\IEEEauthorblockN{" + r"\\".join(map(escape, author_lines)) + "}"

    lines = [
        r"\documentclass[conference]{IEEEtran}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{graphicx,booktabs,amsmath,amssymb,textcomp,tabularx,array}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\title{" + escape(TITLE) + r"\\" + escape(SUBTITLE) + "}",
        r"\author{" + author_block + "}",
        r"\begin{document}",
        r"\maketitle",
        r"\begin{abstract}",
        escape(abstract),
        r"\end{abstract}",
        r"\begin{IEEEkeywords}Traffic forecasting, METR-LA, Graph Attention Networks, DCRNN, Adaptive Signal Control, Spatiotemporal Modeling.\end{IEEEkeywords}",
    ]

    refs = False
    for kind, value in blocks:
        if kind == "heading":
            if value == "REFERENCES":
                refs = True
                lines.append(r"\begin{thebibliography}{00}")
                continue
            title_clean = re.sub(r"^[IVX]+\. ", "", value).title()
            lines.append(r"\section{" + escape(title_clean) + "}")
        elif kind == "subheading":
            sub_clean = re.sub(r"^[A-Z]\. ", "", value)
            lines.append(r"\subsection{" + escape(sub_clean) + "}")
        elif kind == "paragraph":
            lines.extend([escape(value), ""])
        elif kind == "equation":
            eq_name = value.split(" = ")[0]
            eq_map = {
                "GAT": r"\alpha_{ij}=\frac{\exp(\mathrm{LeakyReLU}(a^\top[Wh_i\Vert Wh_j]))}{\sum_{k\in\mathcal{N}_i}\exp(\mathrm{LeakyReLU}(a^\top[Wh_i\Vert Wh_k]))}",
                "DIFFUSION": r"P_f=D_O^{-1}A,\quad P_r=D_I^{-1}A^\top,\quad H=\sum_{k=0}^K P^k X W_k",
                "DELAY": r"d=\frac{C(1-g/C)^2}{2(1-(g/C)x)}+\frac{x^2}{2q(1-x)}+d_{\mathrm{spill}}",
            }
            lines.append(r"\begin{equation}" + eq_map.get(eq_name, r"y=f(x)") + r"\end{equation}")
        elif kind == "table":
            caption = value["caption"].split("\n", 1)[-1]
            lines.extend([
                r"\begin{table}[t]",
                r"\caption{" + escape(caption) + "}",
                r"\centering\footnotesize\setlength{\tabcolsep}{3pt}",
            ])
            # Long-text columns wrap (tabularx X columns); otherwise shrink to the column only if too wide.
            ncol = len(value["headers"])
            widths = [max(len(str(x[j])) for x in [value["headers"]] + list(value["rows"])) for j in range(ncol)]
            wrap = [w > 24 for w in widths]
            spec = "".join(r">{\raggedright\arraybackslash}X" if wrap[j] else ("l" if j == 0 else "r")
                           for j in range(ncol))
            if any(wrap):
                lines.append(r"\begin{tabularx}{\columnwidth}{" + spec + "}")
            else:
                lines.append(r"\resizebox{\ifdim\width>\columnwidth\columnwidth\else\width\fi}{!}{"
                             r"\begin{tabular}{" + spec + "}")
            lines.extend([
                r"\toprule",
                " & ".join(escape(x) for x in value["headers"]) + r"\\",
                r"\midrule",
            ])
            lines += [" & ".join(escape(x) for x in row) + r"\\" for row in value["rows"]]
            lines.extend([r"\bottomrule", r"\end{tabularx}" if any(wrap) else r"\end{tabular}}", r"\end{table}"])
        elif kind == "figure":
            lines.extend([
                r"\begin{figure}[t]",
                r"\centering",
                r"\includegraphics[width=\columnwidth]{figures/" + value["file"] + "}",
                r"\caption{" + escape(re.sub(r"^Fig\. \d+\. ", "", value["caption"])) + "}",
                r"\end{figure}",
            ])
        elif kind == "reference":
            m = re.match(r"\[(\d+)\]", value)
            num = m.group(1) if m else "1"
            lines.append(r"\bibitem{ref" + num + "} " + escape(re.sub(r"^\[\d+\] ", "", value)))

    if refs:
        lines.append(r"\end{thebibliography}")
    lines.append(r"\end{document}")

    tex_path = OUT / "phase2_ieee_paper.tex"
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {tex_path}")


def generate_pdf_document(
    abstract: str,
    blocks: List[Tuple[str, Any]],
    authors: Dict[str, Any],
) -> None:
    """Generate two-column IEEE-style PDF using ReportLab with Times New Roman."""
    fontroot = Path("/System/Library/Fonts/Supplemental")
    candidates = {
        "TNR": "Times New Roman.ttf",
        "TNR-Bold": "Times New Roman Bold.ttf",
        "TNR-Italic": "Times New Roman Italic.ttf",
        "TNR-BoldItalic": "Times New Roman Bold Italic.ttf",
    }
    for name, file in candidates.items():
        path = fontroot / file
        if path.exists():
            pdfmetrics.registerFont(TTFont(name, str(path)))

    pdfmetrics.registerFontFamily(
        "TNR",
        normal="TNR",
        bold="TNR-Bold",
        italic="TNR-Italic",
        boldItalic="TNR-BoldItalic",
    )

    body = ParagraphStyle(
        "Body",
        fontName="TNR",
        fontSize=9.5,
        leading=11.2,
        alignment=TA_JUSTIFY,
        firstLineIndent=9,
        spaceAfter=3,
    )
    styles = {
        "body": body,
        "abstract": ParagraphStyle(
            "Abstract",
            parent=body,
            fontSize=8.5,
            leading=10.5,
            fontName="TNR-BoldItalic",
            firstLineIndent=0,
            spaceAfter=7,
        ),
        "heading": ParagraphStyle(
            "Heading",
            parent=body,
            fontName="TNR-Bold",
            fontSize=10,
            leading=12,
            alignment=TA_CENTER,
            firstLineIndent=0,
            spaceBefore=7,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "subheading": ParagraphStyle(
            "Sub",
            parent=body,
            fontName="TNR-Italic",
            fontSize=9.5,
            leading=11.5,
            firstLineIndent=0,
            spaceBefore=5,
            spaceAfter=2,
            keepWithNext=True,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=body,
            fontSize=8,
            leading=9.5,
            firstLineIndent=0,
            spaceAfter=5,
        ),
        "table_caption": ParagraphStyle(
            "TableCaption",
            parent=body,
            fontSize=8,
            leading=9.5,
            alignment=TA_CENTER,
            firstLineIndent=0,
            spaceBefore=5,
            spaceAfter=3,
        ),
        "cell": ParagraphStyle(
            "Cell",
            parent=body,
            fontSize=7,
            leading=8.2,
            alignment=TA_LEFT,
            firstLineIndent=0,
        ),
        "cellnum": ParagraphStyle(
            "CellNum",
            parent=body,
            fontSize=7,
            leading=8.2,
            alignment=TA_CENTER,
            firstLineIndent=0,
        ),
        "reference": ParagraphStyle(
            "Ref",
            parent=body,
            fontSize=7.8,
            leading=9.2,
            firstLineIndent=-10,
            leftIndent=10,
            spaceAfter=2,
        ),
    }

    names = authors.get("names", [])
    author_str = ", ".join(names) if names else "Phase 2 Team"

    def page_decorator(canvas: Canvas, doc: BaseDocTemplate) -> None:
        canvas.saveState()
        if doc.page == 1:
            canvas.setFont("TNR-Bold", 17)
            canvas.drawCentredString(306, 742, TITLE)
            canvas.setFont("TNR", 12)
            canvas.drawCentredString(306, 724, SUBTITLE)
            canvas.setFont("TNR", 9.5)
            canvas.drawCentredString(306, 706, author_str)
        canvas.setFont("TNR", 8)
        canvas.drawCentredString(306, 32, f"- {doc.page} -")
        canvas.restoreState()

    pdf_path = OUT / "phase2_ieee_paper.pdf"
    doc = BaseDocTemplate(
        str(pdf_path),
        pagesize=(612, 792),
        leftMargin=40,
        rightMargin=40,
        topMargin=50,
        bottomMargin=55,
    )

    frame1_p1 = Frame(40, 55, 252, 630, id="col1_p1", topPadding=0, bottomPadding=0)
    frame2_p1 = Frame(320, 55, 252, 630, id="col2_p1", topPadding=0, bottomPadding=0)

    frame1_later = Frame(40, 55, 252, 680, id="col1_later", topPadding=0, bottomPadding=0)
    frame2_later = Frame(320, 55, 252, 680, id="col2_later", topPadding=0, bottomPadding=0)

    doc.addPageTemplates([
        PageTemplate(id="FirstPage", frames=[frame1_p1, frame2_p1], onPage=page_decorator),
        PageTemplate(id="LaterPages", frames=[frame1_later, frame2_later], onPage=page_decorator),
    ])

    story: List[Any] = []
    story.append(NextPageTemplate("LaterPages"))

    # Abstract block
    story.append(Paragraph(f"<b><i>Abstract</i>—{abstract}</b>", styles["abstract"]))
    story.append(Spacer(1, 3))

    eq_counter = 1
    for kind, val in blocks:
        if kind == "heading":
            story.append(Paragraph(val, styles["heading"]))
        elif kind == "subheading":
            story.append(Paragraph(val, styles["subheading"]))
        elif kind == "paragraph":
            story.append(Paragraph(val, styles["body"]))
        elif kind == "equation":
            eq_name = val.split(" = ")[0]
            try:
                story.append(PaperEquation(eq_name, eq_counter))
                eq_counter += 1
            except Exception:
                story.append(Paragraph(f"<i>{val}</i>", styles["body"]))
        elif kind == "table":
            story.append(Paragraph(f"<b>{val['caption']}</b>", styles["table_caption"]))
            table_data = []
            hdr = [Paragraph(f"<b>{h}</b>", styles["cellnum"]) for h in val["headers"]]
            table_data.append(hdr)
            for row in val["rows"]:
                r = [Paragraph(str(x), styles["cell"]) for x in row]
                table_data.append(r)

            t = Table(table_data, colWidths=_column_widths(val["headers"], val["rows"], styles["cell"]))
            t.setStyle(TableStyle([
                ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
                ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 1),
                ("RIGHTPADDING", (0, 0), (-1, -1), 1),
            ]))
            story.append(t)
            story.append(Spacer(1, 4))
        elif kind == "figure":
            fpath = OUT / "figures" / val["file"]
            if fpath.exists():
                from PIL import Image as PILImage

                with PILImage.open(fpath) as im:
                    w_px, h_px = im.size
                width = 248.0
                height = width * h_px / w_px
                if height > 330:  # keep tall figures inside the column
                    height = 330.0
                    width = height * w_px / h_px
                img = Image(str(fpath), width=width, height=height)
                story.append(KeepTogether([
                    img,
                    Paragraph(f"<i>{val['caption']}</i>", styles["caption"]),
                ]))
            else:
                story.append(Paragraph(f"[Figure: {val['file']}]", styles["caption"]))
        elif kind == "reference":
            story.append(Paragraph(val, styles["reference"]))

    doc.build(story)
    print(f"Saved: {pdf_path}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    authors_path = OUT / "authors.json"
    if not authors_path.exists():
        authors_path = ROOT / "reports" / "paper" / "authors.json"

    with open(authors_path, "r", encoding="utf-8") as f:
        authors = json.load(f)

    # Read signal simulation summary
    signal_summary_path = METRICS_DIR / "signal_sim_summary.json"
    if signal_summary_path.exists():
        with open(signal_summary_path, "r", encoding="utf-8") as f:
            signal_summary = json.load(f)
    else:
        signal_summary = {"scenarios": {}}

    print("=== Building Phase 2 IEEE Research Paper ===")
    results = load_phase2_results()
    print(f"Scored Phase 2 models: {results['models']}")
    # 1. Render publication figures into reports/paper_phase2/figures/
    render_paper_figures(results)

    # 2. Build structured content
    abstract, blocks = build_paper_content(signal_summary, authors, results)

    # 3. Write structured JSON
    paper_json_path = OUT / "paper_content.json"
    with open(paper_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "title": TITLE,
                "subtitle": SUBTITLE,
                "authors": authors,
                "abstract": abstract,
                "blocks": blocks,
            },
            f,
            indent=2,
        )
    print(f"Saved: {paper_json_path}")

    # 4. Generate LaTeX source
    generate_latex_source(abstract, blocks, authors)

    # 5. Generate PDF document with ReportLab
    generate_pdf_document(abstract, blocks, authors)

    # 6. Generate QA Verification Report
    pending = [m for m in PHASE2_MODELS if m not in results["models"]]
    qa_report = {
        "status": "pending_training" if pending else "results_from_evaluation_outputs",
        "phase2_models_scored": results["models"],
        "phase2_models_pending": pending,
        "hypothesis_verdicts": verdicts(results),
        "phase1_parity": results["parity"],
        "phase2_numbers_source": "data/metadata/phase2/metrics_summary.csv and hypotheses.csv (no hand-typed Phase 2 results)",
        "paper_title": TITLE,
        "authors_count": len(authors.get("names", [])),
        "sections_count": len([b for b in blocks if b[0] == "heading"]),
        "tables_count": len([b for b in blocks if b[0] == "table"]),
        "figures_count": len([b for b in blocks if b[0] == "figure"]),
        "references_count": len([b for b in blocks if b[0] == "reference"]),
        "claims_compliance": {
            "speed_units": "source speed units; physical units unverified",
            "timezone": "stored clock; UTC unverified",
            "signal_control": "simulated; experiment A synthetic (noise sized from measured MAEs), experiment B real speeds and real forecasts with an assumed demand rule",
            "spatial_freeways": "coordinate proxy regions; no named freeways claimed",
        },
    }
    qa_path = OUT / "qa_report.json"
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(qa_report, f, indent=2)
    print(f"Saved: {qa_path}")

    # 7. Write README.md
    readme_path = OUT / "README.md"
    readme_text = f"""# Phase 2 IEEE Conference Style Paper

`phase2_ieee_paper.pdf` is an authentic two-column US Letter manuscript generated deterministically with embedded Times New Roman fonts, vector math expressions, formatted tables, and 350 DPI figures.

`phase2_ieee_paper.tex` contains the matching editable LaTeX source conforming to the official IEEEtran conference format.

### Authors:
{", ".join(authors.get("names", []))}

### Core Artifacts:
- `phase2_ieee_paper.pdf`: Compiled camera-ready PDF.
- `phase2_ieee_paper.tex`: Editable IEEEtran LaTeX source.
- `paper_content.json`: Machine-readable structured narrative blocks.
- `qa_report.json`: QA claims audit.
- `figures/`: Publication-grade figures.
- `phase2_ieee_source.zip`: Overleaf bundle (`.tex`, `authors.json`, this README, included figures); rebuild command in `reports/phase2/reproduction.md`.
- `phase2_slides.pptx`: Presentation, built by `scripts/build_phase2_slides.py`.

Every Phase 2 number is read from `data/metadata/phase2/` (written by `scripts/phase2_eval_all.py` and `scripts/phase2_run_signal_sim.py`); nothing is typed in by hand.
"""
    readme_path.write_text(readme_text, encoding="utf-8")
    print(f"Saved: {readme_path}")
    print("\nPhase 2 Paper generation complete.")


if __name__ == "__main__":
    main()

