"""Generate a two-column IEEE-style PDF and matching IEEEtran LaTeX source.

Content, tables and plots derive from verified evaluation artifacts. The PDF is
rendered with embedded Times New Roman fonts on US Letter; the editable source
uses the official IEEEtran conference class when compiled in a TeX environment.
"""

from __future__ import annotations
from pathlib import Path
import sys, json, re, html

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from reportlab.platypus import (
    BaseDocTemplate,
    PageTemplate,
    Frame,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether,
    NextPageTemplate,
    Flowable,
    FrameBreak,
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.pdfgen.canvas import Canvas
from matplotlib.mathtext import MathTextParser
from matplotlib.font_manager import FontProperties
from functools import partial

OUT = ROOT / "reports/paper"
TITLE = "Local Traffic Speed Forecasting on METR-LA"
SUBTITLE = "A Reproducible Phase 1 Benchmark"
LABELS = {
    "persistence": "Persistence",
    "historical_average": "Historical average",
    "random_forest_32": "RF 32",
    "xgboost_32": "XGBoost 32",
    "xgboost_ha": "XGBoost + HA 36",
}
ORDER = list(LABELS)
HORIZONS = ["5min", "15min", "30min", "60min"]
REFERENCES = [
    'Y. Li, R. Yu, C. Shahabi, and Y. Liu, "Diffusion convolutional recurrent neural network: Data-driven traffic forecasting," in Proc. Int. Conf. Learning Representations, 2018. arXiv:1707.01926.',
    'L. Breiman, "Random forests," Machine Learning, vol. 45, pp. 5-32, 2001, doi: 10.1023/A:1010933404324.',
    'T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining, 2016, pp. 785-794, doi: 10.1145/2939672.2939785.',
    'G. C. Cawley and N. L. C. Talbot, "On over-fitting in model selection and subsequent selection bias in performance evaluation," Journal of Machine Learning Research, vol. 11, pp. 2079-2107, 2010.',
    'P. Velickovic, G. Cucurull, A. Casanova, A. Romero, P. Lio, and Y. Bengio, "Graph attention networks," in Proc. Int. Conf. Learning Representations, 2018. arXiv:1710.10903.',
]


class PaperEquation(Flowable):
    """Render mathematical glyphs and fraction rules as embedded vector text."""

    formulas = {
        "MAE": r"$\mathrm{MAE}=\frac{1}{N}\sum_{i=1}^{N}|y_i-\hat{y}_i|$",
        "RMSE": r"$\mathrm{RMSE}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2}$",
        "MAPE": r"$\mathrm{MAPE}=\frac{100}{N}\sum_{i=1}^{N}\frac{|y_i-\hat{y}_i|}{\max(y_i,10^{-5})}$",
    }

    def __init__(self, name, number):
        super().__init__()
        self.number = number
        with matplotlib.rc_context({"mathtext.fontset": "stix"}):
            self.vector = MathTextParser("path").parse(
                self.formulas[name], dpi=72, prop=FontProperties(size=9)
            )
        self.width = 252
        self.height = self.vector.height + 7
        for font, size, code, x, y in self.vector.glyphs:
            name = "Math" + Path(font.fname).stem
            if name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(name, font.fname))

    def draw(self):
        canvas = self.canv
        canvas.saveState()
        canvas.translate((self.width - self.vector.width) / 2, self.vector.depth + 3)
        for font, size, code, x, y in self.vector.glyphs:
            canvas.setFont("Math" + Path(font.fname).stem, size)
            canvas.drawString(x, y, chr(code))
        for x, y, width, height in self.vector.rects:
            canvas.rect(x, y, width, height, fill=1, stroke=0)
        canvas.restoreState()
        canvas.setFont("TNR", 9)
        canvas.drawRightString(252, self.vector.depth + 3, f"({self.number})")


def figures(m, s, result):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 6.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )
    assets = OUT / "figures"
    assets.mkdir(parents=True, exist_ok=True)
    palette = ["#777777", "#d97706", "#00876c", "#2166ac", "#762a83"]
    fig, ax = plt.subplots(figsize=(3.5, 2.35), constrained_layout=True)
    for model, color in zip(ORDER, palette):
        q = (
            m[(m.split == "test") & (m.period == "all") & (m.model == model)]
            .set_index("horizon")
            .reindex(HORIZONS)
        )
        ax.plot(
            [5, 15, 30, 60],
            q.mae,
            marker="o",
            ms=3,
            lw=1.1,
            label=LABELS[model],
            color=color,
        )
    ax.set(xlabel="Forecast horizon (min)", ylabel="MAE (source speed units)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)
    fig.savefig(assets / "horizon_mae.png", dpi=400)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(3.5, 2.3), constrained_layout=True)
    periods = ["off_peak", "morning_peak", "evening_peak"]
    for i, (model, color) in enumerate(zip(ORDER, palette)):
        q = (
            m[(m.split == "test") & (m.horizon == "15min") & (m.model == model)]
            .set_index("period")
            .reindex(periods)
        )
        ax.bar(
            np.arange(3) + (i - 2) * 0.15, q.mae, 0.14, label=LABELS[model], color=color
        )
    ax.set_xticks(range(3), ["Off peak", "Morning", "Evening"])
    ax.set_ylabel("15-min MAE (source units)")
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(assets / "temporal_mae.png", dpi=400)
    plt.close(fig)
    geo = pd.read_csv(ROOT / "data/metadata/sensors.csv", dtype={"sensor_id": str})
    q = s[
        (s.split == "test")
        & (s.horizon == "15min")
        & (s.model == "xgboost_ha")
        & (s.period == "evening_peak")
    ].merge(geo[["sensor_id", "latitude", "longitude"]], on="sensor_id")
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.4), constrained_layout=True)
    for ax, col, title, cmap, lo, hi in zip(
        axes,
        ["actual_mean", "mae"],
        ["Observed mean speed", "Forecast MAE"],
        ["RdYlGn", "magma_r"],
        [15, 0],
        [70, 12],
    ):
        sc = ax.scatter(
            q.longitude,
            q.latitude,
            c=q[col],
            s=5,
            cmap=cmap,
            vmin=lo,
            vmax=hi,
            rasterized=True,
        )
        ax.set(title=title, xlabel="Longitude")
        fig.colorbar(sc, ax=ax, orientation="horizontal", shrink=0.9, pad=0.03)
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("Latitude")
    fig.savefig(assets / "spatial_errors.png", dpi=400)
    plt.close(fig)
    # A retrospective example is disclosed rather than presented as a causal event.
    example = result["example"]
    c = pd.read_parquet(
        ROOT / "data/processed/predictions/person_c_test_predictions.parquet",
        filters=[("sensor_id", "==", example["sensor_id"])],
    )
    d = pd.read_parquet(
        ROOT / "data/processed/predictions/person_d_test_predictions.parquet",
        filters=[("sensor_id", "==", example["sensor_id"])],
    )
    center = pd.Timestamp(example["center_target_time"])
    t = pd.DatetimeIndex(c.origin_timestamp) + pd.Timedelta(hours=1)
    keep = (t >= center - pd.Timedelta(hours=4)) & (t <= center + pd.Timedelta(hours=4))
    fig, ax = plt.subplots(figsize=(3.5, 2.35), constrained_layout=True)
    ax.plot(
        t[keep],
        c.target_60min.where(c.target_valid_60min).to_numpy()[keep],
        color="black",
        label="Observed",
        lw=1.2,
    )
    for label, values, color in [
        ("Persistence", c.pred_persistence_60min, "#777777"),
        ("RF 32", d.pred_random_forest_60min, "#00876c"),
        ("XGB + HA", c.pred_xgboost_ha_60min, "#762a83"),
    ]:
        ax.plot(t[keep], values.to_numpy()[keep], label=label, color=color, lw=0.9)
    ax.set(xlabel="Target time on stored clock", ylabel="Speed (source units)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(ncol=2)
    ax.grid(alpha=0.2)
    fig.savefig(assets / "forecast_example.png", dpi=400)
    plt.close(fig)
    return assets


def build_content(m, s, ci, result, audit):
    bench = m[(m.split == "test") & (m.period == "all")]

    def score(model, h, metric="mae", period="all"):
        return float(
            m[
                (m.split == "test")
                & (m.horizon == h)
                & (m.model == model)
                & (m.period == period)
            ].iloc[0][metric]
        )

    gains = {
        h: 100 * (1 - score("xgboost_ha", h) / score("persistence", h))
        for h in HORIZONS
    }
    peakpenalty = 100 * (
        score("xgboost_ha", "15min", period="evening_peak")
        / score("xgboost_ha", "15min", period="off_peak")
        - 1
    )
    abstract = (
        f"We present a reproducible Phase 1 benchmark for local traffic-speed forecasting on METR-LA. "
        "The preserved dataset contains 34,272 five-minute timestamps from 207 sensors. A shared causal pipeline generates "
        "32 features, uses chronological training, validation and test partitions, and leaves unavailable targets unimputed. "
        "Persistence, historical averages, Random Forest and two XGBoost configurations are evaluated at 5, 15, 30 and 60 minutes. "
        f"The 36-feature XGBoost extension with training-history averages obtains test MAEs of "
        f'{score("xgboost_ha","5min"):.3f}, {score("xgboost_ha","15min"):.3f}, '
        f'{score("xgboost_ha","30min"):.3f} and {score("xgboost_ha","60min"):.3f}, '
        f'reducing persistence MAE by {gains["5min"]:.1f}% to {gains["60min"]:.1f}%. '
        "Independent prediction-row scoring adds all-model temporal slices, sensor-level forecast-error maps and paired "
        "day-block uncertainty. The 32-feature learner comparison is distinguished from the history-feature extension. "
        "Unknown speed units, timezone and freeway identities are retained as limitations. These results establish a "
        "traceable local baseline for a future graph-model experiment; graph superiority is not evaluated here."
    )
    blocks = []

    def heading(text):
        blocks.append(("heading", text))

    def para(text):
        blocks.append(("paragraph", text))

    def sub(text):
        blocks.append(("subheading", text))

    def table(caption, headers, rows):
        blocks.append(("table", {"caption": caption, "headers": headers, "rows": rows}))

    def figure(file, caption):
        blocks.append(("figure", {"file": file, "caption": caption}))

    heading("I. INTRODUCTION")
    para(
        "Traffic-speed forecasting provides a measurable foundation for congestion analysis. A useful first stage is to determine how far a sensor's own recent measurements and recurring temporal patterns can predict its future speed. This question should be answered before attributing an improvement to network structure or adding simulation and control objectives."
    )
    para(
        "Phase 1 therefore compares local forecasts under a common data contract. Each forecast row describes one sensor at one prediction origin. Models may be fitted to pooled examples from all sensors, but an individual input contains only the target sensor's own history and allowed temporal variables. Neighboring measurements, coordinates, adjacency-derived features and sensor identity are excluded from the shared 32-feature input."
    )
    para(
        "The contribution is a reproducible benchmark with shared quality rules, timestamp boundaries and target masks. Five configurations are scored on identical held-out rows, followed by temporal, sensor and regional error analysis."
    )
    heading("II. RELATED WORK AND SCOPE")
    para(
        "The METR-LA dataset is associated with the traffic-forecasting evaluation in DCRNN [1]. Its spatiotemporal framework uses a directed-graph formulation, whereas this phase deliberately restricts predictive inputs to local observations. Our custom target offsets, sample eligibility and feature design should not be treated as an exact reproduction of the published DCRNN experiment."
    )
    para(
        "Random Forest combines randomized trees [2], while XGBoost provides efficient tree boosting [3]. Their regression configurations have different loss preferences. To limit selection bias [4], this evaluation preserves fitted booster artifacts and fixed RF settings. Graph attention [5] is a candidate for a future spatial experiment."
    )
    heading("III. DATA AND CHRONOLOGICAL PROTOCOL")
    sub("A. Raw data quality and provenance")
    para(
        "The source speed table contains 34,272 ordered five-minute timestamps and 207 sensors, covering March 1 through June 27, 2012 on the stored clock. There are no duplicate or absent timestamps and no NaN, infinite or negative raw readings. The 575,302 exact-zero cells represent 8.109% of measurements and are conservatively treated as unavailable benchmark readings. These include 2,148 all-sensor-zero timestamps in 97 contiguous blocks; their physical causes are not established. Raw bytes are preserved and checked against the acquisition manifest."
    )
    para(
        "The speed table's sensor order matches the available 207 by 207 adjacency inventory and location metadata. This alignment supports later graph work, but graph assets do not enter the Phase 1 predictive features. The inspected metadata does not independently verify physical speed units or timezone. We therefore use source speed units and the stored clock throughout, including the peak definitions."
    )
    table(
        "TABLE I\nSHARED CHRONOLOGICAL PARTITIONS",
        ["Partition", "Grid times", "Origins", "Rows"],
        [
            ["Train", "23,990", "23,967", "4,961,169"],
            ["Validation", "3,427", "3,416", "707,112"],
            ["Test", "6,855", "6,844", "1,416,708"],
        ],
    )
    sub("B. Sample boundaries and unavailable observations")
    para(
        "The timestamp grid is partitioned chronologically into 70% training, 10% validation and 20% test observations. For origin t, history spans t-11 through t; target offsets are 1, 3, 6 and 12 five-minute steps. All four targets must belong to the same partition. Earlier observed history can come from a preceding partition because it is already available at the prediction origin. No time shuffling or model-specific split is permitted."
    )
    para(
        "Input zeros and other invalid readings are masked and forward filled. Leading gaps use the corresponding sensor's training median. Training medians, historical averages and reference speeds are fitted on the training interval only. No backfill or future interpolation is used, and target values remain raw. Long outages may therefore produce stale repeated inputs; two missingness features preserve evidence of this condition. The test valid-target counts are 1,245,052, 1,245,040, 1,245,022 and 1,244,986 at the four horizons."
    )
    heading("IV. FEATURES MODELS AND COMMON METRICS")
    sub("A. Shared features and model configurations")
    para(
        "The 32 shared features comprise 12 current/past speed lags; eight causal rolling mean, standard-deviation, minimum and maximum statistics; three speed-change terms; seven cyclic/calendar/peak variables; and two missingness indicators. Every rolling window ends at the forecast origin. RF ablations use 12 lags (A), 19 lags plus temporal features (B), 27 lags plus temporal and rolling features (C), and all 32 features (D)."
    )
    para(
        "Persistence predicts the forward-filled current speed at every horizon. Historical average predicts the sensor's training-only mean for the target weekday and five-minute slot, excluding unavailable observations and using its training mean when a lookup cell is absent. RF uses 100 trees, maximum depth 15, bootstrap max_samples 0.1 and seed 42. The same settings are held fixed for all 16 horizon/feature-group fits."
    )
    para(
        "Each XGBoost model predicts the residual from current speed with an absolute-error objective. The preserved settings use histogram trees, depth 8, learning rate 0.1, minimum child weight 50, row and column subsampling 0.8, and seed 42. Training used validation early stopping. Completion re-evaluates the eight saved boosters rather than changing their fitted parameters."
    )
    para(
        "The history-augmented booster adds expected speed at target and origin times, their difference, and the current deviation from expected speed. Training lookups exclude the row's own relevant reading. Its 36 local features provide additional information; RF 32 versus XGBoost 32 remains the direct learner comparison."
    )
    sub("B. Aligned evaluation and error definitions")
    para(
        "The evaluation first verifies unique origin/sensor keys, exact row order, target arrays, target-validity flags and stored-clock temporal flags across prediction files and the shared dataset. Every model must supply finite predictions. Invalid targets are excluded identically; a failed prediction on a valid target raises an error rather than silently reducing the scored sample. An empty slice is represented as unscored, not zero error."
    )
    para(
        "For N finite positive observed targets, MAE and RMSE use the usual absolute and squared differences. MAPE is 100 times the average absolute error divided by max(y, 0.00001). The explicit denominator floor retains positive near-zero targets in all three metrics. Scores are computed from saved prediction values, with counts reported for each horizon and group."
    )
    blocks.append(("equation", "MAE = (1/N) sum |y - yhat|"))
    blocks.append(("equation", "RMSE = sqrt[(1/N) sum (y - yhat)^2]"))
    blocks.append(("equation", "MAPE = (100/N) sum |y - yhat| / max(y, 10^-5)"))
    heading("V. HELD OUT FORECASTING RESULTS")
    for metric, roman, label in [
        ("mae", "II", "TEST MAE"),
        ("rmse", "III", "TEST RMSE"),
        ("mape", "IV", "TEST MAPE PERCENT"),
    ]:
        rows = []
        for model in ORDER:
            rows.append(
                [LABELS[model]]
                + [
                    (
                        f"{score(model,h,metric):.3f}"
                        if metric != "mape"
                        else f"{score(model,h,metric):.2f}"
                    )
                    for h in HORIZONS
                ]
            )
        table(
            f"TABLE {roman}\n{label}",
            ["Model", "5 min", "15 min", "30 min", "60 min"],
            rows,
        )
    para(
        f'XGBoost plus HA achieves the lowest MAE among the evaluated configurations at every horizon. Its MAE reduction relative to persistence grows from {gains["5min"]:.1f}% at 5 minutes to {gains["60min"]:.1f}% at 60 minutes. Historical-average MAE is approximately horizon independent because it queries a recurring target-time profile; small differences arise from the eligible target rows.'
    )
    para(
        f'The local information restriction matters for interpretation. At 60 minutes, plain XGBoost has MAE {score("xgboost_32","60min"):.3f}, compared with {score("historical_average","60min"):.3f} for historical average. Adding the historical features yields {score("xgboost_ha","60min"):.3f}. This demonstrates a useful feature extension on this split; it does not isolate a graph effect or show that a local model family has reached a universal limit.'
    )
    para(
        "MAE and RMSE can rank the learners differently. Plain XGBoost optimizes absolute error, whereas RF uses a squared-error splitting criterion. RF has lower RMSE than plain XGBoost at the longer horizons even when its MAE is higher. This is an observed tradeoff between the fitted configurations and objectives, so all three metrics are retained instead of selecting only the metric favorable to one model."
    )
    figure(
        "horizon_mae.png",
        "Fig. 1. Test MAE across forecast horizons. The history-augmented booster uses 36 features; RF and plain XGBoost share 32.",
    )
    sub("A. Temporal slices and feature ablations")
    para(
        "Peak groups are weekday origin hours [07:00,10:00) and [16:00,19:00) on the stored clock. Their union is combined peak; off peak includes all other origins and weekends. Targets may fall after an origin-period boundary."
    )
    para(
        f'For 15-minute XGBoost plus HA forecasts, off-peak, morning and evening MAEs are {score("xgboost_ha","15min",period="off_peak"):.3f}, {score("xgboost_ha","15min",period="morning_peak"):.3f} and {score("xgboost_ha","15min",period="evening_peak"):.3f}. Evening-origin MAE is {peakpenalty:.1f}% higher than off-peak MAE. RF uses the same slices and target masks. These stored-clock differences are descriptive, with no event-level causal attribution.'
    )
    figure(
        "temporal_mae.png",
        "Fig. 2. The 15-minute temporal comparison includes all five configurations and identical valid targets within each origin-period slice.",
    )
    para(
        "The RF ablations isolate the value of the declared feature groups under identical fixed fitting settings. Temporal and rolling features are particularly useful relative to the lag-only configuration at longer horizons. Adding every remaining feature need not improve every score: the 27-feature group can slightly outperform the full 32-feature group. Complete ablation metrics, fit times and feature definitions accompany the machine-readable evidence."
    )
    heading("VI. SPATIAL ERRORS AND PAIRED UNCERTAINTY")
    sub("A. Observed and predicted congestion diagnostics")
    para(
        "Spatial evaluation joins verified sensor coordinates to per-sensor forecast errors on valid held-out targets. It compares observed and predicted means, MAE, RMSE, MAPE and observation counts, rather than using observed speed variability as a forecast-error surrogate. Morning and evening maps are generated separately for RF and XGBoost plus HA. Geographic aggregates sum errors and counts before division, avoiding unweighted means of sensor scores with unequal data availability."
    )
    para(
        "Coordinates are verified, but named freeways and traffic direction are not. Regions are labelled coordinate bins. Each sensor's training 85th percentile provides an operational speed reference. Descriptive peak summaries slice raw observation times and use valid-count weights; forecast maps slice origins and score future targets."
    )
    para(
        "For a reproducible congestion-frequency diagnostic, observed and predicted speeds below 60% of the sensor reference are counted on identical valid targets. Thresholds are fitted on training observations only. This proxy enables predicted-versus-observed congestion maps without inventing incident or queue labels. Its calibration against verified congestion events remains outside the available evidence."
    )
    figure(
        "spatial_errors.png",
        "Fig. 3. XGBoost plus HA 15-minute evening-origin predictions: observed mean speed and actual forecast MAE by sensor. Both panels use the same valid test targets and source speed units.",
    )
    sub("B. Paired blocks of stored calendar days")
    para(
        "Forecast errors are correlated across nearby times and sensors. We therefore avoid treating approximately 1.245 million target rows as independent samples for an uncertainty calculation. Instead, all observed targets and sensors within each stored calendar day are retained together. We draw 2,000 paired circular moving-block bootstrap samples, each assembled from blocks of two consecutive days, using seed 42."
    )
    para(
        "Each resampled absolute-error difference is divided by the common valid-target count. Positive values favor the candidate. The 2.5% and 97.5% percentiles form conditional intervals for this test period; they do not adjust for model selection or establish validity in other cities or seasons."
    )
    rows = []
    for _, r in ci[ci.horizon == "60min"].iterrows():
        rows.append(
            [
                LABELS[r.baseline] + " vs " + LABELS[r.candidate],
                f"{r.mae_difference:.3f}",
                f"{r.ci95_low:.3f} to {r.ci95_high:.3f}",
            ]
        )
    table(
        "TABLE V\nPAIRED 60 MINUTE MAE REDUCTION",
        ["Baseline vs candidate", "Reduction", "95% interval"],
        rows,
    )
    para(
        "A retrospective time-series example is selected from the highest-MAE sensor for the 60-minute history-augmented booster and centered on its largest valid absolute error. This diagnostic selection is disclosed and is never used for model tuning. The resulting traces illustrate local forecast deviations; they do not identify a verified incident, infer a queue propagation speed or establish causality."
    )
    figure(
        "forecast_example.png",
        f'Fig. 4. Retrospective 60-minute forecast example for sensor {result["example"]["sensor_id"]}. Lines are positioned at target time. The example is selected for error inspection after evaluation.',
    )
    heading("VII. REPRODUCIBILITY AND LIMITATIONS")
    para(
        "The repository provides features, predictions, model artifacts, common scoring and executed notebooks. SHA-256 manifests record raw data, model, dataset and forecast inputs with runtime versions. Per-sensor error sums reconstruct pooled MAE, and valid-target counts are checked against the raw-derived dataset."
    )
    para(
        "RF is regenerated with fixed settings; frozen boosters are re-scored with official XGBoost 3.4.1. An optional numeric-JSON reader preserves tree order and best iteration. It matches both a direct scalar walk and the official runtime exactly on 96 deterministic validation rows for each of the eight boosters."
    )
    para(
        "The lower valid-target fraction in test limits coverage of outage conditions, and forward-filled inputs can remain stale for hours. Evidence covers one chronological split and one seed per configuration. Unverified units, clock, road identity and proxy congestion labels constrain physical and causal interpretation."
    )
    para(
        "A future graph comparison should preserve the splits and masks, include strong temporal comparators and repeated seeds, and test the value of neighboring observations. No graph model or traffic-signal control is evaluated here."
    )
    heading("VIII. CONCLUSION")
    para(
        f'The benchmark integrates local forecasts, all-model temporal slices, sensor errors, congestion maps and paired uncertainty. The history-augmented XGBoost reaches {score("xgboost_ha","60min"):.3f} MAE at 60 minutes, while the shared-feature RF/XGBoost comparison preserves a clear distinction between learner and feature effects. The result is a traceable baseline for subsequent experiments rather than a claim of demonstrated graph superiority.'
    )
    heading("REFERENCES")
    for i, reference in enumerate(REFERENCES, 1):
        blocks.append(("reference", f"[{i}] " + reference))
    return abstract, blocks


def pdf_output(abstract, blocks, authors):
    fontroot = Path("/System/Library/Fonts/Supplemental")
    candidates = {
        "TNR": "Times New Roman.ttf",
        "TNR-Bold": "Times New Roman Bold.ttf",
        "TNR-Italic": "Times New Roman Italic.ttf",
        "TNR-BoldItalic": "Times New Roman Bold Italic.ttf",
    }
    for name, file in candidates.items():
        path = fontroot / file
        if not path.exists():
            raise FileNotFoundError(f"Embedded Times font required: {path}")
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
        fontSize=10,
        leading=11.7,
        alignment=TA_JUSTIFY,
        firstLineIndent=9,
        spaceAfter=4.5,
        allowWidows=0,
        allowOrphans=0,
    )
    styles = {
        "paragraph": body,
        "heading": ParagraphStyle(
            "Heading",
            parent=body,
            fontSize=10,
            leading=12,
            alignment=TA_CENTER,
            firstLineIndent=0,
            spaceBefore=8,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "subheading": ParagraphStyle(
            "Sub",
            parent=body,
            fontName="TNR-Italic",
            firstLineIndent=0,
            spaceBefore=5,
            spaceAfter=3,
            keepWithNext=True,
        ),
        "equation": ParagraphStyle(
            "Equation",
            parent=body,
            fontSize=9,
            leading=11,
            alignment=TA_CENTER,
            firstLineIndent=0,
            spaceAfter=4,
        ),
        "reference": ParagraphStyle(
            "Reference",
            parent=body,
            fontSize=8,
            leading=9.3,
            firstLineIndent=-12,
            leftIndent=12,
            spaceAfter=3,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=body,
            fontSize=8,
            leading=9.4,
            firstLineIndent=0,
            spaceAfter=6,
        ),
        "table_caption": ParagraphStyle(
            "TableCaption",
            parent=body,
            fontSize=8,
            leading=9.5,
            alignment=TA_CENTER,
            firstLineIndent=0,
            spaceBefore=5,
            spaceAfter=4,
        ),
        "cell": ParagraphStyle(
            "Cell",
            parent=body,
            fontSize=8,
            leading=9.2,
            alignment=TA_LEFT,
            firstLineIndent=0,
            spaceAfter=0,
        ),
        "cellnum": ParagraphStyle(
            "CellNum",
            parent=body,
            fontSize=8,
            leading=9.2,
            alignment=TA_CENTER,
            firstLineIndent=0,
            spaceAfter=0,
        ),
    }
    names = ", ".join(authors.get("names", [])) or authors.get(
        "fallback_author", "Phase 1 Project Team"
    )
    affiliation = authors.get("affiliation", "")
    author_names = authors.get("names", [])
    author_lines = (
        [", ".join(author_names[i : i + 3]) for i in range(0, len(author_names), 3)]
        if author_names
        else [names]
    )

    def page(canvas, doc):
        canvas.saveState()
        if doc.page == 1:
            canvas.setFont("TNR", 24)
            canvas.drawCentredString(306, 730, TITLE)
            canvas.drawCentredString(306, 703, SUBTITLE)
            canvas.setFont("TNR", 11)
            for index, author_line in enumerate(author_lines):
                canvas.drawCentredString(306, 674 - 14 * index, author_line)
            if affiliation:
                canvas.setFont("TNR-Italic", 10)
                canvas.drawCentredString(306, 674 - 14 * len(author_lines), affiliation)
        canvas.setFont("TNR", 8)
        canvas.drawCentredString(306, 38, str(doc.page))
        canvas.restoreState()

    doc = BaseDocTemplate(
        str(OUT / "phase1_ieee_paper.pdf"),
        pagesize=(612, 792),
        leftMargin=45,
        rightMargin=45,
        topMargin=54,
        bottomMargin=72,
        title=TITLE + " " + SUBTITLE,
        author=names,
        subject="Phase 1 METR-LA local forecasting benchmark",
        creator="Verified Phase 1 report generator",
        initialFontName="TNR",
    )

    def frames(top):
        return [
            Frame(
                45,
                72,
                252,
                top - 72,
                leftPadding=0,
                rightPadding=0,
                topPadding=0,
                bottomPadding=0,
                id="left",
            ),
            Frame(
                315,
                72,
                252,
                top - 72,
                leftPadding=0,
                rightPadding=0,
                topPadding=0,
                bottomPadding=0,
                id="right",
            ),
        ]

    doc.addPageTemplates(
        [
            PageTemplate(
                id="First", frames=frames(640), onPage=page, autoNextPageTemplate="Body"
            ),
            PageTemplate(id="Body", frames=frames(738), onPage=page),
        ]
    )
    story = [
        Paragraph(
            "<b><i>Abstract - </i>" + html.escape(abstract) + "</b>",
            ParagraphStyle(
                "Abstract",
                parent=body,
                fontSize=9,
                leading=10.5,
                firstLineIndent=0,
                spaceAfter=7,
            ),
        ),
        Paragraph(
            "<b><i>Index Terms - </i>Traffic forecasting, METR-LA, Random Forest, XGBoost, chronological evaluation, spatial errors.</b>",
            ParagraphStyle(
                "Terms",
                parent=body,
                fontSize=9,
                leading=10.5,
                firstLineIndent=0,
                spaceAfter=5,
            ),
        ),
    ]
    equation_number = 0
    for kind, value in blocks:
        if kind == "equation":
            equation_number += 1
            story.append(PaperEquation(value.split(" = ")[0], equation_number))
        elif kind in styles:
            if kind == "heading" and value.startswith("VII."):
                story.append(FrameBreak())
            story.append(Paragraph(html.escape(value), styles[kind]))
        elif kind == "table":
            headers = value["headers"]
            rows = value["rows"]
            widths = (
                [84, 42, 42, 42, 42]
                if len(headers) == 5
                else [63] * 4 if len(headers) == 4 else [128, 42, 82]
            )
            content = []
            for i, row in enumerate([headers] + rows):
                content.append(
                    [
                        Paragraph(
                            (
                                ("<b>" + html.escape(str(v)) + "</b>")
                                if i == 0
                                else html.escape(str(v))
                            ),
                            styles["cell"] if j == 0 else styles["cellnum"],
                        )
                        for j, v in enumerate(row)
                    ]
                )
            table = Table(content, colWidths=widths, repeatRows=1, hAlign="CENTER")
            table.setStyle(
                TableStyle(
                    [
                        ("FONTNAME", (0, 0), (-1, -1), "TNR"),
                        ("LINEABOVE", (0, 0), (-1, 0), 0.5, colors.black),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.black),
                        ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.black),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 2),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ]
                )
            )
            caption = Paragraph(
                html.escape(value["caption"]).replace("\n", "<br/>"),
                styles["table_caption"],
            )
            story.append(KeepTogether([caption, table, Spacer(1, 5)]))
        elif kind == "figure":
            path = OUT / "figures" / value["file"]
            from PIL import Image as PILImage

            with PILImage.open(path) as image:
                w, h = image.size
            story.append(
                KeepTogether(
                    [
                        Spacer(1, 4),
                        Image(str(path), width=252, height=252 * h / w),
                        Spacer(1, 3),
                        Paragraph(html.escape(value["caption"]), styles["caption"]),
                    ]
                )
            )
    doc.build(story, canvasmaker=partial(Canvas, initialFontName="TNR"))


def latex_output(abstract, blocks, authors):
    def escape(text):
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
        return "".join(mapping.get(c, c) for c in str(text))

    names = ", ".join(authors.get("names", [])) or authors.get(
        "fallback_author", "Phase 1 Project Team"
    )
    author_names = authors.get("names", [])
    author_lines = (
        [", ".join(author_names[i : i + 3]) for i in range(0, len(author_names), 3)]
        if author_names
        else [names]
    )
    author = r"\IEEEauthorblockN{" + r"\\".join(map(escape, author_lines)) + "}"
    if authors.get("affiliation"):
        author += r"\IEEEauthorblockA{" + escape(authors["affiliation"]) + "}"
    lines = [
        r"\documentclass[conference]{IEEEtran}",
        r"\usepackage{graphicx,booktabs,amsmath}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\title{" + escape(TITLE) + r"\\" + escape(SUBTITLE) + "}",
        r"\author{" + author + "}",
        r"\begin{document}",
        r"\maketitle",
        r"\begin{abstract}",
        escape(abstract),
        r"\end{abstract}",
        r"\begin{IEEEkeywords}Traffic forecasting, METR-LA, Random Forest, XGBoost, chronological evaluation, spatial errors.\end{IEEEkeywords}",
    ]
    refs = False
    for kind, value in blocks:
        if kind == "heading":
            if value == "REFERENCES":
                refs = True
                lines.append(r"\begin{thebibliography}{00}")
                continue
            lines.append(
                r"\section{" + escape(re.sub(r"^[IVX]+\. ", "", value).title()) + "}"
            )
        elif kind == "subheading":
            lines.append(
                r"\subsection{" + escape(re.sub(r"^[A-Z]\. ", "", value)) + "}"
            )
        elif kind == "paragraph":
            lines.extend([escape(value), ""])
        elif kind == "equation":
            eq = {
                "MAE": r"\mathrm{MAE}=\frac{1}{N}\sum_i |y_i-\hat y_i|",
                "RMSE": r"\mathrm{RMSE}=\sqrt{\frac{1}{N}\sum_i (y_i-\hat y_i)^2}",
                "MAPE": r"\mathrm{MAPE}=\frac{100}{N}\sum_i\frac{|y_i-\hat y_i|}{\max(y_i,10^{-5})}",
            }
            lines.append(
                r"\begin{equation}" + eq[value.split(" = ")[0]] + r"\end{equation}"
            )
        elif kind == "table":
            caption = value["caption"].split("\n", 1)[-1].capitalize()
            lines.extend(
                [
                    r"\begin{table}[t]",
                    r"\caption{" + escape(caption) + "}",
                    r"\centering\footnotesize",
                    r"\begin{tabular}{l" + "r" * (len(value["headers"]) - 1) + "}",
                    r"\toprule",
                    " & ".join(escape(x) for x in value["headers"]) + r"\\",
                    r"\midrule",
                ]
            )
            lines += [
                " & ".join(escape(x) for x in row) + r"\\" for row in value["rows"]
            ]
            lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        elif kind == "figure":
            lines.extend(
                [
                    r"\begin{figure}[t]",
                    r"\centering",
                    r"\includegraphics[width=\columnwidth]{figures/"
                    + value["file"]
                    + "}",
                    r"\caption{"
                    + escape(re.sub(r"^Fig\. \d+\. ", "", value["caption"]))
                    + "}",
                    r"\end{figure}",
                ]
            )
        elif kind == "reference":
            n = re.match(r"\[(\d+)\]", value).group(1)
            lines.append(
                r"\bibitem{ref" + n + "} " + escape(re.sub(r"^\[\d+\] ", "", value))
            )
    if refs:
        lines.append(r"\end{thebibliography}")
    lines.append(r"\end{document}")
    (OUT / "phase1_ieee_paper.tex").write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = json.loads((ROOT / "data/metadata/person_e_results.json").read_text())
    audit = json.loads((ROOT / "data/metadata/phase1_verification.json").read_text())
    if audit["status"] != "passed":
        raise ValueError("Paper requires a passed Phase 1 acceptance audit")
    m = pd.read_csv(ROOT / "data/metadata/phase1_metrics.csv")
    s = pd.read_csv(
        ROOT / "data/metadata/sensor_forecast_metrics.csv", dtype={"sensor_id": str}
    )
    ci = pd.read_csv(ROOT / "data/metadata/paired_bootstrap.csv")
    authors = json.loads((OUT / "authors.json").read_text())
    figures(m, s, result)
    abstract, blocks = build_content(m, s, ci, result, audit)
    pdf_output(abstract, blocks, authors)
    latex_output(abstract, blocks, authors)
    (OUT / "paper_content.json").write_text(
        json.dumps(
            {
                "title": TITLE,
                "subtitle": SUBTITLE,
                "authors": authors,
                "abstract": abstract,
                "blocks": blocks,
            },
            indent=2,
        )
        + "\n"
    )
    author_note = "Authors: " + ", ".join(authors.get("names", [])) + "."
    if not authors.get("names"):
        author_note = (
            "Author: " + authors.get("fallback_author", "Phase 1 Project Team") + "."
        )
    author_note += (
        " Affiliation: " + authors["affiliation"] + "."
        if authors.get("affiliation")
        else " Affiliation has not been supplied and is left blank."
    )
    (OUT / "README.md").write_text(
        """# Phase 1 IEEE style paper\n\n`phase1_ieee_paper.pdf` is a two-column US Letter paper with embedded Times New Roman fonts, numbered references and generated tables/figures. `phase1_ieee_paper.tex` contains the matching editable content using the IEEEtran conference class. It can be compiled in a TeX environment with IEEEtran installed; the supplied PDF is generated deterministically by ReportLab. This is an IEEE-style capstone manuscript, not an accepted/published IEEE article. No conference name, DOI or copyright attribution is invented.\n\n"""
        + author_note
        + " Update `authors.json` and rerun `python scripts/build_phase1_paper.py` to revise the author block.\n\nTables and figures derive from verified Phase 1 forecast rows. Complete numerical evidence is in `data/metadata/`. `paper_content.json` is an editable structured copy of the narrative, tables and figure captions.\n"
    )
    print("Built", OUT / "phase1_ieee_paper.pdf")
    print("Editable IEEEtran source:", OUT / "phase1_ieee_paper.tex")


if __name__ == "__main__":
    main()
