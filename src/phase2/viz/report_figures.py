"""Phase 2 report figures (Member 4), drawn only from evaluation outputs.

Every figure reads the tables written by ``scripts/phase2_eval_all.py``; nothing is
typed in. Colours follow the entity (a model keeps its colour in every figure).
Labels say "source speed units", "stored clock", "proxy congestion" and
"coordinate" positions; no freeway names.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd

from src.phase2.common import ROOT

HORIZONS = ["5min", "15min", "30min", "60min"]
H_MIN = [5, 15, 30, 60]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#8a8985"
# Fixed entity -> colour (categorical slots in fixed order; baselines neutral).
COLOR = {
    "xgboost_ha": "#2a78d6",
    "gru_local": "#eb6834",
    "gat": "#1baf7a",
    "dcrnn": "#6250d6",
    "gat_identity": "#e87ba4",
    "gat_shuffled": "#e34948",
    "dcrnn_identity": "#eda100",
    "persistence": NEUTRAL,
    "random_forest_32": "#b5b4ae",
    "xgboost_32": "#86b6ef",
    "historical_average": "#d4d3cd",
}
LABEL = {
    "persistence": "Persistence", "historical_average": "Historical avg.",
    "random_forest_32": "Random Forest 32", "xgboost_32": "XGBoost 32",
    "xgboost_ha": "XGBoost + HA 36", "gru_local": "GRU Local", "gat": "GAT",
    "dcrnn": "DCRNN", "gat_identity": "GAT identity", "gat_shuffled": "GAT shuffled",
    "dcrnn_identity": "DCRNN identity",
}
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
SEQ_ORANGE = LinearSegmentedColormap.from_list("seq_orange", ["#fde3d6", "#f6a882", "#eb6834", "#b84a1d", "#7a2e10"])
DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a99", "#b8302f"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
})


def _save(fig, out_dir: Path, name: str, files: List[Path]) -> None:
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    files.append(path)


def _test_all(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[(summary["split"] == "test") & (summary["period"] == "all")]


def best_models(summary: pd.DataFrame) -> Dict[str, str]:
    """Best Phase 1 and best Phase 2 model, chosen on VALIDATION mean MAE (no test peeking)."""
    v = summary[(summary["split"] == "val") & (summary["period"] == "all")]
    mean = v.groupby("model")["mae_mean"].mean()
    p1 = [m for m in ("persistence", "historical_average", "random_forest_32", "xgboost_32", "xgboost_ha") if m in mean]
    p2 = [m for m in ("gru_local", "gat", "dcrnn") if m in mean]
    out = {"phase1": mean[p1].idxmin()}
    if p2:
        out["phase2"] = mean[p2].idxmin()
    return out


def fig_horizon_mae(summary, out_dir, files):
    t = _test_all(summary)
    models = [m for m in ("persistence", "xgboost_ha", "gru_local", "gat", "dcrnn") if m in set(t["model"])]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ends = []
    for m in models:
        s = t[t["model"] == m].set_index("horizon").reindex(HORIZONS)
        ls = ":" if m == "persistence" else "-"
        ax.errorbar(H_MIN, s["mae_mean"], yerr=s["mae_sd"].fillna(0), color=COLOR[m], lw=2, ls=ls,
                    marker="o", ms=5, capsize=3, label=LABEL[m])
        ends.append([float(s["mae_mean"].iloc[-1]), m])
    # direct labels at the line ends, nudged apart so they never overlap
    ends.sort()
    gap = 0.045 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + gap)
    for y, m in ends:
        ax.annotate(f"{LABEL[m]} {t[(t['model'] == m) & (t['horizon'] == '60min')]['mae_mean'].iloc[0]:.3f}",
                    (61.5, y), va="center", fontsize=7.5, color=INK2, annotation_clip=False)
    ax.set_xticks(H_MIN)
    ax.set_xlim(2, 62)
    ax.set_xlabel("Forecast horizon (min)")
    ax.set_ylabel("Test MAE (source speed units)")
    ax.set_title("Test MAE by horizon (Phase 2: mean ± sd over seeds)")
    ax.legend(loc="upper left", fontsize=8)
    _save(fig, out_dir, "fig_horizon_mae", files)


def fig_peak_offpeak(summary, out_dir, files):
    t = summary[(summary["split"] == "test") & (summary["horizon"] == "60min")]
    models = [m for m in ("persistence", "xgboost_ha", "gru_local", "gat_identity", "gat", "dcrnn") if m in set(t["model"])]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    x = np.arange(len(models))
    w = 0.38
    for i, (period, color, lab) in enumerate([("peak", "#2a78d6", "Peak (weekday 7-10, 16-19)"),
                                              ("off_peak", "#eb6834", "Off-peak")]):
        vals = [t[(t["model"] == m) & (t["period"] == period)]["mae_mean"].iloc[0] for m in models]
        bars = ax.bar(x + (i - 0.5) * (w + 0.02), vals, w, color=color, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center", fontsize=6.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[m] for m in models], rotation=20, ha="right")
    ax.set_ylabel("Test MAE at 60 min (source speed units)")
    ax.set_title("Peak vs off-peak error, 60-minute horizon (stored clock)")
    ax.legend(fontsize=8, loc="upper right")
    _save(fig, out_dir, "fig_peak_offpeak", files)


def fig_seed_spread(metrics, out_dir, files):
    t = metrics[(metrics["split"] == "test") & (metrics["period"] == "all") & (metrics["seed"].astype(str) != "")]
    models = [m for m in ("gru_local", "gat_identity", "gat", "gat_shuffled", "dcrnn", "dcrnn_identity") if m in set(t["model"])]
    if not models:
        return
    fig, axes = plt.subplots(2, 2, figsize=(7, 0.45 * len(models) * 2 + 2.2), sharey=True)
    for ax, h in zip(axes.flat, HORIZONS):
        for i, m in enumerate(models):
            v = t[(t["model"] == m) & (t["horizon"] == h)]["mae"].to_numpy(float)
            ax.scatter(v, np.full(len(v), i), color=COLOR[m], s=36, edgecolor=SURFACE, linewidth=1, zorder=3)
            ax.plot([v.min(), v.max()], [i, i], color=COLOR[m], lw=1.2)
        ax.set_title(h.replace("min", " min"))
        ax.set_xlabel("Test MAE")
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.grid(axis="y", visible=False)
    for ax in axes[:, 0]:
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([LABEL[m] for m in models])
    fig.suptitle("Spread across training seeds (one dot per seed)", fontsize=10, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir, "fig_seed_spread", files)


def fig_graph_ablation(boot, out_dir, files):
    comps = [("gru_local", "gat"), ("gat_identity", "gat"), ("gat_shuffled", "gat"), ("gat_identity", "gat_shuffled"),
             ("gru_local", "dcrnn"), ("dcrnn_identity", "dcrnn"), ("xgboost_ha", "gat"), ("xgboost_ha", "dcrnn")]
    b = boot[boot["period"] == "all"]
    comps = [c for c in comps if ((b["baseline"] == c[0]) & (b["candidate"] == c[1])).any()]
    if not comps:
        return
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 0.36 * len(comps) * 2 + 2.4), sharey=True)
    for ax, h in zip(axes.flat, HORIZONS):
        for i, (base, cand) in enumerate(comps):
            r = b[(b["baseline"] == base) & (b["candidate"] == cand) & (b["horizon"] == h)].iloc[0]
            col = COLOR[cand]
            ax.plot([r["ci95_low"], r["ci95_high"]], [i, i], color=col, lw=2.2, solid_capstyle="round")
            ax.scatter([r["mae_difference"]], [i], color=col, s=30, zorder=3, edgecolor=SURFACE, linewidth=1)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(h.replace("min", " min"))
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("MAE reduction (95% CI)")
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    for ax in axes[:, 0]:
        ax.set_yticks(range(len(comps)))
        ax.set_yticklabels([f"{LABEL[c]} vs {LABEL[bb]}" for bb, c in comps], fontsize=8)
    axes[0, 0].invert_yaxis()
    fig.suptitle("Paired day-block bootstrap: positive = first model has lower test MAE", fontsize=10, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir, "fig_graph_ablation", files)


def _scatter_map(ax, df, values, cmap, norm=None, vmin=None, vmax=None, title=""):
    sc = ax.scatter(df["longitude"], df["latitude"], c=values, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
                    s=22, edgecolor=SURFACE, linewidth=0.5)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_aspect(1.2, adjustable="datalim")  # ~1/cos(34 deg) so coordinates are not stretched
    for s in ax.spines.values():
        s.set_visible(False)
    return sc


def fig_maps(sens, best, out_dir, files):
    if "phase2" not in best:
        return
    p1, p2 = best["phase1"], best["phase2"]
    for period, plab in (("morning_peak", "morning"), ("evening_peak", "evening")):
        for h in ("15min", "60min"):
            s = sens[(sens["split"] == "test") & (sens["period"] == period) & (sens["horizon"] == h)]
            a = s[s["model"] == p1].set_index("sensor_id")
            b = s[s["model"] == p2].set_index("sensor_id")
            idx = a.index.intersection(b.index)
            a, b = a.loc[idx], b.loc[idx]
            fig, axes = plt.subplots(2, 2, figsize=(10, 6.2))
            axes = axes.flatten()
            sc0 = _scatter_map(axes[0], a, a["actual_sum"] / a["n"], SEQ.reversed(), title="Observed mean speed")
            fig.colorbar(sc0, ax=axes[0], shrink=0.75, label="source speed units")
            vmax = float(np.nanpercentile(np.r_[a["mae"], b["mae"]], 98))
            sc1 = _scatter_map(axes[1], a, a["mae"], SEQ, vmin=0, vmax=vmax, title=f"MAE: {LABEL[p1]}")
            sc2 = _scatter_map(axes[2], b, b["mae"], SEQ, vmin=0, vmax=vmax, title=f"MAE: {LABEL[p2]}")
            fig.colorbar(sc1, ax=axes[1], shrink=0.75, label="MAE (shared scale)")
            fig.colorbar(sc2, ax=axes[2], shrink=0.75, label="MAE (shared scale)")
            d = (b["mae"] - a["mae"]).to_numpy()
            lim = float(np.nanpercentile(np.abs(d), 98)) or 1.0
            sc3 = _scatter_map(axes[3], a, d, DIV, norm=TwoSlopeNorm(0, -lim, lim),
                               title=f"{LABEL[p2]} minus {LABEL[p1]}")
            fig.colorbar(sc3, ax=axes[3], shrink=0.75, label="MAE difference (blue = Phase 2 lower)")
            fig.suptitle(f"Test {plab} peak, {h.replace('min', '-min')} horizon (sensor coordinates; stored clock)",
                         fontsize=10, fontweight="bold")
            _save(fig, out_dir, f"fig_map_{plab}_{h}", files)


def fig_congestion_maps(sens, best, out_dir, files):
    if "phase2" not in best or "actual_congested_count" not in sens:
        return
    p2 = best["phase2"]
    for period, plab in (("morning_peak", "morning"), ("evening_peak", "evening")):
        s = sens[(sens["split"] == "test") & (sens["period"] == period) & (sens["horizon"] == "15min")]
        b = s[s["model"] == p2].set_index("sensor_id")
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
        obs = b["actual_congested_count"] / b["n"]
        pred = b["predicted_congested_count"] / b["n"]
        vmax = float(max(obs.max(), pred.max()))
        sc = _scatter_map(axes[0], b, obs, SEQ_ORANGE, vmin=0, vmax=vmax, title="Observed proxy congestion rate")
        _scatter_map(axes[1], b, pred, SEQ_ORANGE, vmin=0, vmax=vmax, title=f"Predicted by {LABEL[p2]} (15 min)")
        fig.colorbar(sc, ax=axes, shrink=0.75, label="share of origins congested (proxy)")
        fig.suptitle(f"Proxy congestion, test {plab} peak (speed < 60% of sensor's training 85th pct.)",
                     fontsize=10, fontweight="bold")
        _save(fig, out_dir, f"fig_congestion_map_{plab}_15min", files)


def fig_error_by_degree(sens, out_dir, files):
    t = sens[(sens["split"] == "test") & (sens["period"] == "all") & (sens["horizon"] == "60min")]
    if not {"gat", "gat_identity"} <= set(t["model"]):
        return
    g = t.pivot(index="sensor_id", columns="model", values="mae")
    deg = t.drop_duplicates("sensor_id").set_index("sensor_id")["graph_degree"]
    gain = (g["gat_identity"] - g["gat"]).rename("gain").to_frame().join(deg)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.scatter(gain["graph_degree"], gain["gain"], s=16, color=COLOR["gat"], alpha=0.75, edgecolor=SURFACE, linewidth=0.5)
    bins = pd.cut(gain["graph_degree"], [0, 4, 8, 12, 16, 100], right=False)
    m = gain.groupby(bins, observed=True).agg(d=("graph_degree", "mean"), g=("gain", "mean"))
    ax.plot(m["d"], m["g"], color=INK, lw=2, marker="s", ms=5, label="bin mean")
    ax.axhline(0, color=INK2, lw=0.8)
    rho = gain[["graph_degree", "gain"]].corr(method="spearman").iloc[0, 1]
    ax.set_xlabel("Number of graph neighbours (bidirectional, excl. self)")
    ax.set_ylabel("60-min MAE gain: GAT identity − GAT")
    ax.set_title(f"Graph gain per sensor vs degree (Spearman ρ = {rho:.2f}, descriptive)")
    ax.legend(fontsize=8)
    _save(fig, out_dir, "fig_error_by_degree", files)


def fig_case_study(case_path: Path, best, out_dir, files):
    if not case_path.is_file() or "phase2" not in best:
        return
    c = pd.read_csv(case_path, parse_dates=["origin_timestamp"])
    p1, p2 = best["phase1"], best["phase2"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, h, step in zip(axes, ("15min", "60min"), (3, 12)):
        tt = c["origin_timestamp"] + pd.Timedelta(minutes=5 * step)
        ax.plot(tt, c[f"target_{h}"].where(c[f"target_valid_{h}"]), color=INK, lw=2, label="Observed")
        ax.plot(tt, c[f"pred_{p1}_{h}"], color=COLOR[p1], lw=1.8, label=LABEL[p1])
        ax.plot(tt, c[f"pred_{p2}_{h}"], color=COLOR[p2], lw=1.8, label=f"{LABEL[p2]} (seed mean)")
        ax.set_title(f"{h.replace('min', '-min')} forecasts, plotted at target time")
        ax.tick_params(axis="x", rotation=30)
        ax.set_xlabel("Stored clock")
    axes[0].set_ylabel("Speed (source units)")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    day = c["origin_timestamp"].dt.date.iloc[0]
    fig.suptitle(f"Sensor {c['sensor_id'].iloc[0]}, {day}: observed speed vs forecasts (test period)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir, "fig_case_study", files)


def fig_attention_example(attention_dir: Path, out_dir, files):
    f = attention_dir / "gat_seed42.npz"
    if not f.is_file():
        return
    from src.phase2.common import load_graph_pickle

    with np.load(f) as z:
        attn = z["attn"].astype(np.float32)  # [12, layers, heads, N, N]
        origin_ns = z["origin_ns"]
    meta = pd.read_csv(ROOT / "data" / "metadata" / "sensors.csv", dtype={"sensor_id": str})
    ids, _, adj = load_graph_pickle(ROOT / "data" / "raw" / "adj_mx.pkl")
    a = np.asarray(adj) > 0
    np.fill_diagonal(a, False)
    deg = (a | a.T).sum(1)
    i = int(np.argmax(deg))
    times = pd.to_datetime(origin_ns)
    picks = [k for k, t in enumerate(times) if t.dayofweek == times[0].dayofweek][:3]
    fig, axes = plt.subplots(1, len(picks), figsize=(3.4 * len(picks), 3.3))
    axes = np.atleast_1d(axes)
    lat, lon = meta["latitude"].to_numpy(), meta["longitude"].to_numpy()
    for ax, k in zip(axes, picks):
        w = attn[k, -1].mean(axis=0)[i]  # last layer, head mean, row = target sensor i
        ax.scatter(lon, lat, s=6, color="#d4d3cd")
        nb = np.flatnonzero(w > 1e-4)
        for j in nb:
            if j != i:
                ax.plot([lon[i], lon[j]], [lat[i], lat[j]], color=COLOR["gat"], lw=0.5 + 6 * w[j], alpha=0.85)
        ax.scatter([lon[i]], [lat[i]], s=40, color=INK, zorder=3)
        ax.set_title(f"{times[k]:%a %H:%M}  (self weight {w[i]:.2f})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_aspect("equal", adjustable="datalim")
        span = 0.06
        ax.set_xlim(lon[i] - span, lon[i] + span); ax.set_ylim(lat[i] - span, lat[i] + span)
    fig.suptitle(f"GAT attention (last layer, head mean) from sensor {ids[i]} to its neighbours; "
                 "line width = weight. Shows what the model used, not traffic flow.", fontsize=9, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir, "fig_attention_example", files)


def render_report_figures(metrics: pd.DataFrame, summary: pd.DataFrame, boot: pd.DataFrame,
                          sens: pd.DataFrame, events_df: Optional[pd.DataFrame], run_meta: pd.DataFrame,
                          out_dir: Path, attention_dir: Optional[Path] = None,
                          case_path: Optional[Path] = None) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: List[Path] = []
    best = best_models(summary)
    fig_horizon_mae(summary, out_dir, files)
    fig_peak_offpeak(summary, out_dir, files)
    fig_seed_spread(metrics, out_dir, files)
    if not boot.empty:
        fig_graph_ablation(boot, out_dir, files)
    fig_maps(sens, best, out_dir, files)
    fig_congestion_maps(sens, best, out_dir, files)
    fig_error_by_degree(sens, out_dir, files)
    fig_case_study(case_path or (ROOT / "data" / "metadata" / "phase2" / "case_study_717468.csv"), best, out_dir, files)
    if attention_dir is not None:
        fig_attention_example(Path(attention_dir), out_dir, files)
    return files
