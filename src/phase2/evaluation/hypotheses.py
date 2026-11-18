"""Paired day-block bootstrap comparisons and H0-H5 verdicts (Member 4).

Contract: reports/phase2_plan.md sections 5.3 and 5.4.

Conventions
-----------
* ``daily`` rows: ``split, horizon, model, period, day, n, absolute_error_sum``. For a
  model with several seeds, ``absolute_error_sum`` is the mean over seeds, so
  sum / n is exactly the mean seed-wise MAE (plan 5.4). ``n`` does not depend on the
  model (it is the number of valid targets), which the pairing checks.
* A comparison ``(baseline B, candidate A)`` reports ``mae_difference = MAE_B - MAE_A``;
  positive means A has the lower MAE.
* Resampling is the Phase 1 circular moving-block bootstrap over stored days
  (2-day blocks, 2000 resamples, seed 42). ``block_indexes`` reproduces the exact index
  draws of ``src.models.prediction_evaluation.paired_daily_bootstrap``, so a one-term
  contrast equals the Phase 1 result. Contrasts between gains (H3, H4) reuse the same
  resampled days for every term.
* Verdict rule for "A beats B": supported if the whole 95% interval of the MAE
  reduction is above 0, contradicted if entirely below 0, inconclusive otherwise. A
  hypothesis that needs several horizons is supported only if every one is supported,
  contradicted only if every one is contradicted, inconclusive otherwise.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HORIZONS = ("5min", "15min", "30min", "60min")
REPETITIONS = 2000
SEED = 42
BLOCK_DAYS = 2

# Plan 5.4: comparisons as (baseline, candidate, periods).
GRAPH_COMPARISONS: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("gru_local", "gat", ("all", "peak", "off_peak")),
    ("gat_identity", "gat", ("all", "peak", "off_peak")),
    ("gat_shuffled", "gat", ("all",)),
    ("dcrnn", "gat", ("all",)),
    ("xgboost_ha", "gat", ("all",)),
    ("gru_local", "dcrnn", ("all",)),
    ("dcrnn_identity", "dcrnn", ("all",)),
    ("xgboost_ha", "dcrnn", ("all",)),
    # H2 second part: does the wrong graph beat no graph?
    ("gat_identity", "gat_shuffled", ("all",)),
    # Deep non-graph control vs the strongest Phase 1 model (context, not a graph effect)
    ("xgboost_ha", "gru_local", ("all",)),
]
# Same comparisons as Phase 1's paired_bootstrap.csv (harness parity check).
PHASE1_COMPARISONS: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("persistence", "xgboost_ha", ("all",)),
    ("random_forest_32", "xgboost_32", ("all",)),
    ("xgboost_32", "xgboost_ha", ("all",)),
]

BOOTSTRAP_COLUMNS = [
    "split", "baseline", "candidate", "horizon", "period", "days", "block_days",
    "repetitions", "nonempty_repetitions", "seed", "mae_difference", "ci95_low", "ci95_high",
]


def block_indexes(n_days: int, repetitions: int = REPETITIONS, seed: int = SEED,
                  block_days: int = BLOCK_DAYS) -> np.ndarray:
    """Day indexes of every bootstrap resample, identical to Phase 1's draws."""
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n_days, size=(repetitions, int(np.ceil(n_days / block_days))))
    idx = ((starts[:, :, None] + np.arange(block_days)) % n_days).reshape(repetitions, -1)
    return idx[:, :n_days]


def _pair(daily: pd.DataFrame, baseline: str, candidate: str, horizon: str,
          period: str) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
    q = daily[(daily["split"] == "test") & (daily["horizon"] == horizon) & (daily["period"] == period)]
    b = q[q["model"] == baseline].set_index("day").sort_index()
    c = q[q["model"] == candidate].set_index("day").sort_index()
    if b.empty or c.empty:
        return None
    if not b.index.equals(c.index) or not np.array_equal(b["n"].to_numpy(), c["n"].to_numpy()):
        raise ValueError(f"{baseline} vs {candidate} {horizon} {period}: days or valid counts differ")
    diff = (b["absolute_error_sum"] - c["absolute_error_sum"]).to_numpy(dtype=float)
    return diff, b["n"].to_numpy(dtype=float), list(b.index)


def bootstrap_contrast(
    daily: pd.DataFrame,
    terms: Sequence[Tuple[float, str, str, str, str]],
    repetitions: int = REPETITIONS,
    seed: int = SEED,
    block_days: int = BLOCK_DAYS,
) -> Optional[Dict[str, Any]]:
    """Bootstrap ``sum_k coef_k * gain_k`` with gain_k = MAE(baseline_k) - MAE(candidate_k).

    ``terms`` = [(coef, baseline, candidate, horizon, period), ...]. All terms are
    evaluated on the same resampled days. Returns None if any model is missing.
    """
    pairs = []
    days = None
    for coef, base, cand, h, p in terms:
        got = _pair(daily, base, cand, h, p)
        if got is None:
            return None
        diff, n, d = got
        if days is None:
            days = d
        elif d != days:
            raise ValueError("contrast terms must cover the same days")
        pairs.append((coef, diff, n))
    n_days = len(days)
    if n_days < 2:
        raise ValueError("at least two days are required")
    idx = block_indexes(n_days, repetitions, seed, block_days)
    observed = sum(coef * diff.sum() / n.sum() for coef, diff, n in pairs)
    draws = np.zeros(repetitions)
    keep = np.ones(repetitions, dtype=bool)
    for coef, diff, n in pairs:
        den = n[idx].sum(axis=1)
        ok = den > 0
        keep &= ok
        draws[ok] += coef * diff[idx][ok].sum(axis=1) / den[ok]
    draws = draws[keep]
    if len(draws) < min(100, repetitions):
        raise ValueError("too few nonempty bootstrap samples")
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "days": n_days, "block_days": block_days, "repetitions": repetitions,
        "nonempty_repetitions": int(len(draws)), "seed": seed,
        "mae_difference": float(observed), "ci95_low": float(lo), "ci95_high": float(hi),
    }


def run_comparisons(daily: pd.DataFrame,
                    comparisons: Iterable[Tuple[str, str, Tuple[str, ...]]] | None = None) -> pd.DataFrame:
    """All pairwise comparisons whose models are present; absent pairs are skipped."""
    comparisons = list(comparisons) if comparisons is not None else GRAPH_COMPARISONS + PHASE1_COMPARISONS
    rows = []
    for base, cand, periods in comparisons:
        for period in periods:
            for h in HORIZONS:
                res = bootstrap_contrast(daily, [(1.0, base, cand, h, period)])
                if res is not None:
                    rows.append({"split": "test", "baseline": base, "candidate": cand,
                                 "horizon": h, "period": period, **res})
    return pd.DataFrame(rows, columns=BOOTSTRAP_COLUMNS)


def verdict(lo: float, hi: float) -> str:
    if lo > 0:
        return "supported"
    if hi < 0:
        return "contradicted"
    return "inconclusive"


def combine(verdicts: Sequence[str]) -> str:
    if not verdicts or any(v == "not_evaluated" for v in verdicts):
        return "not_evaluated"
    if all(v == "supported" for v in verdicts):
        return "supported"
    if all(v == "contradicted" for v in verdicts):
        return "contradicted"
    return "inconclusive"


HYPOTHESIS_COLUMNS = [
    "id", "statement", "horizon", "period", "comparison", "mae_difference",
    "ci95_low", "ci95_high", "verdict", "overall_verdict", "note",
]


def evaluate_hypotheses(daily: pd.DataFrame, sensor_gain: Optional[pd.DataFrame] = None,
                        onset_gain: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """H0-H5 rows. Component rows carry their own verdict; ``overall_verdict`` is the
    hypothesis-level verdict under the plan's rule."""
    out: List[Dict[str, Any]] = []

    def add(hid, statement, horizon, period, comparison, res, comp_verdict, note=""):
        out.append({
            "id": hid, "statement": statement, "horizon": horizon, "period": period,
            "comparison": comparison,
            "mae_difference": np.nan if res is None else res["mae_difference"],
            "ci95_low": np.nan if res is None else res["ci95_low"],
            "ci95_high": np.nan if res is None else res["ci95_high"],
            "verdict": comp_verdict, "overall_verdict": None, "note": note,
        })

    def finish(hid, overall):
        for r in out:
            if r["id"] == hid:
                r["overall_verdict"] = overall

    def pairwise(hid, statement, base, cand, horizons, period="all"):
        vs = []
        for h in horizons:
            res = bootstrap_contrast(daily, [(1.0, base, cand, h, period)])
            v = "not_evaluated" if res is None else verdict(res["ci95_low"], res["ci95_high"])
            vs.append(v)
            add(hid, statement, h, period, f"{base} - {cand}", res, v)
        return vs

    # H1: gat beats gru_local at 30 and 60 min
    s1 = "gat has lower MAE than gru_local at 30 and 60 min"
    finish("H1", combine(pairwise("H1", s1, "gru_local", "gat", ("30min", "60min"))))

    # H2: gat beats gat_identity (every horizon); gat_shuffled no better than gat_identity
    s2 = "gat beats gat_identity, and gat_shuffled is no better than gat_identity"
    va = pairwise("H2", s2 + " [part a: gat vs gat_identity]", "gat_identity", "gat", HORIZONS)
    vb_rows = []
    for h in HORIZONS:
        res = bootstrap_contrast(daily, [(1.0, "gat_identity", "gat_shuffled", h, "all")])
        if res is None:
            v = "not_evaluated"
        else:
            # "no better": the shuffled graph must not be supported as beating identity
            v = "supported" if res["ci95_low"] <= 0 else "contradicted"
        vb_rows.append(v)
        add("H2", s2 + " [part b: gat_shuffled no better than gat_identity]", h, "all",
            "gat_identity - gat_shuffled", res, v,
            note="part b holds unless gat_shuffled's MAE reduction over gat_identity has its whole interval above 0")
    a, b = combine(va), combine(vb_rows)
    if "not_evaluated" in (a, b):
        h2 = "not_evaluated"
    elif a == "supported" and b == "supported":
        h2 = "supported"
    elif a == "contradicted" or b == "contradicted":
        h2 = "contradicted"
    else:
        h2 = "inconclusive"
    finish("H2", h2)

    # H3: gat vs gat_identity gain larger in peak than off-peak (contrast with interval)
    s3 = "gat vs gat_identity gain is larger in peak than off-peak"
    vs = []
    for h in HORIZONS:
        res = bootstrap_contrast(daily, [(1.0, "gat_identity", "gat", h, "peak"),
                                         (-1.0, "gat_identity", "gat", h, "off_peak")])
        v = "not_evaluated" if res is None else verdict(res["ci95_low"], res["ci95_high"])
        vs.append(v)
        add("H3", s3, h, "peak minus off_peak", "gain(peak) - gain(off_peak)", res, v)
    finish("H3", combine(vs))

    # H4: gat vs gat_identity gain grows with horizon (each step tested)
    s4 = "gat vs gat_identity gain grows with horizon (5 < 15 < 30 < 60 min)"
    vs = []
    for h_lo, h_hi in zip(HORIZONS[:-1], HORIZONS[1:]):
        res = bootstrap_contrast(daily, [(1.0, "gat_identity", "gat", h_hi, "all"),
                                         (-1.0, "gat_identity", "gat", h_lo, "all")])
        v = "not_evaluated" if res is None else verdict(res["ci95_low"], res["ci95_high"])
        vs.append(v)
        add("H4", s4, f"{h_hi} minus {h_lo}", "all", "gain(longer) - gain(shorter)", res, v)
    finish("H4", combine(vs))

    # H5: descriptive only
    s5 = "gain is larger at sensors with more neighbours and around congestion onsets (descriptive)"
    note = ""
    if sensor_gain is not None and not sensor_gain.empty:
        rho = sensor_gain[["graph_degree", "gain"]].corr(method="spearman").iloc[0, 1]
        note += f"Spearman(degree, per-sensor 60 min gain) = {rho:.3f} over {len(sensor_gain)} sensors. "
    if onset_gain is not None and not onset_gain.empty:
        for _, r in onset_gain.iterrows():
            note += f"{r['horizon']}: gain at proxy onsets {r['gain_onset']:.3f} vs elsewhere {r['gain_other']:.3f}. "
    add("H5", s5, "60min", "all", "gat_identity - gat", None,
        "descriptive_only" if note else "not_evaluated", note.strip())
    finish("H5", "descriptive_only" if note else "not_evaluated")

    # H0: gat and dcrnn do not differ from xgboost_ha (report either direction)
    s0 = "gat and dcrnn do not differ from xgboost_ha"
    all_v = []
    for model in ("gat", "dcrnn"):
        for h in HORIZONS:
            res = bootstrap_contrast(daily, [(1.0, "xgboost_ha", model, h, "all")])
            if res is None:
                v = "not_evaluated"
            elif res["ci95_low"] <= 0 <= res["ci95_high"]:
                v = "supported"
            else:
                v = "contradicted"
            all_v.append(v)
            direction = "" if res is None else ("candidate lower MAE" if res["mae_difference"] > 0 else "candidate higher MAE")
            add("H0", s0, h, "all", f"xgboost_ha - {model}", res, v,
                note=f"{model}: {direction}; xgboost_ha has historical-average features the deep models lack")
    finish("H0", "not_evaluated" if "not_evaluated" in all_v
           else ("supported" if all(v == "supported" for v in all_v) else "contradicted"))

    return pd.DataFrame(out, columns=HYPOTHESIS_COLUMNS)
