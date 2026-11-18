"""Bootstrap helpers for Phase 2 (Member 4).

The comparison and hypothesis logic lives in ``src.phase2.evaluation.hypotheses``
(period-filtered daily sums, seed averaging, contrast intervals for H3/H4). This
module keeps the names older code imports and the Phase 1 reproduction check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.phase2.common import ROOT
from src.phase2.evaluation.hypotheses import (  # noqa: F401  (re-exported)
    GRAPH_COMPARISONS,
    PHASE1_COMPARISONS,
    block_indexes,
    bootstrap_contrast,
    evaluate_hypotheses,
    run_comparisons,
)

PHASE1_BOOTSTRAP_PATH = ROOT / "data" / "metadata" / "paired_bootstrap.csv"


def verify_phase1_bootstrap_reproduction(
    computed_bootstrap_df: pd.DataFrame,
    expected_csv_path: Path | str = PHASE1_BOOTSTRAP_PATH,
    tol: float = 1e-6,
) -> Dict[str, float]:
    """Check that the computed intervals reproduce Phase 1 ``paired_bootstrap.csv``."""
    expected = pd.read_csv(expected_csv_path)
    comp = computed_bootstrap_df
    if "period" in comp:
        comp = comp[comp["period"] == "all"]
    merged = expected.merge(comp, on=["baseline", "candidate", "horizon"], suffixes=("_expected", "_computed"))
    if len(merged) != len(expected):
        raise AssertionError(f"missing {len(expected) - len(merged)} Phase 1 comparisons")
    out = {}
    for col in ("mae_difference", "ci95_low", "ci95_high"):
        out[f"max_{col}_deviation"] = float(np.max(np.abs(merged[f"{col}_expected"] - merged[f"{col}_computed"])))
        if out[f"max_{col}_deviation"] > tol:
            raise AssertionError(f"{col} deviates by {out[f'max_{col}_deviation']}")
    return out
