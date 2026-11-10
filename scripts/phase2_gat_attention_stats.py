"""Compute attention statistics from exported GAT attention maps.

Owner: Member 2 (phase2/m2-gat). Contract: reports/phase2_plan.md section 8.2.

Computes:
- Per-sensor mean attention entropy
- Attention mass on self vs neighbours
- Attention mass on upstream vs downstream neighbours (from directed 'given' graph)
- Split by peak vs off-peak origins

Writes data/metadata/phase2/attention_stats.csv.
Exits cleanly if the attention file does not exist.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.models.prediction_evaluation import temporal_masks
from src.phase2.graph import load_graph


def compute_attention_stats(
    attn: np.ndarray,
    origin_ns: np.ndarray,
    adj_directed: np.ndarray,
    sensor_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Compute per-sensor attention statistics across origins, layers, and heads.

    Args:
        attn: [B, layers, heads, N, N] float array where sum over last dim == 1.
        origin_ns: [B] int64 nanosecond timestamps.
        adj_directed: [N, N] directed adjacency matrix ('given').
            A[i, j] > 0 means j is reachable from i (downstream of i).
        sensor_ids: Optional list of N sensor ID strings.

    Returns:
        DataFrame with columns:
        sensor_id, sensor_idx, period, entropy, mass_self, mass_neighbours,
        mass_downstream, mass_upstream, num_downstream, num_upstream
    """
    b, layers, heads, n, _ = attn.shape
    if sensor_ids is None:
        sensor_ids = [str(i) for i in range(n)]

    # Masks for given graph direction:
    # A[i, j] > 0 means node j is downstream from node i
    is_self = np.eye(n, dtype=bool)
    is_downstream = (adj_directed > 0) & (~is_self)
    is_upstream = (adj_directed.T > 0) & (~is_self)
    is_neighbour = is_downstream | is_upstream

    # Classify origins into peak vs off-peak
    timestamps = pd.to_datetime(origin_ns, unit="ns")
    masks = temporal_masks(timestamps)

    rows = []
    # Average across layers and heads: [B, N, N]
    mean_attn = attn.mean(axis=(1, 2))  # [B, N, N]

    periods = ["all", "peak", "off_peak"]

    for period in periods:
        p_mask = masks[period]
        if not np.any(p_mask):
            continue
        # Attention for this period: [B_sub, N, N] -> mean over origins -> [N, N]
        period_attn = mean_attn[p_mask].mean(axis=0)  # [N, N]

        # Row-wise entropy: -sum(p * log(p))
        eps = 1e-12
        entropy = -np.sum(period_attn * np.log(period_attn + eps), axis=-1)  # [N]

        # Mass self: diagonal elements
        mass_self = np.diag(period_attn)  # [N]

        for i in range(n):
            row = period_attn[i]
            down_mask = is_downstream[i]
            up_mask = is_upstream[i]
            neigh_mask = is_neighbour[i]

            m_down = float(row[down_mask].sum())
            m_up = float(row[up_mask].sum())
            m_neigh = float(row[neigh_mask].sum())

            rows.append(
                {
                    "sensor_id": sensor_ids[i],
                    "sensor_idx": i,
                    "period": period,
                    "entropy": float(entropy[i]),
                    "mass_self": float(mass_self[i]),
                    "mass_neighbours": m_neigh,
                    "mass_downstream": m_down,
                    "mass_upstream": m_up,
                    "num_downstream": int(down_mask.sum()),
                    "num_upstream": int(up_mask.sum()),
                }
            )

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute GAT attention statistics.")
    parser.add_argument(
        "--attn-file",
        type=Path,
        default=None,
        help="Path to gat attention npz file (e.g. data/processed/phase2/attention/gat_seed42.npz)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/metadata/phase2/attention_stats.csv"),
        help="Output CSV destination",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed to look for if --attn-file is omitted",
    )
    parser.add_argument(
        "--test-synthetic",
        action="store_true",
        help="Run verification on synthetic data and exit",
    )
    args = parser.parse_args()

    if args.test_synthetic:
        print("Running verification on synthetic attention data...")
        b, layers, heads, n = 4, 1, 2, 10
        raw_logits = np.random.randn(b, layers, heads, n, n)
        # Softmax over last dimension
        exp = np.exp(raw_logits - raw_logits.max(axis=-1, keepdims=True))
        attn = exp / exp.sum(axis=-1, keepdims=True)
        # Mock origin timestamps
        origin_ns = pd.date_range("2012-06-01 08:00", periods=b, freq="3h").asi8
        adj = (np.random.rand(n, n) > 0.7).astype(np.float32)
        np.fill_diagonal(adj, 1.0)
        df = compute_attention_stats(attn, origin_ns, adj)
        assert not df.empty, "Synthetic DataFrame is empty"
        assert "entropy" in df.columns
        print(f"Synthetic test PASSED. Generated {len(df)} rows.")
        return 0

    attn_path = args.attn_file
    if attn_path is None:
        out_root = Path(os.environ.get("PHASE2_OUT", "data/processed/phase2"))
        candidate = out_root / "attention" / f"gat_seed{args.seed}.npz"
        if candidate.is_file():
            attn_path = candidate
        else:
            # Check for any gat_seed*.npz
            candidates = list((out_root / "attention").glob("gat_seed*.npz"))
            if candidates:
                attn_path = candidates[0]

    if attn_path is None or not attn_path.is_file():
        print(
            f"Attention file not found (checked {attn_path}). "
            "Attention statistics will be generated when full training exports attention maps."
        )
        return 0

    print(f"Loading attention maps from {attn_path}...")
    data = np.load(attn_path)
    attn = data["attn"].astype(np.float32)  # [B, layers, heads, N, N]
    origin_ns = data["origin_ns"]

    # Directed graph for upstream/downstream distinction
    adj_directed = load_graph("given")

    df = compute_attention_stats(attn, origin_ns, adj_directed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Attention stats saved to {args.out} ({len(df)} rows).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
