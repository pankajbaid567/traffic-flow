"""Road-graph loader and control variants for Phase 2 (Member 1).

Source: ``data/raw/adj_mx.pkl`` (DCRNN, pinned commit in ``data/raw/manifest.json``):
a dense 207 x 207 float32 matrix in the same sensor order as the speed table.

Edge direction (verified from the DCRNN code, ``scripts/gen_adj_mx.py``)
-----------------------------------------------------------------------
``A[i, j]`` is a thresholded Gaussian kernel of the **road distance from sensor i
to sensor j** (``distances_la_2012.csv`` columns ``from,to,cost``; the file fills
``dist_mx[from_index, to_index]``)::

    A[i, j] = exp(-(d(i -> j) / std(d)) ** 2),  set to 0 if < 0.1,  A[i, i] = 1

So the matrix is **directed** (1,722 non-zeros, not symmetric). With row
normalisation ``P = D^-1 A`` and the product ``P @ h``, sensor ``i`` aggregates
over sensors ``j`` with ``A[i, j] > 0``, i.e. sensors that can be *reached from*
``i`` along the road (``i`` sees its downstream neighbours). The transposed
matrix (``variant="given_T"``) lets ``i`` see the sensors that reach it
(upstream). DCRNN uses both (``D^-1 A`` and ``D^-1 A^T``). The weights come only
from sensor distances, never from the speed series.

"Downstream/upstream" here follows the stored distance direction; Phase 1 notes
that traffic direction is not independently verified, so treat it as the graph's
convention, not a verified flow direction.

Variants returned by :func:`load_graph` (dense float32 ``[N, N]``, CPU tensor)
-------------------------------------------------------------------------------
``given``           A as supplied (directed)
``given_T``         A transposed (direction check)
``bidirectional``   max(A, A^T)
``identity``        self-loops only (same architecture, no neighbours)
``shuffled``        ``base`` with sensors relabelled by a fixed derangement:
                    ``S[i, j] = base[s(i), s(j)]``. Same degree multiset, wrong
                    neighbours (every sensor gets another sensor's neighbourhood).
``random_matched``  random graph with the same number of off-diagonal edges as
                    ``base`` (symmetric if ``base`` is symmetric), weights taken
                    by shuffling ``base``'s edge weights, diagonal 1.

``base`` (default ``bidirectional``) is the graph that ``shuffled`` and
``random_matched`` are derived from; use the same base as the main model.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from src.data.load_data import load_graph_metadata, load_sensor_metadata
from src.phase2.common import ROOT

VARIANTS = ("given", "given_T", "bidirectional", "identity", "shuffled", "random_matched")
_ADJ_CACHE: dict[str, np.ndarray] = {}


def load_adjacency(path: str | Path = ROOT / "data/raw/adj_mx.pkl", check_order: bool = True) -> np.ndarray:
    """Load the supplied adjacency as float32 ``[N, N]``; optionally check sensor order."""
    key = str(path)
    if key not in _ADJ_CACHE:
        ids, _, adj = load_graph_metadata(path)
        adj = np.asarray(adj, dtype=np.float32)
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1] or adj.shape[0] != len(ids):
            raise ValueError(f"adjacency must be square with one row per sensor id, got {adj.shape}, {len(ids)} ids")
        if check_order:
            sensors = load_sensor_metadata()["sensor_id"].astype(str).tolist()
            if ids != sensors:
                raise ValueError("adjacency sensor order differs from data/metadata/sensors.csv")
        _ADJ_CACHE[key] = adj
    return _ADJ_CACHE[key].copy()


def derangement(n: int, seed: int) -> np.ndarray:
    """Random permutation with no fixed point (a single n-cycle through a random order)."""
    if n < 2:
        raise ValueError("a derangement needs at least 2 nodes")
    order = np.random.default_rng(seed).permutation(n)
    mapping = np.empty(n, dtype=np.int64)
    mapping[order] = np.roll(order, -1)
    return mapping


def shuffle_graph(base: np.ndarray, seed: int = 0) -> np.ndarray:
    """Relabel sensors: ``S[i, j] = base[s(i), s(j)]`` with ``s`` a derangement."""
    s = derangement(base.shape[0], seed)
    return np.ascontiguousarray(base[np.ix_(s, s)])


def random_matched_graph(base: np.ndarray, seed: int = 0) -> np.ndarray:
    """Random graph with the same number of off-diagonal edges and shuffled edge weights."""
    n = base.shape[0]
    rng = np.random.default_rng(seed)
    out = np.zeros_like(base)
    symmetric = np.allclose(base, base.T)
    if symmetric:
        iu = np.triu_indices(n, k=1)
        weights = base[iu]
        weights = weights[weights > 0]
        candidates = np.flatnonzero(np.ones(len(iu[0]), dtype=bool))
        chosen = rng.choice(candidates, size=len(weights), replace=False)
        rows, cols = iu[0][chosen], iu[1][chosen]
        out[rows, cols] = rng.permutation(weights)
        out[cols, rows] = out[rows, cols]
    else:
        off = ~np.eye(n, dtype=bool)
        weights = base[off]
        weights = weights[weights > 0]
        flat = np.flatnonzero(off.ravel())
        chosen = rng.choice(flat, size=len(weights), replace=False)
        out.ravel()[chosen] = rng.permutation(weights)
    np.fill_diagonal(out, 1.0)
    return out


def make_variant(adj: np.ndarray, variant: str, seed: int = 0, base: str = "bidirectional") -> np.ndarray:
    """Pure-numpy variant builder (also used by tests on small hand-made graphs)."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown graph variant {variant!r}; choose from {VARIANTS}")
    if variant == "given":
        return adj.copy()
    if variant == "given_T":
        return np.ascontiguousarray(adj.T)
    if variant == "bidirectional":
        return np.maximum(adj, adj.T)
    if variant == "identity":
        return np.eye(adj.shape[0], dtype=np.float32)
    if base in ("shuffled", "random_matched", "identity"):
        raise ValueError(f"base graph cannot be {base!r}")
    base_adj = make_variant(adj, base, seed)
    builder: Callable[[np.ndarray, int], np.ndarray] = (
        shuffle_graph if variant == "shuffled" else random_matched_graph
    )
    return builder(base_adj, seed)


def load_graph(variant: str = "given", seed: int = 0, base: str = "bidirectional") -> torch.Tensor:
    """Dense float32 ``[207, 207]`` adjacency for ``variant`` (see module docstring)."""
    adj = load_adjacency()
    return torch.from_numpy(make_variant(adj, variant, seed=seed, base=base).astype(np.float32))


def distances_csv_path() -> Path | None:
    """Locate DCRNN's ``distances_la_2012.csv``: ``$PHASE2_DISTANCES_CSV`` or ``../DCRNN/data/sensor_graph``."""
    env = os.environ.get("PHASE2_DISTANCES_CSV")
    candidates = [Path(env)] if env else []
    candidates.append(ROOT.parent / "DCRNN/data/sensor_graph/distances_la_2012.csv")
    for path in candidates:
        if path.is_file():
            return path
    return None


def regenerate_adjacency(distances_csv: str | Path, sensor_ids: list[str], normalized_k: float = 0.1) -> np.ndarray:
    """Re-run DCRNN's ``get_adjacency_matrix`` (float32 throughout, as in the original)."""
    import pandas as pd

    frame = pd.read_csv(distances_csv, dtype={"from": str, "to": str})
    n = len(sensor_ids)
    index = {s: i for i, s in enumerate(sensor_ids)}
    dist = np.full((n, n), np.inf, dtype=np.float32)
    for src, dst, cost in frame.itertuples(index=False):
        if src in index and dst in index:
            dist[index[src], index[dst]] = cost
    finite = dist[~np.isinf(dist)].flatten()
    std = finite.std()
    adj = np.exp(-np.square(dist / std))
    adj[adj < normalized_k] = 0
    return adj.astype(np.float32)


def verify_adjacency(distances_csv: str | Path | None = None) -> dict:
    """Compare ``adj_mx.pkl`` with a matrix regenerated from road distances.

    Returns a dict with ``status`` = ``"skipped"`` (distances file absent) or
    ``"checked"`` plus ``max_abs_diff`` and edge-count agreement.
    """
    path = Path(distances_csv) if distances_csv else distances_csv_path()
    if path is None or not path.is_file():
        return {"status": "skipped", "reason": "distances_la_2012.csv not found (set PHASE2_DISTANCES_CSV)"}
    ids = load_sensor_metadata()["sensor_id"].astype(str).tolist()
    given = load_adjacency()
    regenerated = regenerate_adjacency(path, ids)
    return {
        "status": "checked",
        "distances_csv": str(path),
        "max_abs_diff": float(np.abs(given - regenerated).max()),
        "nnz_given": int((given > 0).sum()),
        "nnz_regenerated": int((regenerated > 0).sum()),
        "same_support": bool(np.array_equal(given > 0, regenerated > 0)),
        "symmetric": bool(np.allclose(given, given.T)),
    }
