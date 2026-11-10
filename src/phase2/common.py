"""Small shared helpers: device choice, seeding, atomic file writes, output paths."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]

HORIZON_STEPS = (1, 3, 6, 12)
HORIZON_NAMES = ("5min", "15min", "30min", "60min")
HISTORY_STEPS = 12
NUM_CHANNELS = 7
CHANNEL_NAMES = (
    "speed_z",
    "missing_flag",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
)


def resolve_device(name: str | None = "auto") -> torch.device:
    """``auto`` picks cuda, then mps, then cpu. Explicit names are validated."""
    name = (name or "auto").lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    if name == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("--device mps requested but MPS is not available")
    return torch.device(name)


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (all devices). Device kernels may stay non-deterministic."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def config_hash(obj: Any) -> str:
    """Short stable hash of a JSON-able config."""
    text = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def git_commit(root: Path = ROOT) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sync_device(device: torch.device) -> None:
    """Wait for queued kernels so wall-clock timings are honest."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write via a temp file in the same folder, then rename (safe if Colab/Drive disconnects)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_bytes(path, (json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n").encode())


def atomic_torch_save(obj: Any, path: Path) -> None:
    import io

    buffer = io.BytesIO()
    torch.save(obj, buffer)
    atomic_write_bytes(path, buffer.getvalue())


def output_root(out: str | Path | None = None, subset: float = 1.0) -> Path:
    """Where Phase 2 artifacts go: ``--out`` > ``$PHASE2_OUT`` > ``data/processed/phase2``.

    Subset runs (smoke tests) get their own ``subset_<frac>`` folder so they can never
    be mistaken for real results.
    """
    base = Path(out) if out else Path(os.environ.get("PHASE2_OUT") or ROOT / "data/processed/phase2")
    if subset < 1.0:
        base = base / f"subset_{subset:g}"
    return base
