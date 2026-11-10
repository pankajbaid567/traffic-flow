"""Measure seconds per epoch and peak memory of small dummy models (Member 1).

Trains ``dummy_gru`` for 2 epochs on a subset of METR-LA on each requested device, in a
separate subprocess per run so peak memory is not shared. Epoch 2 is reported (epoch 1
includes warm-up). It then extrapolates linearly to the full split sizes.

These numbers are a LOWER BOUND for GAT/DCRNN, which do far more work per batch than the
dummy GRU. Use them to size caps, then replace them with the real models' numbers.

    python scripts/phase2_measure_speed.py [--devices cpu mps] [--subset 0.05]
    (on Colab: --devices cuda)

Nothing is written to the repository; checkpoints go to a temp folder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CASES = {
    "small (hidden 16, no graph, batch 32)": {"params": {"hidden": 16, "mix": False}, "batch_size": 32, "graph": "identity"},
    "heavier (hidden 64, neighbour mix, batch 32)": {"params": {"hidden": 64, "mix": True}, "batch_size": 32, "graph": "bidirectional"},
}
FULL_TRAIN_ORIGINS = 23967
FULL_VAL_ORIGINS = 3416


def worker(device: str, case: str, subset: float) -> None:
    import torch

    from src.phase2.common import resolve_device, set_seed
    from src.phase2.data import load_data
    from src.phase2.graph import load_graph
    from src.phase2.registry import build_model
    from src.phase2.train import fit, model_cfg

    spec = CASES[case]
    dev = resolve_device(device)
    data = load_data({"device": str(dev), "subset": subset})
    adj = load_graph(spec["graph"])
    set_seed(0)
    model = build_model("dummy_gru", model_cfg(data, spec["params"]))
    with tempfile.TemporaryDirectory() as tmp:
        result = fit(
            model, data.datasets(), adj,
            {"lr": 1e-3, "batch_size": spec["batch_size"], "epochs": 2, "patience": 5, "scheduler": "none"},
            dev, Path(tmp), seed=0, label=case, log=lambda s: None,
        )
        rows = [json.loads(line) for line in (Path(tmp) / "epochs.jsonl").read_text().splitlines()]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / 1e6 if sys.platform == "darwin" else rss / 1e3  # macOS reports bytes, Linux KB
    device_mb = None
    if dev.type == "mps":
        device_mb = torch.mps.driver_allocated_memory() / 1e6
    elif dev.type == "cuda":
        device_mb = torch.cuda.max_memory_allocated() / 1e6
    print(
        "RESULT "
        + json.dumps(
            {
                "device": str(dev),
                "case": case,
                "params": result["params"],
                "epoch_seconds": [r["epoch_seconds"] for r in rows],
                "batches_train": rows[-1]["batches"],
                "n_train": len(data.dataset("train")),
                "n_val": len(data.dataset("val")),
                "peak_rss_mb": round(rss_mb, 1),
                "device_memory_mb": None if device_mb is None else round(device_mb, 1),
            }
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--devices", nargs="+", default=["cpu", "mps"])
    parser.add_argument("--subset", type=float, default=0.05)
    parser.add_argument("--worker", nargs=2, metavar=("DEVICE", "CASE"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker[0], args.worker[1], args.subset)
        return 0

    print(f"subset {args.subset}: epoch 2 timing (epoch 1 = warm-up), linear extrapolation to the full splits\n")
    header = f"{'device':<7}{'model':<46}{'params':>8}{'s/epoch':>9}{'full epoch':>12}{'peak RSS':>10}{'device mem':>12}"
    print(header)
    for device in args.devices:
        for case in CASES:
            proc = subprocess.run(
                [sys.executable, "-B", __file__, "--worker", device, case, "--subset", str(args.subset)],
                capture_output=True, text=True, cwd=ROOT,
            )
            line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
            if line is None:
                print(f"{device:<7}{case:<46}  FAILED: {proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else 'no output'}")
                continue
            r = json.loads(line[7:])
            warm = r["epoch_seconds"][-1]
            scale = (FULL_TRAIN_ORIGINS + FULL_VAL_ORIGINS * 0.3) / (r["n_train"] + r["n_val"] * 0.3)
            # training cost ~ train batches; validation is forward-only, counted at ~0.3x per origin
            full_minutes = warm * scale / 60
            dev_mem = "-" if r["device_memory_mb"] is None else f"{r['device_memory_mb']:.0f} MB"
            print(f"{r['device']:<7}{case:<46}{r['params']:>8}{warm:>8.1f}s{full_minutes:>10.1f} m{r['peak_rss_mb']:>8.0f} MB{dev_mem:>12}")
    print("\n'full epoch' = measured warm epoch scaled by origins (train + 0.3 x validation); data load time excluded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
