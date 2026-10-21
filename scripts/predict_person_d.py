"""Regenerate all-row RF predictions from saved full-feature models."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
from src.data.dataset_builder import load_partition
from src.data.features import ALL_FEATURE_COLUMNS, TARGET_HORIZONS


def main():
    outdir = ROOT / "data/processed/predictions"
    outdir.mkdir(parents=True, exist_ok=True)
    for split in ["val", "test"]:
        df = load_partition(split)
        keep = [
            "origin_timestamp",
            "sensor_id",
            "is_morning_peak",
            "is_evening_peak",
            "speed_lag_0",
        ]
        keep += [
            f"{prefix}_{h}"
            for h in TARGET_HORIZONS.values()
            for prefix in ["target", "target_valid"]
        ]
        out = df[keep].copy()
        out["sensor_id"] = out.sensor_id.astype(str)
        for h in TARGET_HORIZONS.values():
            path = ROOT / f"models/person_d/rf_exp_D_{h}.joblib"
            model = joblib.load(path)
            if list(model.feature_names_in_) != ALL_FEATURE_COLUMNS:
                raise ValueError("RF feature contract differs from shared 32 features")
            out[f"pred_random_forest_{h}"] = model.predict(
                df[ALL_FEATURE_COLUMNS]
            ).astype(np.float32)
            print(split, h, "predicted", len(out), flush=True)
            del model
        out.to_parquet(outdir / f"person_d_{split}_predictions.parquet", index=False)


if __name__ == "__main__":
    main()
