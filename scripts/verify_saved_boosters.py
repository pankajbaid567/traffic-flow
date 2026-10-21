"""Compare compiled offline predictions to a direct scalar JSON tree walk.

This checks every saved model on deterministic real validation rows. It does not
claim execution through the unavailable official XGBoost training package.
"""

from pathlib import Path
import json, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.models.saved_booster import SavedBooster
from src.models.baselines import HistoricalAverage, HA_FEATURE_COLUMNS
from src.data.dataset_builder import load_partition
from src.data.features import ALL_FEATURE_COLUMNS, TARGET_HORIZONS
from src.data.load_data import load_raw_speed_data


def reference_predict(document, frame, features):
    learner = document["learner"]
    trees = learner["gradient_booster"]["model"]["trees"]
    best = int(learner["attributes"]["best_iteration"])
    base = json.loads(learner["learner_model_param"]["base_score"])
    base = np.float32(base[0] if isinstance(base, list) else base)
    X = frame[features].to_numpy(np.float32)
    output = []
    for x in X:
        total = base
        for tree in trees[: best + 1]:
            node = 0
            while tree["left_children"][node] != -1:
                value = x[tree["split_indices"][node]]
                left = (
                    bool(tree["default_left"][node])
                    if np.isnan(value)
                    else value < np.float32(tree["split_conditions"][node])
                )
                node = (
                    tree["left_children"][node]
                    if left
                    else tree["right_children"][node]
                )
            total = np.float32(total + np.float32(tree["split_conditions"][node]))
        output.append(total)
    return frame.speed_lag_0.to_numpy(float) + np.asarray(output, dtype=np.float32)


def main():
    df = load_partition("val")
    indices = np.linspace(0, len(df) - 1, 96, dtype=int)
    sample = df.iloc[indices].copy()
    contract = json.loads((ROOT / "data/metadata/split_contract.json").read_text())
    history = HistoricalAverage().fit(
        load_raw_speed_data(), contract["partitions"]["train"]["end_inclusive"]
    )
    checks = []
    for steps, h in TARGET_HORIZONS.items():
        extras = history.features(
            sample.origin_timestamp,
            sample.sensor_id.cat.codes.to_numpy(),
            sample.speed_lag_0.to_numpy(),
            steps,
            False,
        )
        for col in HA_FEATURE_COLUMNS:
            sample[col] = extras[col].to_numpy()
        for prefix, features in [
            ("xgb", ALL_FEATURE_COLUMNS),
            ("xgb_ha", ALL_FEATURE_COLUMNS + HA_FEATURE_COLUMNS),
        ]:
            path = ROOT / f"models/person_c/{prefix}_{h}.json"
            compiled = SavedBooster(path).predict_frame(sample, features)
            scalar = reference_predict(json.loads(path.read_text()), sample, features)
            maximum = float(np.max(np.abs(compiled - scalar)))
            if maximum != 0:
                raise AssertionError(
                    f"Compiled and scalar JSON walks differ: {path} {maximum}"
                )
            check = {
                "model": path.name,
                "validation_rows": len(sample),
                "maximum_absolute_difference": maximum,
            }
            try:
                import xgboost as xgb
            except ImportError:
                check["official_runtime_comparison"] = "unavailable"
            else:
                official = xgb.Booster(params={"nthread": 2})
                official.load_model(path)
                delta = official.predict(
                    xgb.DMatrix(
                        sample[features].to_numpy(np.float32), feature_names=features
                    ),
                    iteration_range=(0, official.best_iteration + 1),
                )
                prediction = sample.speed_lag_0.to_numpy(float) + delta
                difference = float(np.max(np.abs(compiled - prediction)))
                check["official_runtime_version"] = xgb.__version__
                check["official_runtime_maximum_absolute_difference"] = difference
                if difference > 1e-4:
                    raise AssertionError(
                        f"Offline and official prediction difference: {path} {difference}"
                    )
            checks.append(check)
            print(path.name, maximum, flush=True)
    (ROOT / "data/metadata/saved_booster_verification.json").write_text(
        json.dumps(
            {
                "method": "direct scalar JSON walk, compiled numeric reader, and official XGBoost when available",
                "checks": checks,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
