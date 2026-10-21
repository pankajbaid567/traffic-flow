"""Inference for the repository's numeric, scalar XGBoost JSON artifacts.

Prefer the official xgboost runtime when installed. The offline reader supports
only ordinary gbtree regression with numerical splits and one tree per round;
it rejects categorical, multi-target and other booster formats. It evaluates the
stored leaf values in tree order using float32, including best_iteration.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def _predict(X, offsets, left, right, features, conditions, default_left, base):
    result = np.empty(X.shape[0], dtype=np.float32)
    for row in prange(X.shape[0]):
        total = np.float32(base)
        for tree in range(len(offsets) - 1):
            offset = offsets[tree]
            node = offset
            while left[node] != -1:
                value = X[row, features[node]]
                go_left = (
                    default_left[node] if np.isnan(value) else value < conditions[node]
                )
                node = offset + (left[node] if go_left else right[node])
            total = np.float32(total + conditions[node])
        result[row] = total
    return result


class SavedBooster:
    runtime = "numeric_json_numba"

    def __init__(self, path):
        self.path = Path(path)
        document = json.loads(self.path.read_text())
        learner = document["learner"]
        param = learner["learner_model_param"]
        booster = learner["gradient_booster"]
        if (
            booster["name"] != "gbtree"
            or int(param.get("num_target", "1")) != 1
            or int(param["num_class"]) != 0
        ):
            raise ValueError("Offline reader supports scalar gbtree regression only")
        if learner["objective"]["name"] not in [
            "reg:absoluteerror",
            "reg:squarederror",
        ]:
            raise ValueError(
                "Offline reader requires an identity-link regression objective"
            )
        self.feature_names = learner["feature_names"]
        self.best_iteration = int(learner["attributes"]["best_iteration"])
        model = booster["model"]
        if model.get("iteration_indptr", list(range(len(model["trees"]) + 1))) != list(
            range(len(model["trees"]) + 1)
        ):
            raise ValueError(
                "Offline reader requires exactly one tree per boosting round"
            )
        trees = model["trees"][: self.best_iteration + 1]
        if any(any(t["split_type"]) or t["categories"] for t in trees):
            raise ValueError("Categorical splits are not supported by offline reader")
        if any(model["tree_info"]):
            raise ValueError("Multi-class trees are not supported")
        self.trees = trees
        base = json.loads(param["base_score"])
        self.base = np.float32(base[0] if isinstance(base, list) else base)
        self.offsets = np.cumsum([0] + [len(t["left_children"]) for t in trees]).astype(
            np.int32
        )
        self.arrays = [
            np.concatenate([np.asarray(t[k], dtype=d) for t in trees])
            for k, d in [
                ("left_children", np.int32),
                ("right_children", np.int32),
                ("split_indices", np.int32),
                ("split_conditions", np.float32),
                ("default_left", np.bool_),
            ]
        ]

    def predict_frame(self, df, features):
        if list(features) != self.feature_names:
            raise ValueError("Saved booster feature order mismatch")
        X = np.ascontiguousarray(df[features].to_numpy(np.float32))
        delta = _predict(X, self.offsets, *self.arrays, self.base)
        return df["speed_lag_0"].to_numpy(float) + delta

    def get_score(self, importance_type="gain"):
        if importance_type != "gain":
            raise ValueError("Only gain importance is supported")
        gains = {}
        for t in self.trees:
            for i, child in enumerate(t["left_children"]):
                if child != -1:
                    name = self.feature_names[t["split_indices"][i]]
                    gains.setdefault(name, []).append(t["loss_changes"][i])
        return {name: float(np.mean(values)) for name, values in gains.items()}


def load_saved_booster(path):
    try:
        import xgboost as xgb
    except ImportError:
        return SavedBooster(path)
    booster = xgb.Booster(params={"nthread": 4})
    booster.load_model(path)
    return booster
