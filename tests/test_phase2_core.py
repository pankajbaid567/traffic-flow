"""Tests for the Phase 2 core (Member 1): data, graph, registry, trainer, CLI.

Tests that need ``data/raw/metr-la.h5`` / ``adj_mx.pkl`` (git-ignored, copied from the
shared Drive) are skipped with an explicit message if those files are absent; every
other test runs on small synthetic data.

    python -m unittest discover -s tests -p "test_phase2_*.py" -v
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.features import compute_temporal_features
from src.models.metrics import compute_metrics
from src.phase2 import cli
from src.phase2.common import HORIZON_NAMES, ROOT, set_seed
from src.phase2.data import Phase2Data, load_data
from src.phase2.graph import (
    VARIANTS,
    derangement,
    load_adjacency,
    load_graph,
    make_variant,
    verify_adjacency,
)
from src.phase2.registry import build_model, get_builder, list_models, register_model
from src.phase2.train import (
    InterruptedForTest,
    TrainConfig,
    attention_origin_positions,
    export_attention,
    fit,
    predict,
)

RAW = ROOT / "data/raw/metr-la.h5"
ADJ = ROOT / "data/raw/adj_mx.pkl"
HAS_DATA = RAW.is_file() and ADJ.is_file()
NEED_DATA = unittest.skipUnless(
    HAS_DATA, "data/raw/metr-la.h5 or adj_mx.pkl missing: copy them from the shared Drive (plan section 0)"
)


# ----------------------------------------------------------------------------- synthetic data
def make_contract(timestamps: pd.DatetimeIndex) -> dict:
    n = len(timestamps)
    n_train, n_val = int(0.7 * n), int(0.1 * n)
    bounds = {"train": (0, n_train - 1), "validation": (n_train, n_train + n_val - 1), "test": (n_train + n_val, n - 1)}
    return {
        "history_steps": 12,
        "horizon_steps": [1, 3, 6, 12],
        "partitions": {
            name: {
                "start_inclusive": str(timestamps[lo]),
                "end_inclusive": str(timestamps[hi]),
                "grid_timestamps": hi - lo + 1,
            }
            for name, (lo, hi) in bounds.items()
        },
    }


def make_frame(steps: int = 1500, nodes: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2012-03-05", periods=steps, freq="5min")  # starts on a Monday
    t = np.arange(steps)[:, None]
    values = 50 + 12 * np.sin(2 * np.pi * t / 288) + rng.normal(0, 3, (steps, nodes)) + np.arange(nodes)
    values = np.clip(values, 5, 70)
    values[rng.random((steps, nodes)) < 0.05] = 0.0  # unavailable readings
    values[rng.random((steps, nodes)) < 0.005] = np.nan
    values[:5, 0] = 0.0  # leading gap -> training-median fallback
    values[1100:1140, 2] = 0.0  # outage inside the validation/test area
    return pd.DataFrame(values, index=index, columns=[str(1000 + i) for i in range(nodes)])


def make_data(frame: pd.DataFrame | None = None) -> Phase2Data:
    frame = make_frame() if frame is None else frame
    return Phase2Data(frame, make_contract(frame.index), device="cpu")


_REAL: Phase2Data | None = None


def real_data() -> Phase2Data:
    global _REAL
    if _REAL is None:
        _REAL = load_data({"device": "cpu"})
    return _REAL


# ----------------------------------------------------------------------------- (a) parity
@NEED_DATA
class Phase1ParityTest(unittest.TestCase):
    """The new pipeline plus ``last_value`` must reproduce Phase 1 persistence exactly."""

    @classmethod
    def setUpClass(cls):
        cls.data = real_data()
        cls.metrics = pd.read_csv(ROOT / "data/metadata/phase1_metrics.csv")

    def test_origin_counts_equal_phase1(self):
        sizes = {k: len(v) for k, v in self.data.datasets().items()}
        self.assertEqual(sizes, {"train": 23967, "val": 3416, "test": 6844})

    def test_shapes_and_channels(self):
        self.assertEqual(tuple(self.data.features.shape), (34272, 207, 7))
        self.assertEqual(self.data.features.dtype, torch.float32)
        batch = self.data.dataset("test").get_batch([0, 1, 2])
        self.assertEqual(tuple(batch["x"].shape), (3, 12, 207, 7))
        self.assertEqual(tuple(batch["y"].shape), (3, 4, 207))
        self.assertEqual(batch["y_valid"].dtype, torch.bool)

    def test_last_value_reproduces_persistence(self):
        model = build_model("last_value", {})
        eye = torch.eye(207)
        expected_counts = {"5min": 1245052, "15min": 1245040, "30min": 1245022, "60min": 1244986}
        for split, label in [("test", "test"), ("val", "val")]:
            ds = self.data.dataset(split)
            y, valid = ds.targets_numpy()
            pred = predict(model, ds, eye, "cpu")
            masks = ds.period_masks()
            for k, horizon in enumerate(HORIZON_NAMES):
                for period in ("all", "peak", "off_peak"):
                    m = valid[:, k, :] & masks[period][:, None]
                    row = self.metrics[
                        (self.metrics.split == label)
                        & (self.metrics.horizon == horizon)
                        & (self.metrics.model == "persistence")
                        & (self.metrics.period == period)
                    ].iloc[0]
                    self.assertEqual(int(m.sum()), int(row.n), f"{split} {horizon} {period} count")
                    got = compute_metrics(y[:, k, :][m], pred[:, k, :][m])
                    self.assertAlmostEqual(got["mae"], row.mae, delta=1e-3, msg=f"{split} {horizon} {period} MAE")
                    self.assertAlmostEqual(got["rmse"], row.rmse, delta=1e-3, msg=f"{split} {horizon} {period} RMSE")
                    if split == "test" and period == "all":
                        self.assertEqual(int(m.sum()), expected_counts[horizon])

    def test_graph_and_sensor_order(self):
        sensors = pd.read_csv(ROOT / "data/metadata/sensors.csv", dtype={"sensor_id": str}).sensor_id.tolist()
        self.assertEqual(self.data.sensor_ids, sensors)
        self.assertEqual(load_adjacency().shape, (207, 207))


# ----------------------------------------------------------------------------- (b) leakage
class LeakageTest(unittest.TestCase):
    def setUp(self):
        self.frame = make_frame()
        self.data = make_data(self.frame)
        self.train_end = self.data.partitions["train"].end_idx
        self.val_end = self.data.partitions["validation"].end_idx

    def test_scaler_and_fill_ignore_validation_and_test_values(self):
        changed = self.frame.copy()
        rng = np.random.default_rng(1)
        changed.iloc[self.train_end + 1 :] = rng.uniform(1, 200, changed.iloc[self.train_end + 1 :].shape)
        other = make_data(changed)
        self.assertEqual(self.data.scaler, other.scaler)
        np.testing.assert_array_equal(self.data.fill_medians, other.fill_medians)
        np.testing.assert_array_equal(
            self.data.features[: self.train_end + 1].numpy(), other.features[: self.train_end + 1].numpy()
        )
        np.testing.assert_array_equal(self.data.filled[: self.train_end + 1], other.filled[: self.train_end + 1])
        np.testing.assert_array_equal(self.data.targets[: self.train_end + 1].numpy(), other.targets[: self.train_end + 1].numpy())

    def test_scaler_uses_valid_training_cells_only(self):
        raw = self.frame.to_numpy()[: self.train_end + 1]
        valid = np.isfinite(raw) & (raw > 0)
        self.assertAlmostEqual(self.data.scaler.mean, float(raw[valid].mean()), places=9)
        self.assertAlmostEqual(self.data.scaler.std, float(raw[valid].std()), places=9)

    def test_fill_is_causal(self):
        """Changing a reading at time t0 never changes filled inputs before t0."""
        t0 = 1300  # inside the test partition (starts at index 1200 for 1500 steps)
        changed = self.frame.copy()
        changed.iloc[t0:, :] = changed.iloc[t0:, :].to_numpy() * 0.5 + 3.0
        other = make_data(changed)
        np.testing.assert_array_equal(self.data.filled[:t0], other.filled[:t0])
        # origin windows ending before t0 are identical (scaler is unchanged: t0 is after train end)
        self.assertEqual(self.data.scaler, other.scaler)
        a, b = self.data.dataset("test"), other.dataset("test")
        keep = np.flatnonzero(a.origins < t0)
        self.assertGreater(len(keep), 5)
        np.testing.assert_array_equal(a.get_batch(keep)["x"].numpy(), b.get_batch(keep)["x"].numpy())

    def test_leading_gap_uses_training_median(self):
        sensor0 = self.frame.iloc[:, 0].to_numpy()
        valid_train = sensor0[: self.train_end + 1]
        valid_train = valid_train[np.isfinite(valid_train) & (valid_train > 0)]
        np.testing.assert_allclose(self.data.filled[:5, 0], np.median(valid_train), rtol=1e-6)
        self.assertEqual(float(self.data.features[0, 0, 1]), 1.0)  # missing flag set

    def test_targets_never_filled(self):
        raw = self.frame.to_numpy()
        invalid = ~(np.isfinite(raw) & (raw > 0))
        self.assertTrue(invalid.any())
        self.assertFalse(self.data.target_valid.numpy()[invalid].any())
        self.assertTrue((self.data.targets.numpy()[invalid] == 0).all())  # never imputed
        self.assertTrue((self.data.filled[invalid] > 0).all())  # inputs are filled
        ok = ~invalid
        np.testing.assert_allclose(self.data.targets.numpy()[ok], raw[ok].astype(np.float32))

    def test_batch_alignment_history_and_targets(self):
        ds = self.data.dataset("val")
        for pos in (0, 7, len(ds) - 1):
            t = int(ds.origins[pos])
            batch = ds.get_batch([pos])
            np.testing.assert_array_equal(batch["x"][0].numpy(), self.data.features[t - 11 : t + 1].numpy())
            for k, step in enumerate((1, 3, 6, 12)):
                np.testing.assert_array_equal(batch["y"][0, k].numpy(), self.data.targets[t + step].numpy())
        # all four targets stay inside the validation partition
        part = self.data.partitions["validation"]
        self.assertLessEqual(int(ds.origins.max()) + 12, part.end_idx)
        self.assertGreaterEqual(int(ds.origins.min()) + 1, part.start_idx)

    def test_future_target_changes_do_not_change_inputs(self):
        ds_a = self.data.dataset("test")
        t = int(ds_a.origins[10])
        changed = self.frame.copy()
        changed.iloc[t + 1 : t + 13, :] = 99.0
        other = make_data(changed)
        batch_a, batch_b = ds_a.get_batch([10]), other.dataset("test").get_batch([10])
        np.testing.assert_array_equal(batch_a["x"].numpy(), batch_b["x"].numpy())
        self.assertFalse(np.array_equal(batch_a["y"].numpy(), batch_b["y"].numpy()))

    def test_time_channels_match_phase1_features(self):
        ref = compute_temporal_features(self.data.timestamps)
        feats = self.data.features[:, 0, 2:].numpy()
        cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]
        np.testing.assert_allclose(feats, ref[cols].to_numpy(dtype=np.float32), atol=1e-6)

    def test_peak_masks_follow_phase1_rule(self):
        masks = self.data.dataset("test").period_masks()
        ts = self.data.dataset("test").timestamps
        weekday = ts.day_of_week < 5
        morning = weekday & (ts.hour >= 7) & (ts.hour < 10)
        evening = weekday & (ts.hour >= 16) & (ts.hour < 19)
        np.testing.assert_array_equal(masks["peak"], morning | evening)
        np.testing.assert_array_equal(masks["off_peak"], ~(morning | evening))

    def test_subset_selects_spread_origins(self):
        small = Phase2Data(self.frame, make_contract(self.frame.index), device="cpu", subset=0.1)
        full = self.data.dataset("train").origins
        sub = small.dataset("train").origins
        self.assertLess(len(sub), len(full) * 0.2)
        self.assertTrue(set(sub).issubset(set(full)))
        self.assertGreater(sub.max() - sub.min(), (full.max() - full.min()) * 0.9)

    def test_shuffle_order_depends_only_on_seed_and_epoch(self):
        ds = self.data.dataset("train")
        a = [b["pos"].tolist() for b in ds.batches(64, shuffle=True, seed=5, epoch=3)]
        b = [b["pos"].tolist() for b in ds.batches(64, shuffle=True, seed=5, epoch=3)]
        c = [b["pos"].tolist() for b in ds.batches(64, shuffle=True, seed=5, epoch=4)]
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


# ----------------------------------------------------------------------------- (c) graph
class GraphVariantTest(unittest.TestCase):
    def setUp(self):
        # directed 3-node example: 0 -> 1 -> 2 (weights), diagonal 1
        self.adj = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.4], [0.0, 0.0, 1.0]], dtype=np.float32)
        rng = np.random.default_rng(3)
        n = 30
        raw = (rng.random((n, n)) < 0.15) * rng.uniform(0.2, 1.0, (n, n))
        np.fill_diagonal(raw, 1.0)
        self.big = raw.astype(np.float32)

    def test_basic_variants(self):
        np.testing.assert_array_equal(make_variant(self.adj, "given"), self.adj)
        np.testing.assert_array_equal(make_variant(self.adj, "given_T"), self.adj.T)
        both = make_variant(self.adj, "bidirectional")
        np.testing.assert_array_equal(both, both.T)
        np.testing.assert_array_equal(both, np.maximum(self.adj, self.adj.T))
        np.testing.assert_array_equal(make_variant(self.adj, "identity"), np.eye(3))
        with self.assertRaises(ValueError):
            make_variant(self.adj, "nonsense")

    def test_direction_convention(self):
        """A[i, j] > 0 means j is reachable from i: row-normalised A @ h lets i read j."""
        weights = self.adj / self.adj.sum(axis=1, keepdims=True)
        one_hot_node1 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        out = weights @ one_hot_node1
        self.assertGreater(out[0], 0)  # node 0 reads node 1 (0 -> 1 exists)
        self.assertEqual(out[2], 0)  # node 2 cannot read node 1 (no 1 -> 2 entry in this row)
        reverse = (self.adj.T / self.adj.T.sum(axis=1, keepdims=True)) @ one_hot_node1
        self.assertGreater(reverse[2], 0)  # transposed: node 2 reads node 1 (upstream)

    def test_shuffled_keeps_degrees_but_moves_neighbours(self):
        base = make_variant(self.big, "bidirectional")
        shuffled = make_variant(self.big, "shuffled", seed=1)
        self.assertEqual(shuffled.shape, base.shape)
        np.testing.assert_array_equal(np.diag(shuffled), np.ones(len(base)))
        self.assertEqual(sorted((base > 0).sum(1).tolist()), sorted((shuffled > 0).sum(1).tolist()))
        self.assertEqual(sorted(base[base > 0].tolist()), sorted(shuffled[shuffled > 0].tolist()))
        self.assertFalse(np.array_equal(base, shuffled))
        np.testing.assert_array_equal(shuffled, make_variant(self.big, "shuffled", seed=1))  # deterministic
        self.assertFalse(np.array_equal(shuffled, make_variant(self.big, "shuffled", seed=2)))
        sigma = derangement(len(base), 1)
        self.assertTrue((sigma != np.arange(len(base))).all())
        self.assertEqual(sorted(sigma.tolist()), list(range(len(base))))

    def test_random_matched_keeps_edge_count(self):
        for base_name in ("bidirectional", "given"):
            base = make_variant(self.big, base_name)
            rand = make_variant(self.big, "random_matched", seed=4, base=base_name)
            off = ~np.eye(len(base), dtype=bool)
            self.assertEqual(int((base > 0)[off].sum()), int((rand > 0)[off].sum()))
            np.testing.assert_array_equal(np.diag(rand), np.ones(len(base)))
            if base_name == "bidirectional":
                np.testing.assert_array_equal(rand, rand.T)
            self.assertFalse(np.array_equal((base > 0), (rand > 0)))

    @NEED_DATA
    def test_real_graph(self):
        adj = load_adjacency()
        self.assertEqual(adj.shape, (207, 207))
        self.assertEqual(adj.dtype, np.float32)
        self.assertEqual(int((adj > 0).sum()), 1722)
        np.testing.assert_array_equal(np.diag(adj), np.ones(207))
        self.assertFalse(np.allclose(adj, adj.T))  # directed
        for variant in VARIANTS:
            g = load_graph(variant)
            self.assertEqual(tuple(g.shape), (207, 207))
            self.assertEqual(g.dtype, torch.float32)
        self.assertTrue(torch.equal(load_graph("bidirectional"), load_graph("bidirectional").T))

    @NEED_DATA
    def test_regenerated_kernel_matches_pickle_if_distances_present(self):
        result = verify_adjacency()
        if result["status"] == "skipped":
            self.skipTest(result["reason"])
        self.assertTrue(result["same_support"])
        self.assertLess(result["max_abs_diff"], 1e-5)


# ----------------------------------------------------------------------------- registry
class RegistryTest(unittest.TestCase):
    def test_builtin_models_registered(self):
        names = list_models()
        for name in ("last_value", "dummy_gru"):
            self.assertIn(name, names)

    def test_unknown_model_lists_choices(self):
        with self.assertRaises(KeyError) as ctx:
            get_builder("does_not_exist")
        self.assertIn("last_value", str(ctx.exception))

    def test_duplicate_name_rejected(self):
        with self.assertRaises(ValueError):

            @register_model("last_value")
            def other(cfg):  # pragma: no cover
                return None


# ----------------------------------------------------------------------------- (d) trainer
class TrainerTest(unittest.TestCase):
    def setUp(self):
        self.data = make_data()
        self.datasets = self.data.datasets()
        self.adj = torch.eye(self.data.num_nodes)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, seed=7, **params):
        set_seed(seed)
        cfg = {"num_nodes": self.data.num_nodes, "in_channels": 7, "history": 12, "num_horizons": 4}
        cfg.update({"hidden": 8, "dropout": 0.2, **params})
        return build_model("dummy_gru", cfg)

    TRAIN = {"lr": 0.01, "batch_size": 64, "epochs": 4, "patience": 10, "scheduler": "plateau"}

    def test_resume_equals_uninterrupted_run(self):
        full_dir, part_dir = self.root / "full", self.root / "part"
        full_model = self.build()
        full = fit(full_model, self.datasets, self.adj, self.TRAIN, "cpu", full_dir, seed=7, log=lambda s: None)

        interrupted = self.build()
        with self.assertRaises(InterruptedForTest):
            fit(interrupted, self.datasets, self.adj, self.TRAIN, "cpu", part_dir, seed=7, interrupt_after=2, log=lambda s: None)
        self.assertFalse((part_dir / "done.json").exists())
        self.assertTrue((part_dir / "last.pt").is_file())
        resumed_model = self.build()  # fresh process: new model object, weights come from last.pt
        resumed = fit(resumed_model, self.datasets, self.adj, self.TRAIN, "cpu", part_dir, seed=7, log=lambda s: None)

        self.assertEqual(resumed["epochs_run"], full["epochs_run"])
        self.assertEqual(resumed["best_epoch"], full["best_epoch"])
        self.assertAlmostEqual(resumed["best_val_mae"], full["best_val_mae"], places=10)
        for (name, a), (_, b) in zip(full_model.state_dict().items(), resumed_model.state_dict().items()):
            self.assertTrue(torch.equal(a, b), f"parameter {name} differs after resume")
        hist_a = [json.loads(l)["val_mae_mean"] for l in (full_dir / "epochs.jsonl").read_text().splitlines()]
        hist_b = [json.loads(l)["val_mae_mean"] for l in (part_dir / "epochs.jsonl").read_text().splitlines()]
        self.assertEqual(hist_a, hist_b)

    def test_finished_run_is_not_retrained(self):
        model = self.build()
        run_dir = self.root / "run"
        first = fit(model, self.datasets, self.adj, self.TRAIN, "cpu", run_dir, seed=7, log=lambda s: None)
        messages = []
        second = fit(self.build(), self.datasets, self.adj, self.TRAIN, "cpu", run_dir, seed=7, log=messages.append)
        self.assertEqual(first, second)
        self.assertTrue(any("already finished" in m for m in messages))

    def test_early_stopping(self):
        cfg = {**self.TRAIN, "lr": 0.0, "patience": 2, "epochs": 10}  # no learning -> no improvement after epoch 1
        result = fit(self.build(), self.datasets, self.adj, cfg, "cpu", self.root / "es", seed=7, log=lambda s: None)
        self.assertEqual(result["stop_reason"], "early_stop")
        self.assertEqual(result["epochs_run"], 3)
        self.assertEqual(result["best_epoch"], 1)

    def test_max_epochs_and_time_cap(self):
        result = fit(self.build(), self.datasets, self.adj, {**self.TRAIN, "epochs": 2}, "cpu", self.root / "me", seed=7, log=lambda s: None)
        self.assertEqual((result["stop_reason"], result["epochs_run"]), ("max_epochs", 2))
        capped = fit(
            self.build(), self.datasets, self.adj, {**self.TRAIN, "epochs": 50, "max_minutes": 1e-6}, "cpu",
            self.root / "cap", seed=7, log=lambda s: None,
        )
        self.assertEqual(capped["stop_reason"], "time_cap")
        self.assertGreaterEqual(capped["epochs_run"], 1)
        self.assertTrue((self.root / "cap" / "best.pt").is_file())

    def test_time_cap_is_cumulative_over_resume(self):
        run_dir = self.root / "cum"
        cfg = {**self.TRAIN, "epochs": 50, "max_minutes": 1e-6, "patience": 50}
        with self.assertRaises(InterruptedForTest):  # interrupt_after only fires if not already stopping
            fit(self.build(), self.datasets, self.adj, {**cfg, "max_minutes": 10}, "cpu", run_dir, seed=7, interrupt_after=1, log=lambda s: None)
        result = fit(self.build(), self.datasets, self.adj, cfg, "cpu", run_dir, seed=7, log=lambda s: None)
        self.assertEqual(result["stop_reason"], "time_cap")
        self.assertEqual(result["epochs_run"], 1)  # elapsed time was carried over from the checkpoint

    def test_untrainable_model_is_only_evaluated(self):
        model = build_model("last_value", {})
        result = fit(model, self.datasets, self.adj, self.TRAIN, "cpu", self.root / "lv", seed=0, log=lambda s: None)
        self.assertEqual(result["stop_reason"], "no_trainable_params")
        self.assertEqual(result["params"], 0)

    def test_wrong_shape_and_nonfinite_outputs_are_reported(self):
        class WrongShape(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x, adj):
                return x[:, -1, :, 0].unsqueeze(1).expand(-1, 3, -1) + self.w

        with self.assertRaisesRegex(ValueError, r"contract is \[B, 4, N\]"):
            fit(WrongShape(), self.datasets, self.adj, self.TRAIN, "cpu", self.root / "ws", seed=0, log=lambda s: None)

        class NanModel(torch.nn.Module):
            def forward(self, x, adj):
                return torch.full((x.shape[0], 4, x.shape[2]), float("nan"))

        with self.assertRaises(FloatingPointError):
            predict(NanModel(), self.datasets["val"], self.adj, "cpu")

    def test_unknown_train_key_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown train config"):
            TrainConfig.from_dict({"lr": 0.1, "learning_rate": 0.2})

    def test_predict_matches_dataset_order(self):
        model = self.build()
        ds = self.datasets["val"]
        pred = predict(model, ds, self.adj, "cpu", batch_size=17)
        pred2 = predict(model, ds, self.adj, "cpu", batch_size=64)
        self.assertEqual(pred.shape, (len(ds), 4, self.data.num_nodes))
        np.testing.assert_allclose(pred, pred2, atol=1e-4)


@unittest.skipUnless(
    getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available(),
    "MPS not available on this machine",
)
class MpsDeviceTest(unittest.TestCase):
    """MPS has no float64: the whole fit/evaluate/predict/export path must avoid it."""

    def test_fit_predict_on_mps(self):
        data = Phase2Data(make_frame(), make_contract(make_frame().index), device="mps")
        datasets = data.datasets()
        set_seed(0)
        model = build_model("dummy_gru", {"num_nodes": data.num_nodes, "in_channels": 7, "hidden": 8, "mix": True})
        with tempfile.TemporaryDirectory() as tmp:
            result = fit(model, datasets, torch.eye(data.num_nodes), {"epochs": 2, "batch_size": 64}, "mps", tmp, seed=0, log=lambda s: None)
            self.assertEqual(result["epochs_run"], 2)
            pred = predict(model, datasets["test"], torch.eye(data.num_nodes), "mps")
            self.assertTrue(np.isfinite(pred).all())
            self.assertEqual(pred.shape, (len(datasets["test"]), 4, data.num_nodes))


# ----------------------------------------------------------------------------- attention export
@NEED_DATA
class AttentionExportTest(unittest.TestCase):
    def test_attention_origins_are_the_planned_twelve(self):
        ds = real_data().dataset("test")
        positions = attention_origin_positions(ds)
        self.assertEqual(len(positions), 12)
        ts = ds.timestamps[positions]
        self.assertTrue(ts.is_monotonic_increasing)
        self.assertEqual(sorted(set(ts.normalize())), [pd.Timestamp(f"2012-06-0{d}") for d in (4, 5, 6, 7)])
        clocks = sorted({(t.hour, t.minute) for t in ts})
        self.assertEqual(clocks, [(8, 0), (13, 0), (17, 30)])

    def test_export_attention_writes_expected_array(self):
        class FakeAttention(torch.nn.Module):
            def forward(self, x, adj):  # pragma: no cover
                return x[:, -1, :, 0].unsqueeze(1).expand(-1, 4, -1)

            def attention_maps(self, x, adj):
                b, n = x.shape[0], x.shape[2]
                return [torch.softmax(torch.randn(b, 3, n, n), dim=-1) for _ in range(2)]

        ds = real_data().dataset("test")
        with tempfile.TemporaryDirectory() as tmp:
            path = export_attention(FakeAttention(), ds, torch.eye(207), "cpu", "fake", 42, tmp)
            z = np.load(path)
            self.assertEqual(z["attn"].shape, (12, 2, 3, 207, 207))
            self.assertEqual(z["attn"].dtype, np.float16)
            self.assertEqual(z["origin_ns"].shape, (12,))
        self.assertIsNone(export_attention(torch.nn.Linear(1, 1), ds, torch.eye(207), "cpu", "x", 1, "/tmp/unused"))


# ----------------------------------------------------------------------------- (e) CLI
TINY_RUNS = {
    "name": "tiny",
    "train": {"lr": 0.01, "batch_size": 64, "epochs": 2, "patience": 2, "max_minutes": 2},
    "groups": [
        {
            "search": {
                "model": "dummy_gru",
                "graph": "bidirectional",
                "candidates": [{"hidden": 4}, {"hidden": 8, "mix": True}],
            },
            "final": [
                {"model_id": "tiny", "model": "dummy_gru", "graph": "bidirectional", "params": "from_search", "seeds": [1, 2]},
                {"model_id": "tiny_identity", "model": "dummy_gru", "graph": "identity", "params": "from_search", "seeds": [1]},
            ],
        }
    ],
}


@NEED_DATA
class CliEndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.runs = cls.dir / "runs_tiny.yaml"
        cls.runs.write_text(yaml.safe_dump(TINY_RUNS))
        cls.out = cls.dir / "out"
        cls.root = cls.out / "subset_0.02"
        code = cli.main(["all", "--runs", str(cls.runs), "--device", "cpu", "--subset", "0.02", "--out", str(cls.out)])
        assert code == 0

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_search_and_pick_outputs(self):
        table = pd.read_csv(self.root / "search/tiny.csv")
        self.assertEqual(len(table), 2)
        self.assertTrue(set(["val_mae_5min", "val_mae_60min", "val_mae_mean", "stop_reason", "param_count"]).issubset(table.columns))
        picked = yaml.safe_load((self.root / "final/tiny.yaml").read_text())
        best = int(table.sort_values(["val_mae_mean", "candidate"]).iloc[0].candidate)
        self.assertEqual(picked["picked"][0]["candidate"], best)
        self.assertEqual([r["model_id"] for r in picked["runs"]], ["tiny", "tiny_identity"])
        self.assertEqual(picked["runs"][0]["params"], TINY_RUNS["groups"][0]["search"]["candidates"][best])
        self.assertEqual(picked["runs"][1]["params"], picked["runs"][0]["params"])  # controls inherit tuned params
        self.assertNotEqual(picked["runs"][0]["params"], "from_search")

    def test_prediction_files_follow_plan_schema(self):
        sensors = pd.read_csv(ROOT / "data/metadata/sensors.csv", dtype={"sensor_id": str}).sensor_id.tolist()
        data = load_data({"device": "cpu", "subset": 0.02})
        for model_id, seed in [("tiny", 1), ("tiny", 2), ("tiny_identity", 1)]:
            for split, key in (("val", "val"), ("test", "test")):
                z = np.load(self.root / f"predictions/{model_id}/seed{seed}_{split}.npz")
                ds = data.dataset(key)
                self.assertEqual(set(z.files), {"pred", "origin_ns", "sensor_ids"})
                self.assertEqual(z["pred"].shape, (len(ds), 4, 207))
                self.assertEqual(z["pred"].dtype, np.float32)
                self.assertTrue(np.isfinite(z["pred"]).all())
                self.assertEqual(z["sensor_ids"].tolist(), sensors)
                self.assertTrue(np.all(np.diff(z["origin_ns"]) > 0))
                np.testing.assert_array_equal(z["origin_ns"], ds.origin_ns)
            meta = json.loads((self.root / f"predictions/{model_id}/seed{seed}_meta.json").read_text())
            for key in ("git_commit", "config_hash", "seed", "device", "torch_version", "params", "epochs_run",
                        "best_epoch", "best_val_mae", "train_seconds", "stop_reason", "subset"):
                self.assertIn(key, meta)
            self.assertEqual(meta["seed"], seed)
            self.assertEqual(meta["device"], "cpu")
        self.assertTrue((self.root / "best/tiny_seed1.pt").is_file())

    def test_no_test_predictions_during_search(self):
        for run_dir in (self.root / "search/tiny").iterdir():
            self.assertFalse(list(run_dir.glob("*.npz")))
        self.assertFalse((self.root / "predictions/dummy_gru").exists())

    def test_rerun_skips_and_only_retrains_requested_seed(self):
        target = self.root / "predictions/tiny/seed2_test.npz"
        before_other = (self.root / "predictions/tiny/seed1_test.npz").stat().st_mtime_ns
        target.unlink()
        (self.root / "predictions/tiny/seed2_meta.json").unlink()
        code = cli.main(["all", "--runs", str(self.runs), "--device", "cpu", "--subset", "0.02",
                         "--out", str(self.out), "--only", "tiny:2"])
        self.assertEqual(code, 0)
        self.assertTrue(target.is_file())
        self.assertTrue((self.root / "predictions/tiny/seed2_meta.json").is_file())
        self.assertEqual((self.root / "predictions/tiny/seed1_test.npz").stat().st_mtime_ns, before_other)

    def test_only_requires_finished_pick(self):
        with tempfile.TemporaryDirectory() as fresh:
            with self.assertRaises(SystemExit):
                cli.main(["all", "--runs", str(self.runs), "--device", "cpu", "--subset", "0.02",
                          "--out", fresh, "--only", "tiny"])

    def test_smoke_command_passes_on_run_file(self):
        self.assertEqual(cli.main(["smoke", "--runs", str(self.runs), "--device", "cpu"]), 0)

    def test_example_run_file_validates(self):
        runs = cli.load_runs(ROOT / "configs/phase2/runs_example.yaml")
        self.assertEqual(runs["kind"], "spec")
        self.assertEqual(cli.parse_only(["a:1,b", "c"]), [("a", 1), ("b", None), ("c", None)])


class RunFileValidationTest(unittest.TestCase):
    def write(self, obj) -> Path:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        tmp.write(yaml.safe_dump(obj))
        tmp.close()
        self.addCleanup(lambda: Path(tmp.name).unlink())
        return Path(tmp.name)

    def test_rejects_bad_files(self):
        good = {"name": "x", "train": {}, "groups": [{"final": [
            {"model_id": "a", "model": "last_value", "graph": "identity", "params": {}, "seeds": [1]}]}]}
        cli.load_runs(self.write(good))
        bad_graph = json.loads(json.dumps(good))
        bad_graph["groups"][0]["final"][0]["graph"] = "nope"
        with self.assertRaisesRegex(ValueError, "graph variant"):
            cli.load_runs(self.write(bad_graph))
        no_search = json.loads(json.dumps(good))
        no_search["groups"][0]["final"][0]["params"] = "from_search"
        with self.assertRaisesRegex(ValueError, "from_search"):
            cli.load_runs(self.write(no_search))
        dup = json.loads(json.dumps(good))
        dup["groups"].append(json.loads(json.dumps(good["groups"][0])))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            cli.load_runs(self.write(dup))
        bad_train = json.loads(json.dumps(good))
        bad_train["train"] = {"learning_rate": 1}
        with self.assertRaisesRegex(ValueError, "unknown train config"):
            cli.load_runs(self.write(bad_train))
        with self.assertRaises(KeyError):
            bad_model = json.loads(json.dumps(good))
            bad_model["groups"][0]["final"][0]["model"] = "ghost"
            cli.load_runs(self.write(bad_model))


if __name__ == "__main__":
    unittest.main()
