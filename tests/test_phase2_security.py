"""Security regression tests for Phase 2: unsafe pickles and checkpoints are refused,
and the Colab clone cell never puts the GitHub token in a URL."""

from __future__ import annotations

import os
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
import torch

from src.phase2.common import ROOT, load_graph_pickle
from src.phase2.train import _load_checkpoint, _numpy_rng_from_plain, _numpy_rng_to_plain


class _RunsCode:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return (Path.touch, (self.marker,))


class GraphPickleTest(unittest.TestCase):
    def test_plain_graph_pickle_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adj.pkl"
            adj = np.eye(2, dtype=np.float32)
            with path.open("wb") as f:
                pickle.dump((["a", "b"], {"a": 0, "b": 1}, adj), f, protocol=2)
            ids, id_to_index, loaded = load_graph_pickle(path)
            self.assertEqual(ids, ["a", "b"])
            self.assertEqual(id_to_index, {"a": 0, "b": 1})
            np.testing.assert_array_equal(loaded, adj)

    def test_pickle_with_code_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "ran"
            path = Path(tmp) / "evil.pkl"
            with path.open("wb") as f:
                pickle.dump((["a"], {"a": 0}, _RunsCode(marker)), f)
            with self.assertRaises(pickle.UnpicklingError):
                load_graph_pickle(path)
            self.assertFalse(marker.exists())


class CheckpointTest(unittest.TestCase):
    def test_numpy_rng_round_trip(self):
        np.random.seed(7)
        np.random.rand(3)
        saved = _numpy_rng_to_plain(np.random.get_state())
        expected = np.random.rand(5)
        np.random.set_state(_numpy_rng_from_plain(saved))
        np.testing.assert_array_equal(np.random.rand(5), expected)

    def test_checkpoint_with_code_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "ran"
            path = Path(tmp) / "evil.pt"
            torch.save({"model": _RunsCode(marker)}, path)
            old = os.environ.pop("PHASE2_TRUST_CHECKPOINTS", None)
            try:
                with self.assertRaises(RuntimeError):
                    _load_checkpoint(path)
            finally:
                if old is not None:
                    os.environ["PHASE2_TRUST_CHECKPOINTS"] = old
            self.assertFalse(marker.exists())


class ColabCloneCellTest(unittest.TestCase):
    def test_token_never_goes_into_a_url(self):
        for rel in ("scripts/phase2_build_colab_notebook.py", "notebooks/phase2_colab_train.ipynb"):
            text = (ROOT / rel).read_text()
            self.assertNotIn("{TOKEN}@", text, rel)
            self.assertIn("GIT_CONFIG_VALUE_0", text, rel)
            self.assertIn("safe_extract", text, rel)
            self.assertNotIn("z.extractall(WORK)", text, rel)


if __name__ == "__main__":
    unittest.main()
