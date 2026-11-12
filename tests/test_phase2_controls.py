"""Unit tests for non-graph deep temporal control model (gru_local).

Owner: Member 3 (phase2/m3-dcrnn-controls). Contract: reports/phase2_plan.md section 8.3.
"""

from __future__ import annotations

import unittest

import torch
from torch import nn

from src.phase2.models.controls import GRULocal, build
from src.phase2.registry import get_builder


class GRULocalTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.b = 4
        self.t = 12
        self.n = 10
        self.c = 7
        self.num_horizons = 4
        self.x = torch.randn(self.b, self.t, self.n, self.c)

    def test_registered_and_build(self) -> None:
        builder = get_builder("gru_local")
        self.assertIsNotNone(builder)
        cfg = {"num_nodes": self.n, "hidden": 32, "layers": 1}
        model = builder(cfg)
        self.assertIsInstance(model, GRULocal)
        self.assertFalse(model.needs_graph)

    def test_output_shape_and_finiteness(self) -> None:
        for layers in [1, 2]:
            with self.subTest(layers=layers):
                model = GRULocal(num_nodes=self.n, hidden=32, layers=layers)
                model.eval()
                with torch.no_grad():
                    out = model(self.x)
                self.assertEqual(out.shape, (self.b, self.num_horizons, self.n))
                self.assertTrue(torch.isfinite(out).all().item())

    def test_sensor_independence(self) -> None:
        """Modifying sensor j's input MUST NOT affect sensor i's output (i != j).

        This is the defining property of gru_local: zero spatial mixing.
        """
        model = GRULocal(num_nodes=self.n, hidden=32, layers=2, dropout=0.0)
        model.eval()

        with torch.no_grad():
            out_base = model(self.x)

        # Perturb sensor 0 only
        x_perturbed = self.x.clone()
        x_perturbed[:, :, 0, :] += 10.0

        with torch.no_grad():
            out_perturbed = model(x_perturbed)

        # Sensor 0 output must change
        self.assertFalse(
            torch.allclose(out_base[:, :, 0], out_perturbed[:, :, 0], atol=1e-4)
        )
        # All other sensors (1..N-1) must be strictly identical
        self.assertTrue(
            torch.allclose(out_base[:, :, 1:], out_perturbed[:, :, 1:], atol=1e-5),
            "gru_local leaked information across sensors! Independent sensor property violated.",
        )

    def test_node_permutation_equivariance(self) -> None:
        """Permuting sensor order in x permutes the output identically."""
        model = GRULocal(num_nodes=self.n, hidden=32, layers=1, dropout=0.0)
        model.eval()

        perm = torch.randperm(self.n)
        x_perm = self.x[:, :, perm, :]

        with torch.no_grad():
            out_orig = model(self.x)
            out_perm = model(x_perm)

        out_orig_permuted = out_orig[:, :, perm]
        self.assertTrue(
            torch.allclose(out_orig_permuted, out_perm, atol=1e-5),
            "gru_local is not node-permutation equivariant!",
        )

    def test_overfit_on_small_batch(self) -> None:
        """Loss should drop strongly on 64 fixed synthetic samples."""
        b = 64
        x = torch.randn(b, 12, 10, 7)
        y = torch.randn(b, 4, 10)

        model = GRULocal(num_nodes=10, hidden=32, layers=1, dropout=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = nn.MSELoss()

        model.train()
        initial_loss = 0.0
        final_loss = 0.0
        for step in range(80):
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            if step == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

        self.assertLess(
            final_loss,
            initial_loss * 0.4,
            f"Overfit test failed: initial {initial_loss:.4f}, final {final_loss:.4f}",
        )

    def test_parameter_counts(self) -> None:
        cfgs = [
            {"hidden": 32, "layers": 1, "dropout": 0.0},
            {"hidden": 64, "layers": 1, "dropout": 0.0},
            {"hidden": 32, "layers": 2, "dropout": 0.1},
            {"hidden": 64, "layers": 2, "dropout": 0.2},
        ]
        for cfg in cfgs:
            model = GRULocal(num_nodes=207, **cfg)
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.assertGreater(param_count, 500)
            print(f"[gru_local param count] cfg={cfg} -> {param_count:,} parameters")


if __name__ == "__main__":
    unittest.main()

