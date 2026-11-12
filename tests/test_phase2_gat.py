"""Unit tests for Graph Attention Network (GAT).

Owner: Member 2 (phase2/m2-gat). Contract: reports/phase2_plan.md section 8.2.
"""

from __future__ import annotations

import unittest

import torch
from torch import nn

from src.phase2.models.gat import GATModel, build
from src.phase2.registry import get_builder


class GATModelTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.b = 4
        self.t = 12
        self.n = 10
        self.c = 7
        self.num_horizons = 4
        # Synthetic random feature tensor
        self.x = torch.randn(self.b, self.t, self.n, self.c)
        # Random directed graph with ~30% connectivity + self loops
        prob = torch.rand(self.n, self.n)
        self.adj = (prob > 0.7).float()
        self.adj.fill_diagonal_(1.0)
        self.eye_adj = torch.eye(self.n)

    def test_registered_and_build(self) -> None:
        builder = get_builder("gat")
        self.assertIsNotNone(builder)
        cfg = {"num_nodes": self.n, "hidden": 32, "heads": 2, "layers": 1}
        model = builder(cfg)
        self.assertIsInstance(model, GATModel)
        self.assertTrue(model.needs_graph)

    def test_output_shape_and_finiteness(self) -> None:
        for encoder in ["gru", "conv"]:
            with self.subTest(encoder=encoder):
                model = GATModel(
                    num_nodes=self.n,
                    hidden=32,
                    heads=2,
                    layers=2,
                    temporal_encoder=encoder,
                )
                model.eval()
                with torch.no_grad():
                    out = model(self.x, self.adj)
                self.assertEqual(out.shape, (self.b, self.num_horizons, self.n))
                self.assertTrue(torch.isfinite(out).all().item())

    def test_attention_maps_properties(self) -> None:
        heads = 4
        layers = 2
        model = GATModel(
            num_nodes=self.n,
            hidden=32,
            heads=heads,
            layers=layers,
            dropout=0.0,
        )
        model.eval()
        with torch.no_grad():
            maps = model.attention_maps(self.x, self.adj)

        self.assertEqual(len(maps), layers)
        for layer_idx, attn in enumerate(maps):
            self.assertEqual(attn.shape, (self.b, heads, self.n, self.n))
            self.assertTrue(torch.isfinite(attn).all().item())

            # Row sum must equal 1.0 everywhere
            row_sums = attn.sum(dim=-1)
            self.assertTrue(
                torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5),
                f"Layer {layer_idx} row sums do not equal 1",
            )

            # Masked non-edges must be strictly 0.0
            mask = (self.adj > 0) | torch.eye(self.n, dtype=torch.bool)
            non_edges = ~mask.view(1, 1, self.n, self.n)
            masked_entries = attn[non_edges.expand_as(attn)]
            self.assertTrue(
                torch.allclose(masked_entries, torch.zeros_like(masked_entries), atol=1e-7),
                f"Layer {layer_idx} has non-zero attention on disconnected edges",
            )

    def test_node_permutation_equivariance(self) -> None:
        """Permuting the node order in x and adj should permute the output identically."""
        model = GATModel(
            num_nodes=self.n,
            hidden=32,
            heads=2,
            layers=1,
            dropout=0.0,
        )
        model.eval()

        perm = torch.randperm(self.n)
        x_perm = self.x[:, :, perm, :]
        adj_perm = self.adj[perm][:, perm]

        with torch.no_grad():
            out_orig = model(self.x, self.adj)
            out_perm = model(x_perm, adj_perm)

        out_orig_permuted = out_orig[:, :, perm]
        self.assertTrue(
            torch.allclose(out_orig_permuted, out_perm, atol=1e-5),
            "GAT is not node-permutation equivariant!",
        )

    def test_neighbour_independence_with_identity_graph(self) -> None:
        """With an identity graph (self-loops only), modifying sensor j's input

        must NOT affect sensor i's output (i != j).
        """
        model = GATModel(
            num_nodes=self.n,
            hidden=32,
            heads=2,
            layers=1,
            dropout=0.0,
        )
        model.eval()

        # Output with original input and identity graph
        with torch.no_grad():
            out_base = model(self.x, self.eye_adj)

        # Perturb sensor 0 only
        x_perturbed = self.x.clone()
        x_perturbed[:, :, 0, :] += 5.0

        with torch.no_grad():
            out_perturbed = model(x_perturbed, self.eye_adj)

        # Sensor 0 should change
        self.assertFalse(
            torch.allclose(out_base[:, :, 0], out_perturbed[:, :, 0], atol=1e-4)
        )
        # All other sensors (1..N-1) must remain identical
        self.assertTrue(
            torch.allclose(out_base[:, :, 1:], out_perturbed[:, :, 1:], atol=1e-5),
            "Modifying sensor 0 affected other sensors under an identity graph!",
        )

    def test_neighbour_influence_with_connected_graph(self) -> None:
        """With a connected graph, modifying sensor 0's input SHOULD affect its neighbours."""
        # Create a graph where node 0 is connected to node 1
        adj = torch.eye(self.n)
        adj[0, 1] = 1.0
        adj[1, 0] = 1.0

        model = GATModel(
            num_nodes=self.n,
            hidden=32,
            heads=2,
            layers=1,
            dropout=0.0,
        )
        model.eval()

        with torch.no_grad():
            out_base = model(self.x, adj)

        x_perturbed = self.x.clone()
        x_perturbed[:, :, 0, :] += 5.0

        with torch.no_grad():
            out_perturbed = model(x_perturbed, adj)

        # Node 1 should be affected through edge (0, 1)
        diff = (out_base[:, :, 1] - out_perturbed[:, :, 1]).abs().max()
        self.assertGreater(
            diff.item(), 1e-4, "Connected neighbour node 1 was not affected by node 0"
        )

    def test_overfit_on_small_synthetic_batch(self) -> None:
        """Training loss should drop strongly on 64 fixed random samples."""
        b = 64
        x = torch.randn(b, 12, 10, 7)
        y = torch.randn(b, 4, 10)
        adj = torch.eye(10)

        model = GATModel(num_nodes=10, hidden=32, heads=2, layers=1, dropout=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = nn.MSELoss()

        model.train()
        initial_loss = 0.0
        final_loss = 0.0
        for step in range(100):
            optimizer.zero_grad()
            pred = model(x, adj)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            if step == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

        self.assertLess(
            final_loss,
            initial_loss * 0.4,
            f"Overfit test failed: initial loss {initial_loss:.4f}, final loss {final_loss:.4f}",
        )

    def test_parameter_counts(self) -> None:
        """Parameter counts must match the fact sheet (reports/phase2/fact_sheet_gat.md).

        Exact values act as regression guards: if model internals change and shift
        the count, this test immediately flags the discrepancy.
        """
        # (hidden, heads, layers, temporal_encoder) -> expected param count
        expected_counts = {
            (32, 2, 1, "gru"): 6276,
            (64, 2, 1, "gru"): 22788,
            (32, 4, 2, "gru"): 7428,
            (32, 2, 1, "conv"): 6148,
        }
        for (hidden, heads, layers, enc), expected in expected_counts.items():
            cfg = {"hidden": hidden, "heads": heads, "layers": layers, "temporal_encoder": enc}
            model = GATModel(num_nodes=207, **cfg)
            actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[GAT param count] cfg={cfg} -> {actual:,} parameters (expected {expected:,})")
            self.assertEqual(
                actual,
                expected,
                f"Parameter count mismatch for cfg={cfg}: got {actual}, expected {expected}",
            )

    def test_attention_stats_synthetic_computation(self) -> None:
        """Verify attention stats computation runs cleanly on synthetic attention maps."""
        import numpy as np
        import pandas as pd
        from scripts.phase2_gat_attention_stats import compute_attention_stats

        b, layers, heads, n = 4, 1, 2, 10
        raw_logits = np.random.randn(b, layers, heads, n, n)
        exp = np.exp(raw_logits - raw_logits.max(axis=-1, keepdims=True))
        attn = exp / exp.sum(axis=-1, keepdims=True)
        origin_ns = pd.date_range("2012-06-01 08:00", periods=b, freq="3h").asi8
        adj = (np.random.rand(n, n) > 0.7).astype(np.float32)
        np.fill_diagonal(adj, 1.0)

        df = compute_attention_stats(attn, origin_ns, adj)
        self.assertFalse(df.empty)
        self.assertIn("entropy", df.columns)
        self.assertIn("mass_self", df.columns)
        self.assertIn("mass_downstream", df.columns)
        self.assertIn("mass_upstream", df.columns)
        self.assertEqual(len(df), n * 3)  # all, peak, off_peak


if __name__ == "__main__":
    unittest.main()
