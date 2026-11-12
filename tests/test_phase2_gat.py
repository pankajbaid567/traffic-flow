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
        """Verify attention stats with hand-computable expected values.

        Uses a 2-node uniform attention matrix so entropy and mass values
        can be derived analytically and asserted exactly (within float tolerance).
        """
        import math
        import numpy as np
        import pandas as pd
        from scripts.phase2_gat_attention_stats import compute_attention_stats

        # --- hand-computable 2-node case ---
        # Attention matrix: each row is [0.6, 0.4] (uniform across both nodes)
        n = 2
        p_self, p_other = 0.6, 0.4
        # Shape: [B=1, layers=1, heads=1, N=2, N=2]
        attn_row = np.array([[p_self, p_other], [p_other, p_self]], dtype=np.float32)
        attn = attn_row[np.newaxis, np.newaxis, np.newaxis, :, :]  # [1,1,1,2,2]

        # One origin in peak hours (8 AM)
        origin_ns = pd.to_datetime(["2012-06-01 08:00"]).asi8

        # Directed graph: node 0 -> node 1 (downstream), no reverse edge
        # adj[i,j]>0 means j is downstream of i
        adj = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32)

        df = compute_attention_stats(attn, origin_ns, adj)

        # Expected entropy for a [0.6, 0.4] distribution:
        eps = 1e-12
        expected_entropy = -(p_self * math.log(p_self + eps) + p_other * math.log(p_other + eps))

        # Filter to 'all' period for sensor 0
        row0 = df[(df["sensor_idx"] == 0) & (df["period"] == "all")].iloc[0]
        self.assertAlmostEqual(row0["entropy"], expected_entropy, places=5,
                               msg="Entropy mismatch for sensor 0")
        self.assertAlmostEqual(row0["mass_self"], p_self, places=5,
                               msg="mass_self mismatch for sensor 0")
        # Node 1 is downstream of node 0: attention mass on downstream = p_other
        self.assertAlmostEqual(row0["mass_downstream"], p_other, places=5,
                               msg="mass_downstream mismatch for sensor 0")
        # Node 1 is NOT upstream of node 0
        self.assertAlmostEqual(row0["mass_upstream"], 0.0, places=5,
                               msg="mass_upstream should be 0.0 for sensor 0")

        # Also verify structural properties on the full random case
        b2, layers2, heads2, n2 = 4, 1, 2, 10
        raw2 = np.random.randn(b2, layers2, heads2, n2, n2)
        exp2 = np.exp(raw2 - raw2.max(axis=-1, keepdims=True))
        attn2 = exp2 / exp2.sum(axis=-1, keepdims=True)
        origin_ns2 = pd.date_range("2012-06-01 08:00", periods=b2, freq="3h").asi8
        adj2 = (np.random.rand(n2, n2) > 0.7).astype(np.float32)
        np.fill_diagonal(adj2, 1.0)

        df2 = compute_attention_stats(attn2, origin_ns2, adj2)
        self.assertFalse(df2.empty)
        for col in ("entropy", "mass_self", "mass_downstream", "mass_upstream"):
            self.assertIn(col, df2.columns)
        self.assertEqual(len(df2), n2 * 3)  # all, peak, off_peak


if __name__ == "__main__":
    unittest.main()
