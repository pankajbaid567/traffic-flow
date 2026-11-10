"""Unit tests for Diffusion Convolutional Recurrent Neural Network (DCRNN).

Owner: Member 3 (phase2/m3-dcrnn-controls). Contract: reports/phase2_plan.md section 8.3.
"""

from __future__ import annotations

import unittest

import torch
from torch import nn

from src.phase2.models.dcrnn import (
    DiffusionConv,
    DCRNNModel,
    build,
    calculate_random_walk_supports,
)
from src.phase2.registry import get_builder


class DCRNNModelTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.b = 4
        self.t = 12
        self.n = 8
        self.c = 7
        self.num_horizons = 4
        self.x = torch.randn(self.b, self.t, self.n, self.c)

        # Random connected directed adjacency with self-loops
        prob = torch.rand(self.n, self.n)
        self.adj = (prob > 0.65).float()
        self.adj.fill_diagonal_(1.0)
        self.eye_adj = torch.eye(self.n)

    def test_registered_and_build(self) -> None:
        builder = get_builder("dcrnn")
        self.assertIsNotNone(builder)
        cfg = {"num_nodes": self.n, "hidden": 32, "layers": 1, "K": 2}
        model = builder(cfg)
        self.assertIsInstance(model, DCRNNModel)
        self.assertTrue(model.needs_graph)

    def test_output_shape_and_finiteness(self) -> None:
        for layers in [1, 2]:
            with self.subTest(layers=layers):
                model = DCRNNModel(
                    num_nodes=self.n, hidden=32, layers=layers, K=2, dropout=0.0
                )
                model.eval()
                with torch.no_grad():
                    out = model(self.x, self.adj)
                self.assertEqual(out.shape, (self.b, self.num_horizons, self.n))
                self.assertTrue(torch.isfinite(out).all().item())

    def test_diffusion_conv_identity_equals_linear(self) -> None:
        """With supports_type='identity' (or identity graph), DiffusionConv

        reduces to a standard linear transformation over channels.
        """
        conv = DiffusionConv(
            in_dim=16, out_dim=32, K=1, supports_type="identity", bias=True
        )
        supports = [self.eye_adj]
        x_node = torch.randn(self.b, self.n, 16)

        out_conv = conv(x_node, supports)  # [B, N, 32]
        # Equivalent linear: out = x @ W_0 + bias
        out_linear = torch.matmul(x_node, conv.weights[0]) + conv.bias

        self.assertTrue(torch.allclose(out_conv, out_linear, atol=1e-5))

    def test_k_zero_gives_no_neighbour_mixing(self) -> None:
        """When K=0, supports contain only I_N so changing neighbour j does not affect i."""
        model = DCRNNModel(num_nodes=self.n, hidden=32, layers=1, K=0)
        model.eval()

        with torch.no_grad():
            out_base = model(self.x, self.adj)

        x_perturbed = self.x.clone()
        x_perturbed[:, :, 0, :] += 5.0

        with torch.no_grad():
            out_perturbed = model(x_perturbed, self.adj)

        # Other nodes should not be affected because K=0 uses only I_N
        self.assertTrue(
            torch.allclose(out_base[:, :, 1:], out_perturbed[:, :, 1:], atol=1e-5),
            "K=0 leaked information across sensors!",
        )

    def test_three_node_line_graph_one_hop_per_diffusion_step(self) -> None:
        """On a 3-node line graph 0 -> 1 -> 2:

        - Hop K=1 forward: node 0 diffuses to node 1, but NOT to node 2.
        - Hop K=2 forward: node 0 diffuses to node 2.
        """
        n = 3
        adj = torch.zeros(n, n)
        adj[0, 1] = 1.0  # 0 -> 1
        adj[1, 2] = 1.0  # 1 -> 2
        adj.fill_diagonal_(1.0)

        supports_k1 = calculate_random_walk_supports(
            adj, K=1, supports_type="diffusion"
        )
        # supports_k1 has: [I, P_f, P_r]
        p_f1 = supports_k1[1]  # forward 1-hop
        self.assertGreater(p_f1[0, 1].item(), 0.0)  # 1 reachable from 0
        self.assertEqual(p_f1[0, 2].item(), 0.0)  # 2 NOT reachable in 1 hop

        supports_k2 = calculate_random_walk_supports(
            adj, K=2, supports_type="diffusion"
        )
        # supports_k2 has: [I, P_f, P_r, P_f^2, P_r^2]
        p_f2 = supports_k2[3]  # forward 2-hop (P_f^2)
        self.assertGreater(
            p_f2[0, 2].item(), 0.0
        )  # 2 IS reachable in 2 hops from 0!

        # Reverse check: 0 is reachable from 1 in 1 hop reverse (P_r[1, 0] > 0)
        p_r1 = supports_k1[2]
        self.assertGreater(p_r1[1, 0].item(), 0.0)
        self.assertEqual(p_r1[2, 0].item(), 0.0)

    def test_dcrnn_identity_neighbour_independence(self) -> None:
        """With supports='identity' or identity graph, modifying sensor j's input

        leaves sensor i's output unchanged.
        """
        model = DCRNNModel(
            num_nodes=self.n,
            hidden=32,
            layers=1,
            supports="identity",
            dropout=0.0,
        )
        model.eval()

        with torch.no_grad():
            out_base = model(self.x, self.eye_adj)

        x_perturbed = self.x.clone()
        x_perturbed[:, :, 0, :] += 5.0

        with torch.no_grad():
            out_perturbed = model(x_perturbed, self.eye_adj)

        self.assertFalse(
            torch.allclose(out_base[:, :, 0], out_perturbed[:, :, 0], atol=1e-4)
        )
        self.assertTrue(
            torch.allclose(out_base[:, :, 1:], out_perturbed[:, :, 1:], atol=1e-5),
            "dcrnn with identity graph affected non-perturbed sensors!",
        )

    def test_node_permutation_equivariance(self) -> None:
        """Permuting sensor order in x and adj permutes output identically."""
        model = DCRNNModel(num_nodes=self.n, hidden=32, layers=1, K=1, dropout=0.0)
        model.eval()

        perm = torch.randperm(self.n)
        x_perm = self.x[:, :, perm, :]
        adj_perm = self.adj[perm][:, perm]

        with torch.no_grad():
            out_orig = model(self.x, self.adj)
            out_perm = model(x_perm, adj_perm)

        out_orig_perm = out_orig[:, :, perm]
        self.assertTrue(
            torch.allclose(out_orig_perm, out_perm, atol=1e-4),
            "DCRNN is not node-permutation equivariant!",
        )

    def test_overfit_on_small_batch(self) -> None:
        """Loss should drop strongly on 64 fixed synthetic samples."""
        b = 64
        x = torch.randn(b, 12, 6, 7)
        y = torch.randn(b, 4, 6)
        adj = torch.eye(6)

        model = DCRNNModel(num_nodes=6, hidden=32, layers=1, K=1, dropout=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = nn.MSELoss()

        model.train()
        initial_loss = 0.0
        final_loss = 0.0
        for step in range(80):
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
            initial_loss * 0.45,
            f"Overfit failed: initial {initial_loss:.4f}, final {final_loss:.4f}",
        )

    def test_parameter_counts(self) -> None:
        cfgs = [
            {"hidden": 32, "layers": 1, "K": 1},
            {"hidden": 32, "layers": 1, "K": 2},
            {"hidden": 64, "layers": 1, "K": 2},
            {"hidden": 32, "layers": 2, "K": 2},
        ]
        for cfg in cfgs:
            model = DCRNNModel(num_nodes=207, **cfg)
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.assertGreater(param_count, 1000)
            print(f"[dcrnn param count] cfg={cfg} -> {param_count:,} parameters")


if __name__ == "__main__":
    unittest.main()
