"""Diffusion Convolutional Recurrent Neural Network (DCRNN) in pure PyTorch.

Owner: Member 3 (phase2/m3-dcrnn-controls). Contract: reports/phase2_plan.md section 4.4, 8.3.
Reference: Li, Yu, Shahabi, Liu (ICLR 2018), "Diffusion Convolutional Recurrent Neural Network".

Features:
- DiffusionConv: Spatiotemporal graph convolution using random-walk diffusion supports:
  forward transition P_f = D_O^-1 A and reverse transition P_r = D_I^-1 A^T.
- Supports calculated INSIDE the model from adj argument with safe division for isolated nodes.
- DCGRUCell: GRU cell with diffusion convolutions replacing matrix multiplications.
- Encoder-Decoder architecture:
  - Encoder processes 12 past steps.
  - Decoder recurrently steps through 12 future horizons and extracts steps 1, 3, 6, 12
    (5, 15, 30, 60 minutes) to match the project supervision contract.
- Parameter ``supports``: "diffusion" (standard) or "identity" (for dcrnn_identity control).
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from src.phase2.registry import register_model


def calculate_random_walk_supports(
    adj: torch.Tensor, K: int = 1, supports_type: str = "diffusion"
) -> List[torch.Tensor]:
    """Compute forward and reverse diffusion transition matrices up to K hops.

    Args:
        adj: Dense adjacency matrix [N, N].
        K: Number of diffusion steps (1 to 3).
        supports_type: "diffusion" (forward + reverse powers) or "identity" ([I_N]).

    Returns:
        List of support tensors of shape [N, N].
    """
    n = adj.size(0)
    device = adj.device
    dtype = adj.dtype
    eye = torch.eye(n, device=device, dtype=dtype)

    if supports_type == "identity" or K == 0:
        return [eye]

    # Forward random walk: P_f = D_O^-1 * A
    d_out = adj.sum(dim=1)
    d_out_inv = torch.where(d_out > 0, 1.0 / d_out, torch.zeros_like(d_out))
    p_forward = d_out_inv.unsqueeze(1) * adj

    # Reverse random walk: P_r = D_I^-1 * A^T
    adj_t = adj.t()
    d_in = adj_t.sum(dim=1)
    d_in_inv = torch.where(d_in > 0, 1.0 / d_in, torch.zeros_like(d_in))
    p_reverse = d_in_inv.unsqueeze(1) * adj_t

    supports = [eye]
    p_f_k = eye
    p_r_k = eye
    for _ in range(K):
        p_f_k = torch.matmul(p_f_k, p_forward)
        p_r_k = torch.matmul(p_r_k, p_reverse)
        supports.append(p_f_k)
        supports.append(p_r_k)

    return supports


class DiffusionConv(nn.Module):
    """Diffusion Convolution Layer: sum_{s} P_s X W_s + b."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        K: int = 2,
        supports_type: str = "diffusion",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.K = K
        self.supports_type = supports_type

        # Number of matrices: if identity/K=0 -> 1 (I_N); if diffusion -> 2*K + 1
        num_supports = 1 if (supports_type == "identity" or K == 0) else (2 * K + 1)
        self.num_supports = num_supports

        self.weights = nn.Parameter(torch.empty(num_supports, in_dim, out_dim))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_dim))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weights)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, supports: List[torch.Tensor]) -> torch.Tensor:
        """Args:

        x: [B, N, in_dim]
        supports: list of [N, N] transition matrices

        Returns:
            [B, N, out_dim]
        """
        # Truncate or match supports if needed
        used_supports = supports[: self.num_supports]
        outputs = []
        for s_idx, p_s in enumerate(used_supports):
            # p_s @ x: [N, N] x [B, N, in_dim] -> [B, N, in_dim]
            px = torch.matmul(p_s, x)
            # px @ W_s: [B, N, in_dim] x [in_dim, out_dim] -> [B, N, out_dim]
            pw = torch.matmul(px, self.weights[s_idx])
            outputs.append(pw)

        out = torch.stack(outputs, dim=0).sum(dim=0)
        if self.bias is not None:
            out = out + self.bias
        return out


class DCGRUCell(nn.Module):
    """Diffusion Convolutional GRU Cell."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        K: int = 2,
        supports_type: str = "diffusion",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        # Reset gate and update gate (combined for efficiency)
        self.conv_ru = DiffusionConv(
            in_dim=in_dim + hidden_dim,
            out_dim=2 * hidden_dim,
            K=K,
            supports_type=supports_type,
        )
        # Candidate state gate
        self.conv_c = DiffusionConv(
            in_dim=in_dim + hidden_dim,
            out_dim=hidden_dim,
            K=K,
            supports_type=supports_type,
        )

    def forward(
        self, x: torch.Tensor, h: torch.Tensor, supports: List[torch.Tensor]
    ) -> torch.Tensor:
        """Args:

        x: [B, N, in_dim]
        h: [B, N, hidden_dim]
        supports: list of [N, N] transition matrices

        Returns:
            h_next: [B, N, hidden_dim]
        """
        # [B, N, in_dim + hidden_dim]
        combined = torch.cat([x, h], dim=-1)
        ru = torch.sigmoid(self.conv_ru(combined, supports))
        r, u = torch.split(ru, self.hidden_dim, dim=-1)

        combined_c = torch.cat([x, r * h], dim=-1)
        c = torch.tanh(self.conv_c(combined_c, supports))

        h_next = u * h + (1.0 - u) * c
        return h_next


class DCRNNModel(nn.Module):
    """Diffusion Convolutional Recurrent Neural Network (Encoder-Decoder).

    - Encoder: Recurrently encodes 12 history steps across the graph.
    - Decoder: Recurrently rolls out 12 future steps, collecting predictions at
      indices [0, 2, 5, 11] (5, 15, 30, 60 minutes).
    - Residual head: Output is added to the last filled input speed x[:, -1, :, 0].
    """

    needs_graph: bool = True

    def __init__(
        self,
        num_nodes: int = 207,
        in_channels: int = 7,
        history: int = 12,
        num_horizons: int = 4,
        hidden: int = 32,
        layers: int = 1,
        K: int = 2,
        dropout: float = 0.0,
        supports: str = "diffusion",
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.history = history
        self.num_horizons = num_horizons
        self.hidden = hidden
        self.num_layers = layers
        self.K = K
        self.supports_type = supports.lower()

        # Encoder layers
        self.encoder_cells = nn.ModuleList(
            [
                DCGRUCell(
                    in_dim=in_channels if l == 0 else hidden,
                    hidden_dim=hidden,
                    K=K,
                    supports_type=self.supports_type,
                )
                for l in range(layers)
            ]
        )

        # Decoder layers (takes 1-dim speed as input at each step)
        self.decoder_cells = nn.ModuleList(
            [
                DCGRUCell(
                    in_dim=1 if l == 0 else hidden,
                    hidden_dim=hidden,
                    K=K,
                    supports_type=self.supports_type,
                )
                for l in range(layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden, 1)

        # Map the 4 target horizons (5, 15, 30, 60 min) to step indices (0, 2, 5, 11)
        self.horizon_steps = [0, 2, 5, 11]

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forecast 4 horizons.

        Args:
            x: [B, 12, N, 7]
            adj: [N, N]

        Returns:
            [B, 4, N] predictions in z-scored units.
        """
        b, t, n, c = x.shape
        supports = calculate_random_walk_supports(
            adj, K=self.K, supports_type=self.supports_type
        )

        # --- ENCODER ---
        h_enc = [torch.zeros(b, n, self.hidden, device=x.device, dtype=x.dtype) for _ in range(self.num_layers)]

        for step in range(t):
            x_t = x[:, step, :, :]  # [B, N, C]
            cur_input = x_t
            for l in range(self.num_layers):
                h_enc[l] = self.encoder_cells[l](cur_input, h_enc[l], supports)
                cur_input = self.dropout(h_enc[l])

        # --- DECODER ---
        h_dec = [h for h in h_enc]
        # Start decoder with the last known filled speed: [B, N, 1]
        last_speed = x[:, -1, :, 0:1]
        dec_input = last_speed

        predictions = []
        max_decode_steps = max(self.horizon_steps) + 1  # 12 steps (indices 0..11)

        for step in range(max_decode_steps):
            cur_input = dec_input
            for l in range(self.num_layers):
                h_dec[l] = self.decoder_cells[l](cur_input, h_dec[l], supports)
                cur_input = self.dropout(h_dec[l])

            # Predict delta relative to last speed for this step: [B, N, 1]
            pred_step = last_speed + self.out_proj(h_dec[-1])

            if step in self.horizon_steps:
                predictions.append(pred_step.squeeze(-1))  # [B, N]

            # Autoregressive feed forward for next decode step
            dec_input = pred_step.detach()

        # Stack horizons -> [B, 4, N]
        out = torch.stack(predictions, dim=1)
        return out


@register_model("dcrnn")
def build(cfg: dict) -> nn.Module:
    """Builder registered for the Phase 2 model registry."""
    return DCRNNModel(
        num_nodes=int(cfg.get("num_nodes", 207)),
        in_channels=int(cfg.get("in_channels", 7)),
        history=int(cfg.get("history", 12)),
        num_horizons=int(cfg.get("num_horizons", 4)),
        hidden=int(cfg.get("hidden", 32)),
        layers=int(cfg.get("layers", 1)),
        K=int(cfg.get("K", 2)),
        dropout=float(cfg.get("dropout", 0.0)),
        supports=str(cfg.get("supports", "diffusion")),
    )
