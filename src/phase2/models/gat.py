"""Graph Attention Network (GAT) for spatiotemporal traffic speed forecasting.

Owner: Member 2 (phase2/m2-gat). Contract: reports/phase2_plan.md section 4.4.

Architecture:
- Shared temporal encoder over the 12 history steps per sensor (GRU or 1D-Conv).
- Stacked dense multi-head Graph Attention (GAT) layers with graph masking.
  Non-edges (where adj == 0 and i != j) are masked to -inf before softmax.
  Self-loops are always permitted.
- Residual connections, LayerNorm, and dropout per GAT layer.
- Linear/MLP head predicting the residual change from the last known filled speed
  (x[:, -1, :, 0]) for the 4 forecast horizons (5, 15, 30, 60 min).
- Exports attention maps via ``attention_maps(x, adj)``.
"""

from __future__ import annotations

import math
from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from src.phase2.registry import register_model


class GATLayer(nn.Module):
    """Dense Multi-Head Graph Attention Layer in plain PyTorch.

    Computes masked attention across nodes based on the adjacency matrix:
    alpha_{i, j} = softmax_j( LeakyReLU( a_src^T W h_i + a_dst^T W h_j + bias(adj_ij) ) )
    where non-edges (adj == 0 and i != j) are masked out with -1e9.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 2,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
        use_edge_weights: bool = False,
    ) -> None:
        super().__init__()
        if out_dim % heads != 0:
            raise ValueError(f"out_dim ({out_dim}) must be divisible by heads ({heads})")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.head_dim = out_dim // heads
        self.negative_slope = negative_slope
        self.use_edge_weights = bool(use_edge_weights)

        # Projections
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_src = nn.Parameter(torch.empty(1, heads, 1, self.head_dim))
        self.attn_dst = nn.Parameter(torch.empty(1, heads, 1, self.head_dim))
        if self.use_edge_weights:
            self.edge_bias = nn.Parameter(torch.empty(1, heads, 1, 1))
        else:
            self.register_parameter("edge_bias", None)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)
        if in_dim != out_dim:
            self.res_proj = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.res_proj = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        gain = nn.init.calculate_gain("leaky_relu", self.negative_slope)
        nn.init.xavier_uniform_(self.fc.weight, gain=gain)
        nn.init.xavier_uniform_(self.attn_src, gain=gain)
        nn.init.xavier_uniform_(self.attn_dst, gain=gain)
        if self.use_edge_weights and self.edge_bias is not None:
            nn.init.zeros_(self.edge_bias)

    def forward(
        self, h: torch.Tensor, adj: torch.Tensor, return_attn: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            h: Node representations [B, N, in_dim].
            adj: Adjacency matrix [N, N].
            return_attn: If True, also return attention weights [B, heads, N, N].

        Returns:
            Updated representations [B, N, out_dim], and optionally attention weights.
        """
        b, n, _ = h.shape
        residual = self.res_proj(h)

        # Project and reshape into multi-head: [B, N, heads, head_dim] -> [B, heads, N, head_dim]
        wh = self.fc(h).view(b, n, self.heads, self.head_dim).permute(0, 2, 1, 3)

        # Self-attention logits: [B, heads, N, 1] and [B, heads, 1, N]
        alpha_src = (wh * self.attn_src).sum(dim=-1, keepdim=True)
        alpha_dst = (wh * self.attn_dst).sum(dim=-1, keepdim=True).permute(0, 1, 3, 2)
        logits = F.leaky_relu(alpha_src + alpha_dst, negative_slope=self.negative_slope)

        if self.use_edge_weights and self.edge_bias is not None:
            # Add edge weight scaled bias
            logits = logits + self.edge_bias * adj.view(1, 1, n, n)

        # Mask non-neighbours (keep adj > 0 and self-loops)
        # Ensure diagonal (self-loop) is always included
        eye = torch.eye(n, device=adj.device, dtype=torch.bool)
        mask = (adj > 0) | eye  # [N, N]
        logits = logits.masked_fill(~mask.view(1, 1, n, n), -1e9)

        # Softmax over neighbours (last dimension)
        attn = F.softmax(logits, dim=-1)
        # Mask out any numerical precision leaks on non-edges
        attn = attn * mask.view(1, 1, n, n).to(attn.dtype)
        # Re-normalise to guarantee row-sum exactly 1
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        attn_drop = self.dropout(attn)

        # Aggregate values: [B, heads, N, N] @ [B, heads, N, head_dim] -> [B, heads, N, head_dim]
        out = torch.matmul(attn_drop, wh)
        # Permute back: [B, N, heads, head_dim] -> [B, N, out_dim]
        out = out.permute(0, 2, 1, 3).contiguous().view(b, n, self.out_dim)

        out = self.layer_norm(residual + out)
        if return_attn:
            return out, attn
        return out


class TemporalEncoder(nn.Module):
    """Encodes 12-step per-sensor temporal inputs into hidden embeddings."""

    def __init__(
        self,
        in_channels: int = 7,
        hidden: int = 32,
        kind: str = "gru",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.kind = kind.lower()
        if self.kind == "gru":
            self.encoder = nn.GRU(in_channels, hidden, batch_first=True)
        elif self.kind == "conv":
            self.encoder = nn.Sequential(
                nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
        else:
            raise ValueError(f"Unknown temporal encoder kind: {kind}; use 'gru' or 'conv'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x of shape [B, T=12, N, C=7].

        Returns: Tensor of shape [B, N, hidden].
        """
        b, t, n, c = x.shape
        # Sensors share weights: reshape to [B * N, T, C]
        seq = x.permute(0, 2, 1, 3).reshape(b * n, t, c)
        if self.kind == "gru":
            _, h = self.encoder(seq)
            out = h[-1].view(b, n, -1)  # [B, N, hidden]
        else:
            # Conv1d expects [B*N, C, T]
            seq_t = seq.permute(0, 2, 1)
            feat = self.encoder(seq_t)  # [B*N, hidden, 1]
            out = feat.view(b, n, -1)  # [B, N, hidden]
        return out


class GATModel(nn.Module):
    """Spatiotemporal Graph Attention Network.

    - Encodes 12 past steps per sensor.
    - Propagates representations through 1-2 multi-head masked GAT layers.
    - Residual head produces horizon forecasts relative to x[:, -1, :, 0].
    """

    needs_graph: bool = True

    def __init__(
        self,
        num_nodes: int = 207,
        in_channels: int = 7,
        history: int = 12,
        num_horizons: int = 4,
        hidden: int = 32,
        heads: int = 2,
        layers: int = 1,
        dropout: float = 0.1,
        temporal_encoder: str = "gru",
        use_edge_weights: bool = False,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.num_horizons = num_horizons
        self.hidden = hidden
        self.heads = heads
        self.num_layers = layers

        self.temporal_encoder = TemporalEncoder(
            in_channels=in_channels,
            hidden=hidden,
            kind=temporal_encoder,
            dropout=dropout,
        )

        self.gat_layers = nn.ModuleList(
            [
                GATLayer(
                    in_dim=hidden,
                    out_dim=hidden,
                    heads=heads,
                    dropout=dropout,
                    use_edge_weights=use_edge_weights,
                )
                for _ in range(layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_horizons),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forecast 4 horizons.

        Args:
            x: [B, 12, N, 7]
            adj: [N, N]

        Returns:
            Predictions [B, 4, N] in z-scored units.
        """
        # Temporal encoding: [B, N, hidden]
        h = self.temporal_encoder(x)

        # Graph attention layers
        for layer in self.gat_layers:
            h = layer(h, adj, return_attn=False)

        # Head predicts residual change from the last known filled speed
        delta = self.head(self.dropout(h)).permute(0, 2, 1)  # [B, 4, N]
        last_speed = x[:, -1, :, 0].unsqueeze(1)  # [B, 1, N]
        return last_speed + delta

    def attention_maps(self, x: torch.Tensor, adj: torch.Tensor) -> List[torch.Tensor]:
        """Extract attention matrices from each GAT layer.

        Returns:
            List of tensors of shape [B, heads, N, N], one per layer.
        """
        h = self.temporal_encoder(x)
        maps: List[torch.Tensor] = []
        for layer in self.gat_layers:
            h, attn = layer(h, adj, return_attn=True)
            maps.append(attn)
        return maps


@register_model("gat")
def build(cfg: dict) -> nn.Module:
    """Builder registered for the Phase 2 model registry."""
    return GATModel(
        num_nodes=int(cfg.get("num_nodes", 207)),
        in_channels=int(cfg.get("in_channels", 7)),
        history=int(cfg.get("history", 12)),
        num_horizons=int(cfg.get("num_horizons", 4)),
        hidden=int(cfg.get("hidden", 32)),
        heads=int(cfg.get("heads", 2)),
        layers=int(cfg.get("layers", 1)),
        dropout=float(cfg.get("dropout", 0.1)),
        temporal_encoder=str(cfg.get("temporal_encoder", cfg.get("encoder", "gru"))),
        use_edge_weights=bool(cfg.get("use_edge_weights", False)),
    )
