"""Non-graph deep temporal control models for traffic forecasting.

Owner: Member 3 (phase2/m3-dcrnn-controls). Contract: reports/phase2_plan.md section 4.4, 8.3.

Provides:
- "gru_local" (needs_graph = False): A shared per-sensor GRU over the 12 history steps
  with an MLP head predicting the residual change from x[:, -1, :, 0] for the 4 horizons.
  Each sensor's output depends strictly on that sensor's own inputs.
"""

from __future__ import annotations

import torch
from torch import nn

from src.phase2.registry import register_model


class GRULocal(nn.Module):
    """Shared per-sensor GRU baseline without road graph neighbour mixing.

    - Input: x of shape [B, 12, N=207, C=7]
    - Weights are shared across all sensors.
    - Each sensor is processed completely independently.
    - Residual head relative to the last filled input speed x[:, -1, :, 0].
    """

    needs_graph: bool = False

    def __init__(
        self,
        num_nodes: int = 207,
        in_channels: int = 7,
        history: int = 12,
        num_horizons: int = 4,
        hidden: int = 32,
        layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.num_horizons = num_horizons
        self.hidden = hidden
        self.num_layers = layers

        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_horizons),
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        """Forecast 4 horizons.

        Args:
            x: [B, 12, N, 7]
            adj: Ignored (needs_graph = False).

        Returns:
            Predictions [B, 4, N] in z-scored units.
        """
        b, t, n, c = x.shape
        # Sensors share weights: reshape to [B * N, T, C]
        seq = x.permute(0, 2, 1, 3).reshape(b * n, t, c)
        _, h = self.gru(seq)
        # Last hidden state of the top GRU layer: [B * N, hidden]
        h_last = h[-1].view(b, n, self.hidden)

        delta = self.head(self.dropout(h_last)).permute(0, 2, 1)  # [B, 4, N]
        last_speed = x[:, -1, :, 0].unsqueeze(1)  # [B, 1, N]
        return last_speed + delta


@register_model("gru_local")
def build(cfg: dict) -> nn.Module:
    """Builder registered for the Phase 2 model registry."""
    return GRULocal(
        num_nodes=int(cfg.get("num_nodes", 207)),
        in_channels=int(cfg.get("in_channels", 7)),
        history=int(cfg.get("history", 12)),
        num_horizons=int(cfg.get("num_horizons", 4)),
        hidden=int(cfg.get("hidden", 32)),
        layers=int(cfg.get("layers", 1)),
        dropout=float(cfg.get("dropout", 0.0)),
    )

