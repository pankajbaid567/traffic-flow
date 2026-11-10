"""Tiny GRU model for pipeline tests, smoke runs and speed measurements.

``dummy_gru`` is NOT a project model and is never reported. It exists so the
trainer, CLI and Colab notebook can be exercised before the real models land.

* shared GRU over the 12 steps for each sensor (weights shared across sensors),
* optional ``mix: true``: one round of row-normalised neighbour averaging of the
  GRU states through ``adj`` (then ``needs_graph = True``),
* linear head predicts a residual to the last filled speed for the 4 horizons.
"""

from __future__ import annotations

import torch
from torch import nn

from src.phase2.registry import register_model


class DummyGRU(nn.Module):
    def __init__(
        self,
        in_channels: int = 7,
        hidden: int = 16,
        num_horizons: int = 4,
        mix: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.needs_graph = bool(mix)
        self.mix = bool(mix)
        self.gru = nn.GRU(in_channels, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, num_horizons)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        b, t, n, c = x.shape
        seq = x.permute(0, 2, 1, 3).reshape(b * n, t, c)  # sensors share weights
        _, h = self.gru(seq)
        h = h[-1].reshape(b, n, -1)  # [B, N, H]
        if self.mix:
            weights = adj / adj.sum(dim=1, keepdim=True).clamp_min(1e-12)
            h = torch.einsum("ij,bjh->bih", weights, h)
        delta = self.head(self.dropout(h)).permute(0, 2, 1)  # [B, 4, N]
        return x[:, -1, :, 0].unsqueeze(1) + delta


@register_model("dummy_gru")
def build(cfg: dict) -> nn.Module:
    return DummyGRU(
        in_channels=int(cfg.get("in_channels", 7)),
        hidden=int(cfg.get("hidden", 16)),
        num_horizons=int(cfg.get("num_horizons", 4)),
        mix=bool(cfg.get("mix", False)),
        dropout=float(cfg.get("dropout", 0.0)),
    )
