"""Non-learned persistence model used for pipeline parity tests.

Output = the filled current speed (channel 0 at the origin) repeated for the 4
horizons, in z-scored units. After the trainer inverts the scaler this is
Phase 1's ``persistence`` forecast, so it must reproduce Phase 1 persistence
metrics exactly (up to float32 round-off).
"""

from __future__ import annotations

import torch
from torch import nn

from src.phase2.registry import register_model


class LastValue(nn.Module):
    needs_graph = False
    trainable = False  # no parameters; the trainer only evaluates it

    def __init__(self, num_horizons: int = 4) -> None:
        super().__init__()
        self.num_horizons = num_horizons

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        last = x[:, -1, :, 0]  # [B, N] filled speed at the origin
        return last.unsqueeze(1).expand(-1, self.num_horizons, -1).contiguous()


@register_model("last_value")
def build(cfg: dict) -> nn.Module:
    return LastValue(num_horizons=int(cfg.get("num_horizons", 4)))
