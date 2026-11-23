"""CNN backbone + bidirectional GRU + attention pooling (notebook 04).

Same 5-block backbone as ``GenreCNN``. Instead of averaging away time, the frequency
axis is pooled and the 13 remaining time steps are read as a sequence:

  (B, 256, 8, 13) -> freq-pool -> (B, 13, 256) -> BiGRU -> (B, 13, 256)
  -> attention pool -> (B, 256) -> MLP head -> (B, 10)

2,750,507 parameters.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..config import CRNNConfig
from .cnn import build_backbone


class AttentionPool(nn.Module):
    """Learned softmax weighting over time steps.

    In a noisy mashup some seconds are dominated by an ESC-50 burst; attention gives
    the model a way to down-weight those steps instead of averaging them in, and unlike
    taking the last GRU state it does not privilege the end of the clip.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.w = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, D)
        weights = F.softmax(self.w(x).squeeze(-1), dim=1)
        return (x * weights.unsqueeze(-1)).sum(dim=1)


class CRNN(nn.Module):
    def __init__(self, cfg: CRNNConfig = CRNNConfig(), n_classes: int = 10) -> None:
        super().__init__()
        self.cnn = build_backbone(cfg.channels, cfg.pool)
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.gru = nn.GRU(
            input_size=cfg.channels[-1], hidden_size=cfg.gru_hidden,
            num_layers=cfg.gru_layers, batch_first=True, bidirectional=True,
        )
        gru_out = cfg.gru_hidden * 2
        self.attn = AttentionPool(gru_out)

        head, in_dim = [nn.LayerNorm(gru_out)], gru_out
        for dim, p in zip(cfg.head_dims, cfg.head_dropouts):
            head += [nn.Linear(in_dim, dim), nn.ReLU(inplace=True), nn.Dropout(p)]
            in_dim = dim
        head.append(nn.Linear(in_dim, n_classes))
        self.head = nn.Sequential(*head)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Xavier here vs He in GenreCNN follows the notebook; not ablated.
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)                               # (B, C, F', T')
        x = self.freq_pool(x).squeeze(2).permute(0, 2, 1)  # (B, T', C)
        x, _ = self.gru(x)
        return self.head(self.attn(x))
