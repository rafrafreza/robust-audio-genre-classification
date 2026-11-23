"""5-block CNN trained from scratch on 128 x 216 log-mel images (notebook 03).

Input (B, 1, 128, 216)
  -> ConvBlock(1->32)    -> (B, 32, 64, 108)
  -> ConvBlock(32->64)   -> (B, 64, 32, 54)
  -> ConvBlock(64->128)  -> (B, 128, 16, 27)
  -> ConvBlock(128->256) -> (B, 256, 8, 13)
  -> ConvBlock(256->256, no pool) -> (B, 256, 8, 13)
  -> global average pool -> MLP head -> (B, 10)

2,454,058 parameters (2,353,312 in the conv backbone).
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from ..config import CNNConfig


class ConvBlock(nn.Module):
    """Two stacked 3x3 convs per resolution level (VGG-style): a 5x5 receptive field
    with fewer parameters and one more non-linearity than a single 5x5 conv."""

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.b = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(x)


def build_backbone(channels: Sequence[int], pool: Sequence[bool]) -> nn.Sequential:
    """Stack ConvBlocks; shared with the CRNN so the two models differ only after it."""
    blocks, in_ch = [], 1
    for out_ch, p in zip(channels, pool):
        blocks.append(ConvBlock(in_ch, out_ch, pool=p))
        in_ch = out_ch
    return nn.Sequential(*blocks)


class GenreCNN(nn.Module):
    def __init__(self, cfg: CNNConfig = CNNConfig(), n_classes: int = 10) -> None:
        super().__init__()
        self.features = build_backbone(cfg.channels, cfg.pool)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        head, in_dim = [nn.Flatten()], cfg.channels[-1]
        for dim, p in zip(cfg.head_dims, cfg.head_dropouts):
            head += [nn.Linear(in_dim, dim), nn.BatchNorm1d(dim), nn.ReLU(inplace=True),
                     nn.Dropout(p)]
            in_dim = dim
        head.append(nn.Linear(in_dim, n_classes))
        self.head = nn.Sequential(*head)
        self._init_weights()

    def _init_weights(self) -> None:
        # He init is matched to ReLU, keeping activation variance roughly constant
        # through the 10 conv layers.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.gap(self.features(x)))
