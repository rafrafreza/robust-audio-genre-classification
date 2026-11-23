"""Model zoo. ``build_model`` covers the PyTorch models; the baseline is sklearn and
lives in ``baseline.py``. AST imports ``transformers`` lazily so the CNN/CRNN paths
do not require it."""

from __future__ import annotations

from typing import Sequence

from torch import nn

from ..config import GENRES, ModelConfig
from .cnn import GenreCNN
from .crnn import CRNN


def build_model(name: str, cfg: ModelConfig, genres: Sequence[str] = GENRES,
                pretrained: bool = True) -> nn.Module:
    """``pretrained=False`` builds AST's architecture only, for loading a checkpoint."""
    if name == "cnn":
        return GenreCNN(cfg, n_classes=len(genres))
    if name == "crnn":
        return CRNN(cfg, n_classes=len(genres))
    if name == "ast":
        from .ast_finetune import build_ast_for_checkpoint, load_pretrained_ast

        return load_pretrained_ast(cfg, genres) if pretrained else build_ast_for_checkpoint(cfg, genres)
    raise ValueError(f"unknown model {name!r}")


__all__ = ["GenreCNN", "CRNN", "build_model"]
