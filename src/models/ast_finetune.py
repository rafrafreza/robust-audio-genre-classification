"""Two-phase fine-tuning of the AudioSet-pretrained Audio Spectrogram Transformer.

Model: ``MIT/ast-finetuned-audioset-10-10-0.4593`` (86,196,490 parameters with a
10-class head). Its 527-class AudioSet head is replaced by a randomly initialised
10-class head.

Why two phases
--------------
A freshly initialised head outputs near-random logits, so the first gradients that
flow back through it are large and uninformative. If the encoder is trainable at
that point, those gradients overwrite pretrained AudioSet features before the head
has learned anything useful (catastrophic forgetting).

* Phase 1 (3 epochs, lr 1e-3): encoder frozen; only the head (LayerNorm + Linear,
  9,226 parameters) trains. The head learns to read genre from fixed AudioSet features.
* Phase 2 (2 epochs, lr 5e-5, i.e. 20x lower): everything unfrozen. Gradients now come
  through a head that already makes sensible predictions, so the encoder is nudged
  rather than rewritten.

Observed in the original run (notebook 05): val macro F1 rose 0.826 -> 0.835 -> 0.852
in phase 1, dipped to 0.806 on the first full fine-tuning epoch, then reached 0.862.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..config import ASTConfig
from ..engine import FitState, fit


def load_feature_extractor(cfg: ASTConfig):
    """The extractor that matches pretraining: 128-bin log-mel, AudioSet mean/std,
    padded to 1024 frames. Using any other front-end would shift the input distribution
    the encoder was trained on."""
    from transformers import AutoFeatureExtractor

    return AutoFeatureExtractor.from_pretrained(cfg.model_name)


class ASTFeaturizer:
    """Waveform (16 kHz) -> ``(1024, 128)`` AST input.

    Note: a 5 s clip is ~498 frames, so roughly half of every input is padding. This
    is how the reported model was trained; longer crops are an obvious improvement.
    """

    def __init__(self, cfg: ASTConfig, extractor=None) -> None:
        self.cfg = cfg
        self.extractor = extractor or load_feature_extractor(cfg)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        out = self.extractor(audio, sampling_rate=self.cfg.audio.sample_rate,
                             return_tensors="pt", padding=True)
        return out["input_values"].squeeze(0).numpy()


def _label_maps(genres: Sequence[str]):
    g2i = {g: i for i, g in enumerate(genres)}
    return g2i, {i: g for g, i in g2i.items()}


def load_pretrained_ast(cfg: ASTConfig, genres: Sequence[str]) -> nn.Module:
    """Pretrained encoder + new head sized for ``genres``."""
    from transformers import ASTForAudioClassification

    g2i, i2g = _label_maps(genres)
    return ASTForAudioClassification.from_pretrained(
        cfg.model_name, num_labels=len(genres), ignore_mismatched_sizes=True,
        label2id=g2i, id2label=i2g,
    )


def build_ast_for_checkpoint(cfg: ASTConfig, genres: Sequence[str]) -> nn.Module:
    """Architecture only (no pretrained download of weights); used before loading a
    fine-tuned state dict at inference time."""
    from transformers import ASTForAudioClassification, AutoConfig

    g2i, i2g = _label_maps(genres)
    hf_cfg = AutoConfig.from_pretrained(cfg.model_name, num_labels=len(genres),
                                        label2id=g2i, id2label=i2g)
    return ASTForAudioClassification(hf_cfg)


def ast_forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return model(input_values=x).logits


def freeze_encoder(model: nn.Module, head_param_key: str = "classifier") -> int:
    """Freeze everything except parameters whose name contains ``head_param_key``.
    Returns the number of trainable parameters."""
    for name, p in model.named_parameters():
        p.requires_grad = head_param_key in name
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def unfreeze_all(model: nn.Module) -> int:
    for p in model.parameters():
        p.requires_grad = True
    return sum(p.numel() for p in model.parameters())


def two_phase_finetune(model: nn.Module, train_dl: DataLoader, val_dl: DataLoader,
                       cfg: ASTConfig, device: torch.device, ckpt_path: str,
                       ckpt_meta: Dict[str, Any]) -> FitState:
    """Run phase 1 (head only) then phase 2 (full). Phase 2 starts from the *final*
    phase-1 weights, and the checkpoint keeps the best epoch across both phases."""
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    n = freeze_encoder(model, cfg.head_param_key)
    print(f"Phase 1: head only ({n:,} trainable params), lr={cfg.lr_head}")
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=cfg.lr_head, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs_head,
                                                       eta_min=cfg.eta_min_head)
    state = fit(model, train_dl, val_dl, opt, sched, cfg.epochs_head, criterion, device,
                cfg.grad_clip, ckpt_path, ckpt_meta, forward=ast_forward, tag="P1 ")

    n = unfreeze_all(model)
    print(f"Phase 2: full fine-tune ({n:,} trainable params), lr={cfg.lr_full}")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_full, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs_full,
                                                       eta_min=cfg.eta_min_full)
    return fit(model, train_dl, val_dl, opt, sched, cfg.epochs_full, criterion, device,
               cfg.grad_clip, ckpt_path, ckpt_meta, forward=ast_forward, state=state,
               tag="P2 ")
