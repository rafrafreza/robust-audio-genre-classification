"""Epoch loops shared by every PyTorch model.

Models differ only in how a batch is passed to them (HF's AST takes
``input_values=``), so that is injected as a ``forward`` callable instead of
duplicating the loop per model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

Forward = Callable[[nn.Module, torch.Tensor], torch.Tensor]


def default_forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return model(x)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    criterion: nn.Module, device: torch.device, grad_clip: float,
                    forward: Forward = default_forward) -> Tuple[float, float]:
    """Returns (mean loss, macro F1) over the epoch."""
    model.train()
    loss_sum, preds, labels = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = forward(model, x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        loss_sum += loss.item()
        preds.extend(logits.argmax(1).cpu().numpy())
        labels.extend(y.cpu().numpy())
    return loss_sum / len(loader), f1_score(labels, preds, average="macro")


@dataclass
class EvalResult:
    loss: float
    macro_f1: float
    preds: np.ndarray
    labels: np.ndarray


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
             device: torch.device, forward: Forward = default_forward) -> EvalResult:
    model.eval()
    loss_sum, preds, labels = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = forward(model, x)
        loss_sum += criterion(logits, y).item()
        preds.extend(logits.argmax(1).cpu().numpy())
        labels.extend(y.cpu().numpy())
    preds_a, labels_a = np.asarray(preds), np.asarray(labels)
    return EvalResult(loss_sum / len(loader),
                      f1_score(labels_a, preds_a, average="macro"), preds_a, labels_a)


@dataclass
class FitState:
    """Carried across phases so AST phase 2 only checkpoints if it beats phase 1."""

    best_f1: float = 0.0
    best_epoch: int = 0
    epochs_done: int = 0
    history: List[Dict[str, float]] = field(default_factory=list)


def fit(model: nn.Module, train_dl: DataLoader, val_dl: DataLoader,
        optimizer: torch.optim.Optimizer, scheduler: Any, epochs: int,
        criterion: nn.Module, device: torch.device, grad_clip: float, ckpt_path: str,
        ckpt_meta: Dict[str, Any], forward: Forward = default_forward,
        state: Optional[FitState] = None, tag: str = "") -> FitState:
    """Train for ``epochs``; checkpoint whenever validation macro F1 improves.

    Selection is on macro F1, not loss: the leaderboard metric is macro F1, and with
    label smoothing the loss stays high even when predictions are confident and right.
    """
    state = state or FitState()
    for _ in range(epochs):
        t0 = time.time()
        tl, tf = train_one_epoch(model, train_dl, optimizer, criterion, device, grad_clip, forward)
        ev = evaluate(model, val_dl, criterion, device, forward)
        scheduler.step()
        state.epochs_done += 1
        ep = state.epochs_done
        state.history.append({"epoch": ep, "train_loss": tl, "train_f1": tf,
                              "val_loss": ev.loss, "val_f1": ev.macro_f1})
        flag = ""
        if ev.macro_f1 > state.best_f1:
            state.best_f1, state.best_epoch = ev.macro_f1, ep
            torch.save({"ep": ep, "state": model.state_dict(), "f1": ev.macro_f1, **ckpt_meta},
                       ckpt_path)
            flag = " *"
        print(f"{tag}Ep {ep:02d} | train {tl:.4f}/{tf:.4f} | val {ev.loss:.4f}/{ev.macro_f1:.4f}"
              f" | {time.time() - t0:.0f}s{flag}")
    return state
