"""Train any of the four models from the raw dataset.

    python -m src.train --model ast --data-root /path/to/messy_mashup
    python -m src.train --model cnn --data-root ... --epochs 5      # quick run
    python -m src.train --model baseline --data-root ...

Writes ``<out-dir>/<model>.pt`` (or ``baseline.joblib``) holding the best-val-F1 weights.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import (CHECKPOINT_NAMES, GENRES, MODEL_CONFIGS, ASTConfig, BaselineConfig,
                     DataLayout, MashupConfig, SplitConfig)
from .dataset import GenreDataset, build_catalog, load_sources, simulate_examples
from .engine import fit
from .models import build_model
from .preprocessing import SpecAugment, mel_spectrogram


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def make_featurizer(model: str, cfg):
    """CNN/CRNN read our own log-mel; AST must use its pretraining front-end."""
    if model == "ast":
        from .models.ast_finetune import ASTFeaturizer

        return ASTFeaturizer(cfg)
    return functools.partial(mel_spectrogram, audio=cfg.audio)


def train_baseline(args: argparse.Namespace) -> None:
    import joblib

    from .models.baseline import build_features, train_baseline as fit_lr

    cfg = BaselineConfig()
    layout = DataLayout()
    catalog = build_catalog(os.path.join(args.data_root, layout.genres_dir), GENRES)
    X, y = build_features(catalog, GENRES, cfg, random.Random(args.seed))
    clf, val_f1 = fit_lr(X, y, cfg, args.seed)
    print(f"Baseline val macro F1 (mashup-level split, leaky): {val_f1:.4f}")
    path = os.path.join(args.out_dir, CHECKPOINT_NAMES["baseline"])
    joblib.dump({"model": "baseline", "estimator": clf, "f1": val_f1,
                 "config": dataclasses.asdict(cfg)}, path)
    print(f"Saved {path}")


def train_torch(args: argparse.Namespace) -> None:
    cfg = MODEL_CONFIGS[args.model]
    if args.epochs is not None:
        if args.model == "ast":
            raise SystemExit("For AST use --epochs-head / --epochs-full.")
        cfg = dataclasses.replace(cfg, epochs=args.epochs)
    if args.model == "ast":
        overrides = {k: v for k, v in (("epochs_head", args.epochs_head),
                                       ("epochs_full", args.epochs_full)) if v is not None}
        cfg = dataclasses.replace(cfg, **overrides)

    device = resolve_device(args.device)
    rng = random.Random(args.seed)
    sources = load_sources(args.data_root, DataLayout(), GENRES, SplitConfig(seed=args.seed))
    featurize = make_featurizer(args.model, cfg)
    mix = MashupConfig()

    X_tr, y_tr = simulate_examples(sources.train_songs, GENRES,
                                   lambda g: len(sources.train_songs[g]) * cfg.n_aug,
                                   sources, featurize, cfg.audio, mix, rng, "train")
    X_vl, y_vl = simulate_examples(sources.val_songs, GENRES, lambda g: cfg.n_val,
                                   sources, featurize, cfg.audio, mix, rng, "val")
    print(f"train {X_tr.shape} | val {X_vl.shape}")

    is_ast = isinstance(cfg, ASTConfig)
    # SpecAugment uses the global `random` module, which PyTorch re-seeds per
    # DataLoader worker; a shared Random instance would give every worker identical masks.
    spec_aug = None if is_ast else SpecAugment(cfg.spec_augment)
    pin = device.type == "cuda"
    workers = cfg.num_workers if args.num_workers is None else args.num_workers
    tr_dl = DataLoader(GenreDataset(X_tr, y_tr, spec_aug, add_channel_dim=not is_ast),
                       cfg.batch_size, shuffle=True, drop_last=True,
                       num_workers=workers, pin_memory=pin)
    vl_dl = DataLoader(GenreDataset(X_vl, y_vl, None, add_channel_dim=not is_ast),
                       cfg.batch_size, shuffle=False,
                       num_workers=workers, pin_memory=pin)

    model = build_model(args.model, cfg).to(device)
    print(f"{args.model}: {sum(p.numel() for p in model.parameters()):,} parameters on {device}")
    ckpt_path = os.path.join(args.out_dir, CHECKPOINT_NAMES[args.model])
    meta = {"model": args.model, "config": dataclasses.asdict(cfg)}

    if is_ast:
        from .models.ast_finetune import two_phase_finetune

        state = two_phase_finetune(model, tr_dl, vl_dl, cfg, device, ckpt_path, meta)
    else:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs,
                                                           eta_min=cfg.eta_min)
        state = fit(model, tr_dl, vl_dl, opt, sched, cfg.epochs, criterion, device,
                    cfg.grad_clip, ckpt_path, meta)
    print(f"Best val macro F1 {state.best_f1:.4f} at epoch {state.best_epoch} -> {ckpt_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=["baseline", "cnn", "crnn", "ast"])
    p.add_argument("--data-root", required=True,
                   help="folder containing genres_stems/, ESC-50-master/, mashups/, test.csv")
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--epochs", type=int, help="override epochs (cnn/crnn)")
    p.add_argument("--epochs-head", type=int, help="override AST phase-1 epochs")
    p.add_argument("--epochs-full", type=int, help="override AST phase-2 epochs")
    p.add_argument("--seed", type=int, default=SplitConfig().seed)
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, help="override DataLoader workers")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)
    if args.model == "baseline":
        train_baseline(args)
    else:
        train_torch(args)


if __name__ == "__main__":
    main()
