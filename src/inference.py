"""Predict genres for the test mashups and write a submission CSV.

    python -m src.inference --checkpoint checkpoints/ast.pt --data-root ... --out submission.csv
    python -m src.inference --checkpoint checkpoints/cnn.pt --data-root ... --tta 1

Test-time augmentation: each clip is longer than the 5 s training window, so we
average softmax outputs over ``--tta`` random 5 s crops instead of betting on one.
"""

from __future__ import annotations

import argparse
import random
from typing import Callable, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from .config import GENRES, MODEL_CONFIGS, BaselineConfig, DataLayout, SplitConfig
from .dataset import MashupDataset
from .engine import Forward, default_forward
from .models import build_model
from .preprocessing import crop_or_pad
from .train import make_featurizer, resolve_device


def load_torch_model(path: str, device: torch.device):
    """Rebuild the architecture named in the checkpoint and load its weights."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    name = ckpt["model"]
    cfg = MODEL_CONFIGS[name]
    model = build_model(name, cfg, pretrained=False)
    model.load_state_dict(ckpt["state"])
    return name, cfg, model.to(device).eval(), ckpt


def forward_for(name: str) -> Forward:
    if name == "ast":
        from .models.ast_finetune import ast_forward

        return ast_forward
    return default_forward


@torch.no_grad()
def predict_proba_tta(model: nn.Module, wav: np.ndarray, featurize: Callable,
                      forward: Forward, n_samples: int, n_tta: int, add_channel_dim: bool,
                      device: torch.device, rng: random.Random) -> np.ndarray:
    """Mean softmax over ``n_tta`` random crops, run as one batch."""
    feats = np.stack([featurize(crop_or_pad(wav, n_samples, rng)) for _ in range(n_tta)])
    x = torch.from_numpy(feats)
    if add_channel_dim:
        x = x.unsqueeze(1)
    return F.softmax(forward(model, x.to(device)), dim=1).mean(0).cpu().numpy()


def run(args: argparse.Namespace) -> pd.DataFrame:
    layout = DataLayout()
    if args.checkpoint.endswith(".joblib"):
        import joblib

        from .models.baseline import predict_waveform

        bundle = joblib.load(args.checkpoint)
        cfg = BaselineConfig()
        ds = MashupDataset(args.data_root, layout, cfg.sample_rate)
        preds: List[int] = [predict_waveform(bundle["estimator"], ds[i][1], cfg)
                            for i in tqdm(range(len(ds)), desc="baseline")]
        return pd.DataFrame({"id": ds.ids, "genre": [GENRES[p] for p in preds]})

    device = resolve_device(args.device)
    name, cfg, model, _ = load_torch_model(args.checkpoint, device)
    n_tta = args.tta if args.tta is not None else cfg.n_tta
    featurize = make_featurizer(name, cfg)
    forward = forward_for(name)
    rng = random.Random(args.seed)
    ds = MashupDataset(args.data_root, layout, cfg.audio.sample_rate)

    # A missing or unreadable test file raises: the original notebooks silently
    # predicted "pop" in that case, which would hide a broken data path.
    preds = []
    for i in tqdm(range(len(ds)), desc=f"{name} TTAx{n_tta}"):
        _, wav = ds[i]
        p = predict_proba_tta(model, wav, featurize, forward, cfg.audio.samples, n_tta,
                              add_channel_dim=name != "ast", device=device, rng=rng)
        preds.append(GENRES[int(np.argmax(p))])
    return pd.DataFrame({"id": ds.ids, "genre": preds})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help=".pt from train.py or baseline.joblib")
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--tta", type=int, help="random crops per clip (default: model config)")
    p.add_argument("--seed", type=int, default=SplitConfig().seed)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sub = run(args)
    sub.to_csv(args.out, index=False)
    print(f"Wrote {len(sub)} predictions to {args.out}")
    print(sub["genre"].value_counts().to_string())


if __name__ == "__main__":
    main()
