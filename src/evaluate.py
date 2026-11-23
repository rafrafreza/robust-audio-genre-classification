"""Rebuild the results table from trained checkpoints.

    python -m src.evaluate --data-root ... --checkpoints checkpoints/*.pt checkpoints/baseline.joblib

Validation F1 is recomputed on a freshly simulated validation set drawn from the
held-out songs (same song split, same seed). The mashups are not the exact ones
from the original notebooks, so expect small differences from the recorded values.

Leaderboard scores cannot be recomputed: the test labels are hidden. They are
copied below from the technical report and shown alongside for reference.
"""

from __future__ import annotations

import argparse
import random
from typing import Dict, List

import joblib
import torch
from torch.utils.data import DataLoader

from .config import GENRES, DataLayout, MashupConfig, SplitConfig
from .dataset import GenreDataset, load_sources, simulate_examples
from .engine import evaluate
from .inference import forward_for, load_torch_model
from .train import make_featurizer, resolve_device

# Kaggle leaderboard macro F1 as recorded in the technical report (§6.2).
REPORTED_LEADERBOARD: Dict[str, float] = {
    "random": 0.09420,
    "baseline": 0.35802,
    "cnn": 0.80789,
    "crnn": 0.83917,
    "ast": 0.88965,
}


def eval_checkpoint(path: str, data_root: str, device: torch.device, seed: int) -> Dict:
    if path.endswith(".joblib"):
        bundle = joblib.load(path)
        return {"model": "baseline", "params": "-", "val_f1": None,
                "ckpt_f1": bundle["f1"], "note": "val split leaks songs; see models/baseline.py"}

    name, cfg, model, ckpt = load_torch_model(path, device)
    sources = load_sources(data_root, DataLayout(), GENRES, SplitConfig(seed=seed))
    X, y = simulate_examples(sources.val_songs, GENRES, lambda g: cfg.n_val, sources,
                             make_featurizer(name, cfg), cfg.audio, MashupConfig(),
                             random.Random(seed), "val")
    dl = DataLoader(GenreDataset(X, y, None, add_channel_dim=name != "ast"),
                    cfg.batch_size, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    res = evaluate(model, dl, criterion, device, forward_for(name))
    params = sum(p.numel() for p in model.parameters())
    return {"model": name, "params": f"{params / 1e6:.2f}M", "val_f1": res.macro_f1,
            "ckpt_f1": ckpt["f1"], "note": ""}


def render(rows: List[Dict]) -> str:
    lines = ["| Model | Params | Val F1 (recomputed) | Val F1 (at training) | Leaderboard F1 |",
             "|---|---|---|---|---|"]
    for r in rows:
        val = "-" if r["val_f1"] is None else f"{r['val_f1']:.4f}"
        lb = REPORTED_LEADERBOARD.get(r["model"])
        lines.append(f"| {r['model']} | {r['params']} | {val} | {r['ckpt_f1']:.4f} | "
                     f"{'-' if lb is None else f'{lb:.5f}'} |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--seed", type=int, default=SplitConfig().seed)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = resolve_device(args.device)
    rows = [eval_checkpoint(c, args.data_root, device, args.seed) for c in args.checkpoints]
    print(render(rows))
    for r in rows:
        if r["note"]:
            print(f"note ({r['model']}): {r['note']}")


if __name__ == "__main__":
    main()
