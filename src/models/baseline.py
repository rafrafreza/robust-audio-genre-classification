"""MFCC (60 coeffs, mean + std = 120-d) + logistic regression (notebook 02).

Reproduces the notebook as it ran, including two known weaknesses that are kept
on purpose so the reported leaderboard score stays reproducible:

* Mashups are full-length and noise-free, with the first stem at full gain and the
  rest at 0.25. The test-like recipe in ``preprocessing.make_mashup`` came later.
* The validation split is over *mashups*, not songs. Every mashup draws from all 100
  songs of a genre, so validation mashups share stems with training ones. That is why
  the notebook reports val macro F1 0.714 against a leaderboard score of 0.358.
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Sequence, Tuple

import librosa
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ..config import BaselineConfig
from ..preprocessing import mfcc_features


def make_baseline_mashup(songs: Sequence[str], cfg: BaselineConfig,
                         rng: random.Random) -> np.ndarray:
    """Mix one stem from each of 4 distinct songs (``rng.sample``), notebook-02 style."""
    selected = rng.sample(list(songs), len(cfg.stem_order))
    mixed = None
    for song, stem in zip(selected, cfg.stem_order):
        try:
            y, _ = librosa.load(os.path.join(song, stem), sr=cfg.sample_rate)
        except Exception:  # noqa: BLE001 - missing stems are skipped, as in the notebook
            continue
        if mixed is None:
            mixed = y
        else:
            n = min(len(mixed), len(y))
            mixed = mixed[:n] + y[:n] * cfg.secondary_stem_gain
    if mixed is None:
        mixed = np.zeros(cfg.sample_rate * cfg.fallback_duration_s, np.float32)
    return mixed


def build_features(catalog: Dict[str, List[str]], genres: Sequence[str],
                   cfg: BaselineConfig, rng: random.Random) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for label, g in enumerate(genres):
        for _ in tqdm(range(cfg.mashups_per_genre), desc=f"MFCC {g}", leave=False):
            wav = make_baseline_mashup(catalog[g], cfg, rng)
            xs.append(mfcc_features(wav, cfg.sample_rate, cfg.n_mfcc))
            ys.append(label)
    return np.asarray(xs), np.asarray(ys)


def train_baseline(X: np.ndarray, y: np.ndarray, cfg: BaselineConfig,
                   seed: int) -> Tuple[LogisticRegression, float]:
    """Fit on an 80% stratified split; return the model and (leaky) val macro F1."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=cfg.test_size, random_state=seed, stratify=y)
    # saga + 3000 iterations still raised a ConvergenceWarning in the original run.
    # The MFCC statistics are not standardised, which is a likely cause (untested).
    clf = LogisticRegression(max_iter=cfg.max_iter, solver=cfg.solver,
                             class_weight=cfg.class_weight, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    return clf, f1_score(y_val, clf.predict(X_val), average="macro")


def predict_waveform(clf: LogisticRegression, wav: np.ndarray, cfg: BaselineConfig) -> int:
    """Whole test clip -> class index (no cropping, no TTA, as in the notebook)."""
    feats = mfcc_features(wav, cfg.sample_rate, cfg.n_mfcc)
    return int(clf.predict(feats[None, :])[0])
