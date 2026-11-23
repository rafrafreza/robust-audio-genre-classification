"""Song catalogue, leakage-free splits, simulated-mashup generation, and Datasets.

Training examples are synthesised once and held in RAM. On a T4, generating
~10k mashups takes ~20 min, while an epoch over the cached spectrograms takes
~30 s, so regenerating per epoch would dominate training time.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .config import AudioConfig, DataLayout, MashupConfig, SplitConfig
from .preprocessing import SpecAugment, detect_stem_names, make_mashup

SongPool = Dict[str, List[str]]
Featurizer = Callable[[np.ndarray], np.ndarray]


def build_catalog(genres_dir: str, genres: Sequence[str]) -> SongPool:
    """Map genre -> sorted list of song directories."""
    catalog: SongPool = {}
    for g in genres:
        gdir = os.path.join(genres_dir, g)
        catalog[g] = sorted(
            os.path.join(gdir, s) for s in os.listdir(gdir)
            if os.path.isdir(os.path.join(gdir, s))
        )
    return catalog


def split_by_song(catalog: SongPool, split: SplitConfig) -> Tuple[SongPool, SongPool]:
    """Hold out whole songs per genre.

    Splitting mashups instead of songs leaks: a validation mashup would share stems
    with training mashups. The MFCC baseline made exactly that mistake (see
    ``models/baseline.py``), which inflated its validation score to 0.71 against
    a 0.36 leaderboard score.
    """
    rng = random.Random(split.seed)
    train, val = {}, {}
    for g, songs in catalog.items():
        shuffled = songs.copy()
        rng.shuffle(shuffled)
        n_val = max(split.min_val_songs, int(len(shuffled) * split.val_fraction))
        val[g], train[g] = shuffled[:n_val], shuffled[n_val:]
    return train, val


def list_noise_files(noise_dir: str, audio_ext: str) -> List[str]:
    return sorted(os.path.join(noise_dir, f) for f in os.listdir(noise_dir)
                  if f.endswith(audio_ext))


@dataclass
class MashupSources:
    """Everything ``make_mashup`` needs, resolved once from disk."""

    train_songs: SongPool
    val_songs: SongPool
    stem_names: List[str]
    noise_files: List[str]


def load_sources(data_root: str, layout: DataLayout, genres: Sequence[str],
                 split: SplitConfig) -> MashupSources:
    catalog = build_catalog(os.path.join(data_root, layout.genres_dir), genres)
    train, val = split_by_song(catalog, split)
    first_song = catalog[genres[0]][0]
    return MashupSources(
        train_songs=train,
        val_songs=val,
        stem_names=detect_stem_names(first_song, layout.audio_ext),
        noise_files=list_noise_files(os.path.join(data_root, layout.noise_dir),
                                     layout.audio_ext),
    )


def simulate_examples(pool: SongPool, genres: Sequence[str], count_for: Callable[[str], int],
                      sources: MashupSources, featurize: Featurizer, audio: AudioConfig,
                      mix: MashupConfig, rng: random.Random, desc: str = ""
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate ``count_for(genre)`` featurised mashups per genre from ``pool``."""
    xs, ys = [], []
    for label, g in enumerate(genres):
        for _ in tqdm(range(count_for(g)), desc=f"{desc} {g}", leave=False):
            wav = make_mashup(g, pool, sources.stem_names, sources.noise_files, audio, mix, rng)
            xs.append(featurize(wav))
            ys.append(label)
    return np.asarray(xs, np.float32), np.asarray(ys, np.int64)


class GenreDataset(Dataset):
    """Pre-computed features + labels, with optional SpecAugment on the fly.

    ``add_channel_dim`` gives CNN-style ``(1, F, T)`` inputs; AST wants ``(T, F)`` as-is.
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray,
                 spec_augment: Optional[SpecAugment] = None, add_channel_dim: bool = True):
        x = torch.from_numpy(features)
        self.x = x.unsqueeze(1) if add_channel_dim else x
        self.y = torch.from_numpy(labels)
        self.spec_augment = spec_augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        x = self.x[i]
        if self.spec_augment is not None:
            x = self.spec_augment(x.clone())
        return x, self.y[i]


class MashupDataset(Dataset):
    """Unlabelled test mashups listed in ``test.csv``; yields ``(id, full waveform)``.

    Whole clips are returned so inference can draw several random crops (TTA).
    """

    def __init__(self, data_root: str, layout: DataLayout, sample_rate: int):
        df = pd.read_csv(os.path.join(data_root, layout.test_csv))
        self.ids = df[layout.test_id_col].tolist()
        self.paths = [os.path.join(data_root, p) for p in df[layout.test_path_col]]
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> Tuple[int, np.ndarray]:
        y, _ = librosa.load(self.paths[i], sr=self.sample_rate, mono=True)
        return self.ids[i], y.astype(np.float32)
