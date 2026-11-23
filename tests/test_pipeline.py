"""End-to-end augmentation pipeline on a synthetic dataset laid out like the real one:
song split -> mashup simulation -> log-mel -> Dataset with SpecAugment."""

import os
import random

import numpy as np
import soundfile as sf

from src.config import GENRES, CNNConfig, DataLayout, MashupConfig, SplitConfig
from src.dataset import GenreDataset, load_sources, simulate_examples
from src.preprocessing import SpecAugment, mel_spectrogram

SR = CNNConfig().audio.sample_rate
STEMS = ("drums.wav", "bass.wav", "vocals.wav", "other.wav")


def _write_fake_dataset(root, genres, songs_per_genre=4, stem_s=8, noise_s=5):
    rng = np.random.default_rng(0)
    layout = DataLayout()
    for g in genres:
        for s in range(songs_per_genre):
            d = os.path.join(root, layout.genres_dir, g, f"{g}.{s:05d}")
            os.makedirs(d)
            for stem in STEMS:
                sf.write(os.path.join(d, stem), 0.1 * rng.standard_normal(SR * stem_s), SR)
    nd = os.path.join(root, layout.noise_dir)
    os.makedirs(nd)
    for i in range(2):
        sf.write(os.path.join(nd, f"noise{i}.wav"), 0.1 * rng.standard_normal(SR * noise_s), SR)


def test_mashup_pipeline_produces_model_ready_spectrograms(tmp_path):
    genres = GENRES[:2]
    _write_fake_dataset(str(tmp_path), genres)
    cfg = CNNConfig()
    sources = load_sources(str(tmp_path), DataLayout(), genres, SplitConfig(seed=0))

    # Song-level split: no song may appear on both sides.
    for g in genres:
        assert not set(sources.train_songs[g]) & set(sources.val_songs[g])
    assert sources.stem_names == sorted(STEMS)

    X, y = simulate_examples(sources.train_songs, genres, lambda g: 2, sources,
                             lambda w: mel_spectrogram(w, cfg.audio), cfg.audio,
                             MashupConfig(), random.Random(0))
    assert X.shape == (4, 128, 216)
    assert X.min() >= 0.0 and X.max() <= 1.0
    assert list(y) == [0, 0, 1, 1]

    ds = GenreDataset(X, y, SpecAugment(cfg.spec_augment, random.Random(0)))
    x0, y0 = ds[0]
    assert tuple(x0.shape) == (1, 128, 216)
    assert int(y0) == 0
