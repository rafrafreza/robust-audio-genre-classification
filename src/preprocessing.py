"""Waveform -> training example: mashup simulation, noise injection, features.

The test set is built by mixing stems from *different* songs of one genre and adding
ESC-50 environmental noise. Training data is clean, per-song stems. Everything in
this module exists to make a training example look like a test example before the
model ever sees it.

Pure NumPy/librosa except ``SpecAugment``, which needs torchaudio and imports it lazily.
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Sequence

import librosa
import numpy as np

from .config import AudioConfig, MashupConfig, SpecAugmentConfig


def detect_stem_names(song_dir: str, audio_ext: str = ".wav") -> List[str]:
    """Read stem filenames from one song folder.

    The dataset is inconsistent ('other.wav' vs 'others.wav'), so names are read from
    disk instead of hardcoded. Note: like the original notebooks, this inspects a
    single song; a song using the other spelling would fail to load and fall back to
    silence in :func:`load_segment`.
    """
    return sorted(f for f in os.listdir(song_dir) if f.endswith(audio_ext))


def peak_normalize(y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Scale to unit peak so that mixing weights, not recording loudness, set balance."""
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    return y / (peak + eps) if peak > 0 else y


def crop_or_pad(y: np.ndarray, n_samples: int, rng: random.Random) -> np.ndarray:
    """Random fixed-length window; zero-pad short clips at the end."""
    if len(y) < n_samples:
        return np.pad(y, (0, n_samples - len(y))).astype(np.float32)
    start = rng.randint(0, len(y) - n_samples)
    return y[start:start + n_samples].astype(np.float32)


def load_segment(path: str, audio: AudioConfig, rng: random.Random) -> np.ndarray:
    """Load a random ``audio.duration_s`` window of a stem, resampled and mono.

    Unreadable files become silence rather than crashing a 20-minute precompute run;
    this mirrors the notebooks and is the reason the stem-name caveat above matters.
    """
    try:
        y, _ = librosa.load(
            path, sr=audio.sample_rate, mono=True,
            duration=audio.duration_s + audio.stem_load_margin_s,
        )
    except Exception:  # noqa: BLE001 - deliberate: see docstring
        return np.zeros(audio.samples, np.float32)
    return crop_or_pad(y, audio.samples, rng)


def load_noise_segment(noise_files: Sequence[str], audio: AudioConfig,
                       rng: random.Random) -> np.ndarray:
    """Random ESC-50 clip, tiled if shorter than the training window."""
    try:
        y, _ = librosa.load(rng.choice(noise_files), sr=audio.sample_rate, mono=True)
        if len(y) < audio.samples:
            y = np.tile(y, audio.samples // len(y) + 1)
        start = rng.randint(0, len(y) - audio.samples)
        return y[start:start + audio.samples].astype(np.float32)
    except Exception:  # noqa: BLE001
        return np.zeros(audio.samples, np.float32)


def make_mashup(genre: str, song_pool: Dict[str, List[str]], stem_names: Sequence[str],
                noise_files: Sequence[str], audio: AudioConfig, mix: MashupConfig,
                rng: random.Random) -> np.ndarray:
    """Synthesise one test-like training example for ``genre``.

    1. Pick ``mix.stems_per_mashup`` songs of the genre.
    2. Shuffle which stem role (drums/bass/vocals/other) each song contributes, so the
       model cannot rely on any one song supplying a particular instrument.
    3. Peak-normalise each stem, then weight it by U(0.4, 1.0) to vary instrument balance.
    4. Add an ESC-50 clip at U(0.05, 0.35) relative amplitude, the test noise range.

    Songs are drawn with replacement (``rng.choices``), as in the notebooks that produced
    the reported scores. With 85 training songs per genre about 7% of mashups reuse a
    song; the test set uses four distinct songs. Switching to ``rng.sample`` would match
    the test recipe exactly but would no longer reproduce the trained checkpoints.
    """
    songs = rng.choices(song_pool[genre], k=mix.stems_per_mashup)
    stems = list(stem_names)
    rng.shuffle(stems)

    mixed = np.zeros(audio.samples, np.float32)
    for song, stem in zip(songs, stems[:mix.stems_per_mashup]):
        y = peak_normalize(load_segment(os.path.join(song, stem), audio, rng), mix.peak_eps)
        mixed += rng.uniform(*mix.stem_weight_range) * y
    mixed = peak_normalize(mixed, mix.peak_eps)

    mixed += rng.uniform(*mix.noise_range) * load_noise_segment(noise_files, audio, rng)
    return peak_normalize(mixed, mix.peak_eps).astype(np.float32)


def mel_spectrogram(y: np.ndarray, audio: AudioConfig) -> np.ndarray:
    """Log-mel image in [0, 1], shape ``(n_mels, spec_width)``.

    Per-example min-max scaling (rather than dataset statistics) removes the loudness
    differences that noise injection introduces, so the CNN sees spectral shape only.
    """
    mel = librosa.feature.melspectrogram(
        y=y, sr=audio.sample_rate, n_fft=audio.n_fft, hop_length=audio.hop_length,
        n_mels=audio.n_mels, fmin=audio.fmin, fmax=audio.fmax,
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-9)
    width = audio.spec_width
    if mel.shape[1] < width:
        mel = np.pad(mel, ((0, 0), (0, width - mel.shape[1])))
    return mel[:, :width].astype(np.float32)


def mfcc_features(y: np.ndarray, sample_rate: int, n_mfcc: int) -> np.ndarray:
    """Mean and std of each MFCC over time -> ``2 * n_mfcc`` vector.

    The std term is what lets a linear model see *variation* in timbre; the mean alone
    cannot separate, e.g., steady classical strings from rhythmic hip-hop.
    """
    y = librosa.util.normalize(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


class SpecAugment:
    """Random frequency and time masking on a ``(1, F, T)`` spectrogram tensor.

    Forces the model not to depend on any single band or moment, which is what noise
    bursts in the test mashups destroy.
    """

    def __init__(self, cfg: SpecAugmentConfig, rng: random.Random | None = None) -> None:
        """``rng`` defaults to the global ``random`` module so each DataLoader worker
        (re-seeded by PyTorch) draws independent masking decisions."""
        import torchaudio.transforms as T  # local import keeps this module torch-free

        self.cfg = cfg
        self.rng = rng if rng is not None else random
        self.freq_mask = T.FrequencyMasking(freq_mask_param=cfg.freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param=cfg.time_mask_param)

    def __call__(self, x):  # x: torch.Tensor
        if self.rng.random() < self.cfg.apply_prob:
            x = self.freq_mask(x)
        if self.rng.random() < self.cfg.apply_prob:
            x = self.time_mask(x)
        return x
