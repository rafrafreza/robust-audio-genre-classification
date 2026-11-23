"""All tunable values live here, one frozen dataclass per model.

Values are taken from the technical report's hyperparameter table and cross-checked
against the notebooks that produced the submitted predictions. Where the two
disagreed, the notebook value is used (it is what actually ran) and the
discrepancy is noted inline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Union

GENRES: Tuple[str, ...] = (
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
)


@dataclass(frozen=True)
class DataLayout:
    """Directory layout of the Messy Mashup dataset, relative to ``--data-root``."""

    genres_dir: str = "genres_stems"
    noise_dir: str = "ESC-50-master/audio"
    test_csv: str = "test.csv"
    audio_ext: str = ".wav"
    test_id_col: str = "id"
    test_path_col: str = "filename"


@dataclass(frozen=True)
class AudioConfig:
    """Waveform and log-mel settings.

    Training crops are short (5 s) so that many distinct mashups fit in RAM; the
    mel settings give a 128 x 216 image, which the CNN treats like a 1-channel photo.
    """

    sample_rate: int = 22050
    duration_s: int = 5
    # Stems are decoded only for duration_s + margin seconds before cropping. This
    # keeps decoding fast but means training crops come from the first 7 s of each
    # 30 s stem. Kept as-is because it is what produced the reported scores.
    stem_load_margin_s: int = 2
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    fmin: int = 20
    fmax: int = 8000

    @property
    def samples(self) -> int:
        return self.sample_rate * self.duration_s

    @property
    def spec_width(self) -> int:
        # librosa's centred STFT yields floor(samples / hop) + 1 frames.
        return self.samples // self.hop_length + 1


@dataclass(frozen=True)
class MashupConfig:
    """How a training example is synthesised to look like a test mashup.

    The test set is built from stems of different songs of the same genre plus
    ESC-50 noise at 0.05-0.35 relative amplitude, so training reproduces that recipe.
    """

    stems_per_mashup: int = 4
    stem_weight_range: Tuple[float, float] = (0.4, 1.0)
    noise_range: Tuple[float, float] = (0.05, 0.35)
    peak_eps: float = 1e-6


@dataclass(frozen=True)
class SpecAugmentConfig:
    freq_mask_param: int = 20
    time_mask_param: int = 40
    # Notebooks apply each mask when random.random() > 0.4, i.e. with p = 0.6.
    apply_prob: float = 0.6


@dataclass(frozen=True)
class SplitConfig:
    """Train/val split is done by *song*, so no stem of a validation song is ever
    mixed into a training example."""

    val_fraction: float = 0.15
    min_val_songs: int = 2
    seed: int = 42


@dataclass(frozen=True)
class BaselineConfig:
    """MFCC + logistic regression (notebook 02)."""

    sample_rate: int = 22050
    n_mfcc: int = 60
    mashups_per_genre: int = 60
    test_size: float = 0.2
    max_iter: int = 3000
    solver: str = "saga"
    class_weight: str = "balanced"
    # Notebook 02 mixes stems in this fixed order: the first stem at full gain and
    # every later stem at 0.25. It does not use the noise/weighting recipe in
    # MashupConfig; that recipe was introduced with the CNN.
    stem_order: Tuple[str, ...] = ("drums.wav", "bass.wav", "vocals.wav", "other.wav")
    secondary_stem_gain: float = 0.25
    fallback_duration_s: int = 5


@dataclass(frozen=True)
class CNNConfig:
    """5-block scratch CNN (notebook 03)."""

    channels: Tuple[int, ...] = (32, 64, 128, 256, 256)
    pool: Tuple[bool, ...] = (True, True, True, True, False)
    head_dims: Tuple[int, ...] = (256, 128)
    # Report lists "Dropout 0.4"; the notebook uses 0.4 then 0.2 in the two hidden layers.
    head_dropouts: Tuple[float, ...] = (0.4, 0.2)
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    eta_min: float = 1e-6
    label_smoothing: float = 0.1
    # Not mentioned in the report for this model, but present in the notebook.
    grad_clip: float = 1.0
    n_aug: int = 6          # simulated mashups per training song
    n_val: int = 30         # validation mashups per genre
    n_tta: int = 10         # random crops averaged at inference
    num_workers: int = 2
    audio: AudioConfig = field(default_factory=AudioConfig)
    spec_augment: SpecAugmentConfig = field(default_factory=SpecAugmentConfig)


@dataclass(frozen=True)
class CRNNConfig:
    """CNN backbone + BiGRU + attention pooling (notebook 04)."""

    channels: Tuple[int, ...] = (32, 64, 128, 256, 256)
    pool: Tuple[bool, ...] = (True, True, True, True, False)
    gru_hidden: int = 128
    gru_layers: int = 1
    head_dims: Tuple[int, ...] = (256, 128)
    head_dropouts: Tuple[float, ...] = (0.5, 0.25)
    batch_size: int = 64
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 2e-4
    eta_min: float = 1e-6
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    n_aug: int = 12
    n_val: int = 200
    n_tta: int = 10
    num_workers: int = 2
    audio: AudioConfig = field(default_factory=AudioConfig)
    spec_augment: SpecAugmentConfig = field(default_factory=SpecAugmentConfig)


@dataclass(frozen=True)
class ASTConfig:
    """Two-phase fine-tuning of the AudioSet-pretrained AST (notebook 05)."""

    model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    # Parameters whose name contains this substring stay trainable in phase 1.
    # For HF's ASTForAudioClassification that is the LayerNorm + Linear head.
    head_param_key: str = "classifier"
    batch_size: int = 16
    epochs_head: int = 3
    epochs_full: int = 2
    lr_head: float = 1e-3
    lr_full: float = 5e-5
    weight_decay: float = 1e-4
    eta_min_head: float = 1e-5
    eta_min_full: float = 1e-7
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    n_aug: int = 8
    n_val: int = 100
    n_tta: int = 8
    num_workers: int = 2
    # AST was pretrained on 16 kHz audio; its feature extractor expects that rate.
    audio: AudioConfig = field(default_factory=lambda: AudioConfig(sample_rate=16000))


ModelConfig = Union[CNNConfig, CRNNConfig, ASTConfig]

MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "cnn": CNNConfig(),
    "crnn": CRNNConfig(),
    "ast": ASTConfig(),
}

CHECKPOINT_NAMES: Dict[str, str] = {
    "baseline": "baseline.joblib",
    "cnn": "cnn.pt",
    "crnn": "crnn.pt",
    "ast": "ast.pt",
}
