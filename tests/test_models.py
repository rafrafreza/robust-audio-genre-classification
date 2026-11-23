"""Shape and size checks for the scratch models.

The parameter counts are asserted exactly because the README quotes them: if the
architecture drifts from what produced the reported scores, this fails.
"""

import torch

from src.config import CNNConfig, CRNNConfig
from src.models import CRNN, GenreCNN

AUDIO = CNNConfig().audio
BATCH = (2, 1, AUDIO.n_mels, AUDIO.spec_width)  # (B, 1, 128, 216)


def _n_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def test_cnn_forward_shape_and_size():
    model = GenreCNN(CNNConfig()).eval()
    with torch.no_grad():
        out = model(torch.randn(BATCH))
    assert out.shape == (2, 10)
    assert _n_params(model.features) == 2_353_312   # backbone, matches notebook 04 printout
    assert _n_params(model) == 2_454_058


def test_crnn_forward_shape_and_size():
    model = CRNN(CRNNConfig()).eval()
    with torch.no_grad():
        feats = model.cnn(torch.randn(BATCH))
        out = model(torch.randn(BATCH))
    assert feats.shape == (2, 256, 8, 13)            # 13 time steps fed to the BiGRU
    assert out.shape == (2, 10)
    assert _n_params(model) == 2_750_507              # notebook 04 printout
