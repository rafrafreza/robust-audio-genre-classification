"""Phase-1 freezing must leave only the classification head trainable.

Uses a tiny randomly initialised AST so the test runs offline in seconds; the
freezing logic depends only on parameter names, which match the pretrained model.
"""

import pytest
import torch

transformers = pytest.importorskip("transformers")

from src.models.ast_finetune import ast_forward, freeze_encoder, unfreeze_all  # noqa: E402


def _tiny_ast(n_labels: int = 10):
    cfg = transformers.ASTConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                                 intermediate_size=64, num_mel_bins=128, max_length=128,
                                 num_labels=n_labels)
    return transformers.ASTForAudioClassification(cfg)


def test_freeze_then_unfreeze():
    model = _tiny_ast()
    total = sum(p.numel() for p in model.parameters())

    trainable = freeze_encoder(model, "classifier")
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert names and all(n.startswith("classifier.") for n in names)
    assert trainable == 32 * 2 + (32 * 10 + 10)       # head LayerNorm + Linear

    logits = ast_forward(model, torch.randn(2, 128, 128))
    assert logits.shape == (2, 10)
    logits.sum().backward()
    assert all(p.grad is None for n, p in model.named_parameters() if not n.startswith("classifier."))

    assert unfreeze_all(model) == total
    assert all(p.requires_grad for p in model.parameters())
