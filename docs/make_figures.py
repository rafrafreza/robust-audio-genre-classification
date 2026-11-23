"""Regenerate the README figures from recorded results.

    python docs/make_figures.py

Every number below is copied from a saved notebook output or the technical report;
the source is noted next to each block. Nothing is recomputed here.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "figures")

# Palette: validated categorical slots 1-2 + blue sequential ramp (light mode).
SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e8e7e3"
BLUE, ORANGE, BLUE_LIGHT = "#2a78d6", "#eb6834", "#86b6ef"
GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]

# Kaggle leaderboard macro F1 -- technical report §6.2.
LEADERBOARD = [
    ("Random baseline", 0.09420),
    ("MFCC + logistic regression", 0.35802),
    ("Scratch CNN", 0.80789),
    ("CRNN (BiGRU + attention)", 0.83917),
    ("AST, two-phase fine-tune", 0.88965),
]

# AST per-epoch macro F1 -- notebook 05, cells 13-14 (P1 epochs 1-3, P2 epochs 4-5).
AST_TRAIN_F1 = [0.7968, 0.8933, 0.9265, 0.8593, 0.9850]
AST_VAL_F1 = [0.8259, 0.8350, 0.8522, 0.8061, 0.8621]
EPOCHS_HEAD = 3

# AST confusion matrix on 1,000 validation mashups (100 per genre) -- notebook 05,
# cell 16 figure. Rows = true genre. Cross-checked against the printed per-class
# precision/recall in the same cell.
AST_CONFUSION = np.array([
    [89, 0, 5, 0, 0, 3, 2, 0, 0, 1],
    [0, 84, 0, 0, 0, 15, 0, 0, 0, 1],
    [1, 1, 90, 2, 0, 0, 0, 0, 1, 5],
    [1, 0, 1, 83, 2, 0, 0, 3, 1, 9],
    [0, 0, 0, 0, 96, 0, 1, 2, 0, 1],
    [0, 1, 1, 0, 0, 98, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 100, 0, 0, 0],
    [0, 0, 3, 3, 13, 0, 0, 78, 1, 2],
    [5, 0, 8, 2, 0, 0, 0, 0, 77, 8],
    [0, 0, 24, 0, 0, 0, 3, 5, 1, 67],
])


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=10, length=0)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def leaderboard_chart() -> None:
    names = [n for n, _ in LEADERBOARD][::-1]
    scores = [s for _, s in LEADERBOARD][::-1]
    fig, ax = plt.subplots(figsize=(8, 3.2), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    # Bars are drawn bottom-up, so index 0 is the last model (AST): highlight it.
    colors = [BLUE if i == 0 else BLUE_LIGHT for i in range(len(names))]
    ax.barh(names, scores, color=colors, height=0.62)
    for y, s in enumerate(scores):
        ax.text(s + 0.012, y, f"{s:.3f}", va="center", fontsize=10, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Kaggle macro F1 (hidden test set, 3,020 mashups)", color=INK_2, fontsize=10)
    ax.set_title("Each model on the noisy mashup test set", loc="left", color=INK,
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "leaderboard.png"), dpi=200)
    plt.close(fig)


def ast_phases_chart() -> None:
    epochs = np.arange(1, len(AST_VAL_F1) + 1)
    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=SURFACE)
    _style(ax)
    ax.axvspan(0.6, EPOCHS_HEAD + 0.5, color="#f0efec", zorder=0)
    ax.text(0.7, 0.995, "Phase 1: head only, lr 1e-3", color=INK_2, fontsize=9, va="top")
    ax.text(EPOCHS_HEAD + 0.6, 0.995, "Phase 2: all layers, lr 5e-5", color=INK_2,
            fontsize=9, va="top")
    ax.plot(epochs, AST_TRAIN_F1, color=ORANGE, linewidth=2, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=2)
    ax.plot(epochs, AST_VAL_F1, color=BLUE, linewidth=2, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=2)
    ax.text(5.08, AST_TRAIN_F1[-1], "train", color=INK_2, fontsize=10, va="center")
    ax.text(5.08, AST_VAL_F1[-1], f"validation {AST_VAL_F1[-1]:.3f}", color=INK_2,
            fontsize=10, va="center")
    ax.annotate("dip after unfreezing", xy=(4, AST_VAL_F1[3]), xytext=(3.55, 0.775),
                fontsize=9, color=INK_2,
                arrowprops=dict(arrowstyle="-", color=INK_3, linewidth=0.8))
    ax.set_xticks(epochs)
    ax.set_xlim(0.6, 5.9)
    ax.set_ylim(0.76, 1.0)
    ax.set_xlabel("Epoch", color=INK_2, fontsize=10)
    ax.set_ylabel("Macro F1", color=INK_2, fontsize=10)
    ax.set_title("AST two-phase fine-tuning", loc="left", color=INK, fontsize=12,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ast_two_phase.png"), dpi=200)
    plt.close(fig)


def confusion_chart() -> None:
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "blue_seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    pct = AST_CONFUSION / AST_CONFUSION.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(6.6, 5.8), facecolor=SURFACE)
    ax.imshow(pct, cmap=cmap, vmin=0, vmax=100)
    for i in range(len(GENRES)):
        for j in range(len(GENRES)):
            v = AST_CONFUSION[i, j]
            if v == 0:
                continue
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if pct[i, j] > 55 else INK)
    ax.set_xticks(range(len(GENRES)), GENRES, rotation=45, ha="right")
    ax.set_yticks(range(len(GENRES)), GENRES)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 10), minor=True)
    ax.set_yticks(np.arange(-0.5, 10), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("Predicted", color=INK_2, fontsize=10)
    ax.set_ylabel("True genre", color=INK_2, fontsize=10)
    ax.set_title("AST validation confusion matrix (100 mashups per genre)", loc="left",
                 color=INK, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ast_confusion.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    assert (AST_CONFUSION.sum(axis=1) == 100).all()
    assert np.allclose(np.diag(AST_CONFUSION) / 100,
                       [0.89, 0.84, 0.90, 0.83, 0.96, 0.98, 1.00, 0.78, 0.77, 0.67])
    leaderboard_chart()
    ast_phases_chart()
    confusion_chart()
    print(f"wrote figures to {OUT}")
