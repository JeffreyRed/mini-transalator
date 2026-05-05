"""
visualize.py — Alignment heatmaps, BLEU/perplexity curves, vocabulary plot.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import List

PALETTE = {
    "bg":        "#0d1117",
    "grid":      "#21262d",
    "text":      "#e6edf3",
    "accent":    "#58a6ff",
    "highlight": "#f78166",
    "muted":     "#8b949e",
    "green":     "#3fb950",
    "yellow":    "#e3b341",
    "purple":    "#bc8cff",
}


# ── Alignment heatmap ─────────────────────────────────────────────────────────

def plot_alignment(
    cross_weights  : "torch.Tensor",   # (n_heads, tgt_len, src_len)
    src_tokens     : List[str],        # English tokens (x-axis)
    tgt_tokens     : List[str],        # Spanish tokens (y-axis)
    title          : str = "",
    layer          : int = -1,         # which decoder layer (default: last)
    save_path      : str = None,
) -> None:
    """
    Plots the cross-attention alignment matrix.

    Unlike mini-cross-attention (anti-diagonal for reversal), here you
    should see a SOFT DIAGONAL from top-left to bottom-right, reflecting
    that English and Spanish have similar word order.

    X-axis = English source tokens (keys — what the decoder looked at)
    Y-axis = Spanish target tokens (queries — what generated each output word)

    Entry [i, j] = how much Spanish token i attended to English token j.
    """
    n_heads = cross_weights.shape[0]
    fig, axes = plt.subplots(1, n_heads, figsize=(6 * n_heads, max(5, len(tgt_tokens) * 0.6 + 2)))
    fig.patch.set_facecolor(PALETTE["bg"])

    if n_heads == 1:
        axes = [axes]

    for h, ax in enumerate(axes):
        w = cross_weights[h].numpy()   # (tgt_len, src_len)

        ax.set_facecolor(PALETTE["bg"])
        im = ax.imshow(w, cmap="Blues", vmin=0, vmax=1, aspect="auto")

        # English tokens on x-axis
        ax.set_xticks(range(len(src_tokens)))
        ax.set_xticklabels(src_tokens, rotation=45, ha="right",
                           fontsize=9, color=PALETTE["accent"],
                           fontfamily="monospace")
        ax.set_xlabel("English  (source — keys)", color=PALETTE["muted"], fontsize=9)

        # Spanish tokens on y-axis
        ax.set_yticks(range(len(tgt_tokens)))
        ax.set_yticklabels(tgt_tokens, fontsize=9,
                           color=PALETTE["highlight"],
                           fontfamily="monospace")
        ax.set_ylabel("Spanish  (target — queries)", color=PALETTE["muted"], fontsize=9)

        # Annotate cells
        for i in range(len(tgt_tokens)):
            for j in range(len(src_tokens)):
                val   = w[i, j]
                color = "white" if val > 0.5 else PALETTE["muted"]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

        for sp in ax.spines.values():
            sp.set_edgecolor(PALETTE["grid"])

        layer_label = f"Layer {layer}" if layer >= 0 else "Last layer"
        ax.set_title(
            f"Head {h}  ·  {layer_label}\n"
            f"Blue = English word attended to when generating Spanish word\n"
            f"Diagonal = similar word order (English ≈ Spanish)",
            color=PALETTE["text"], fontsize=8, pad=8,
        )

    full_title = f"Alignment Matrix  ·  {title}" if title else "Alignment Matrix"
    fig.suptitle(full_title, color=PALETTE["text"], fontsize=11, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
        print(f"Alignment saved → {save_path}")
    plt.show()


def plot_all_layers_alignment(
    all_cross_weights : List["torch.Tensor"],  # list[n_layers] of (n_heads, T, S)
    src_tokens        : List[str],
    tgt_tokens        : List[str],
    title             : str = "",
    save_path         : str = None,
) -> None:
    """
    Plots mean alignment (averaged across heads) for every decoder layer.
    Shows how the alignment sharpens layer by layer.
    """
    n_layers = len(all_cross_weights)
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, max(4, len(tgt_tokens) * 0.5 + 2)))
    fig.patch.set_facecolor(PALETTE["bg"])

    if n_layers == 1:
        axes = [axes]

    for li, (ax, weights) in enumerate(zip(axes, all_cross_weights)):
        w = weights.mean(dim=0).numpy()  # average across heads → (tgt, src)

        ax.set_facecolor(PALETTE["bg"])
        ax.imshow(w, cmap="Blues", vmin=0, vmax=1, aspect="auto")

        ax.set_xticks(range(len(src_tokens)))
        ax.set_xticklabels(src_tokens, rotation=45, ha="right",
                           fontsize=8, color=PALETTE["accent"],
                           fontfamily="monospace")
        ax.set_yticks(range(len(tgt_tokens)))
        ax.set_yticklabels(tgt_tokens, fontsize=8,
                           color=PALETTE["highlight"],
                           fontfamily="monospace")

        for sp in ax.spines.values():
            sp.set_edgecolor(PALETTE["grid"])
        ax.set_title(f"Layer {li + 1}  (mean across heads)",
                     color=PALETTE["text"], fontsize=9, pad=6)

    fig.suptitle(
        f"Alignment by decoder layer  ·  {title}\n"
        f"Later layers typically show sharper alignment",
        color=PALETTE["text"], fontsize=10, y=1.04,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=140, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
        print(f"Layer alignment saved → {save_path}")
    plt.show()


# ── Training curves ───────────────────────────────────────────────────────────

def plot_training(
    train_hist : list,
    val_hist   : list,
    save_path  : str = None,
) -> None:
    """
    Plots BLEU score and perplexity over training epochs (train vs val).

    BLEU is the main translation metric:
      0.0 = no overlap with reference
      1.0 = perfect match
    Perplexity shows how surprised the model is by each token.
    """
    epochs    = list(range(1, len(train_hist) + 1))
    t_bleu    = [h[2] for h in train_hist]
    v_bleu    = [h[2] for h in val_hist]
    t_ppl     = [h[1] for h in train_hist]
    v_ppl     = [h[1] for h in val_hist]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(PALETTE["bg"])

    # BLEU
    ax = axes[0]
    ax.set_facecolor(PALETTE["bg"])
    ax.grid(color=PALETTE["grid"], linewidth=0.5)
    ax.plot(epochs, t_bleu, color=PALETTE["accent"],    linewidth=2, label="Train BLEU")
    ax.plot(epochs, v_bleu, color=PALETTE["highlight"], linewidth=2, label="Val BLEU")
    ax.fill_between(epochs, t_bleu, v_bleu,
                    where=[t > v for t, v in zip(t_bleu, v_bleu)],
                    alpha=0.12, color=PALETTE["highlight"], label="Gap")
    ax.set_xlabel("Epoch", color=PALETTE["muted"])
    ax.set_ylabel("BLEU  (higher = better translation)", color=PALETTE["muted"])
    ax.tick_params(colors=PALETTE["muted"])
    for sp in ax.spines.values(): sp.set_edgecolor(PALETTE["grid"])
    ax.legend(facecolor=PALETTE["grid"], labelcolor=PALETTE["text"], fontsize=9)
    ax.set_title("BLEU Score", color=PALETTE["text"], fontsize=11, pad=8, loc="left")

    # Perplexity
    ax = axes[1]
    ax.set_facecolor(PALETTE["bg"])
    ax.grid(color=PALETTE["grid"], linewidth=0.5)
    ax.plot(epochs, t_ppl, color=PALETTE["accent"],    linewidth=2, label="Train PPL")
    ax.plot(epochs, v_ppl, color=PALETTE["highlight"], linewidth=2, label="Val PPL")
    ax.set_xlabel("Epoch", color=PALETTE["muted"])
    ax.set_ylabel("Perplexity  (lower = better)", color=PALETTE["muted"])
    ax.tick_params(colors=PALETTE["muted"])
    for sp in ax.spines.values(): sp.set_edgecolor(PALETTE["grid"])
    ax.legend(facecolor=PALETTE["grid"], labelcolor=PALETTE["text"], fontsize=9)
    ax.set_title("Perplexity", color=PALETTE["text"], fontsize=11, pad=8, loc="left")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
        print(f"Training curves saved → {save_path}")
    plt.show()


# ── Vocabulary comparison ─────────────────────────────────────────────────────

def plot_vocab_comparison(
    src_vocab  : "Vocabulary",
    tgt_vocab  : "Vocabulary",
    n_words    : int = 30,
    save_path  : str = None,
) -> None:
    """
    Side-by-side bar chart comparing English and Spanish vocabulary sizes
    and showing sample words from each — makes the separate-vocab design visible.
    """
    src_words = [w for w in list(src_vocab.w2i.keys()) if not w.startswith("<")][:n_words]
    tgt_words = [w for w in list(tgt_vocab.w2i.keys()) if not w.startswith("<")][:n_words]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, n_words * 0.3 + 2)))
    fig.patch.set_facecolor(PALETTE["bg"])

    for ax, words, label, color in [
        (axes[0], src_words, f"English vocabulary  ({src_vocab.size} tokens)", PALETTE["accent"]),
        (axes[1], tgt_words, f"Spanish vocabulary  ({tgt_vocab.size} tokens)", PALETTE["highlight"]),
    ]:
        ax.set_facecolor(PALETTE["bg"])
        indices = list(range(len(words)))
        ax.barh(indices, [1] * len(words), color=color, alpha=0.6, edgecolor=PALETTE["bg"])
        ax.set_yticks(indices)
        ax.set_yticklabels(words, fontsize=9, color=PALETTE["text"],
                           fontfamily="monospace")
        ax.set_xticks([])
        ax.set_title(label, color=PALETTE["text"], fontsize=10, pad=8)
        ax.invert_yaxis()
        for sp in ax.spines.values(): sp.set_edgecolor(PALETTE["grid"])

    fig.suptitle(
        "Separate vocabularies — English encoder vs Spanish decoder\n"
        "Cross-attention bridges the two embedding spaces",
        color=PALETTE["text"], fontsize=11, y=1.01,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight",
                    facecolor=PALETTE["bg"])
        print(f"Vocabulary comparison saved → {save_path}")
    plt.show()
