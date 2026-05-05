"""
main.py — End-to-end pipeline for mini-translator.

Usage:
    python main.py

Pipeline:
    1.  Load bilingual corpus, build separate English/Spanish vocabularies
    2.  Show vocabulary comparison
    3.  Build MiniTranslator (4+4 encoder-decoder blocks)
    4.  Train with BLEU + perplexity tracking
    5.  Show translation examples (model vs reference)
    6.  Plot alignment matrices  — soft diagonal (English ≈ Spanish word order)
    7.  Plot layer-by-layer alignment (sharpening across layers)
    8.  Interactive translation loop
    9.  Save model
"""

import torch
from pathlib import Path

from src.tokenizer  import TranslationTokenizer
from src.dataset    import TranslationDataset
from src.model      import MiniTranslator
from src.train      import train, evaluate
from src.utils      import (
    translate, get_alignment,
    show_translations, interactive_translate,
)
from src.visualize  import (
    plot_alignment, plot_all_layers_alignment,
    plot_training, plot_vocab_comparison,
)

# ── Config ────────────────────────────────────────────────────────────────────
CORPUS_PATH   = "data/corpus.txt"
VAL_SPLIT     = 0.1
EMB_DIM       = 128       # larger than mini-cross-attention — real vocabulary needs more capacity
N_HEADS       = 4         # head_dim = 32
N_LAYERS      = 4         # 4 encoder + 4 decoder blocks
FF_DIM        = 256       # 2 × EMB_DIM
MAX_LEN       = 64
EPOCHS        = 80
LR            = 3e-3
MIN_LR        = 1e-5
BATCH_SIZE    = 16
WARMUP_STEPS  = 200
PATIENCE      = 15
OUTPUTS_DIR   = Path("outputs")

# Sentences to inspect in detail (English)
INSPECT_SENTS = [
    "the cat sits on the mat",
    "imagination is more important than knowledge",
    "plants need sunlight to grow",
    "the brain processes information quickly",
    "learning never stops",
]
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # ── 1. Tokeniser ──────────────────────────────────────────────────────────
    print("\n── Tokeniser ───────────────────────────────────────")
    tok = TranslationTokenizer(CORPUS_PATH, val_split=VAL_SPLIT)
    print(tok)
    sv, tv = tok.src_vocab, tok.tgt_vocab
    print(f"\n  English vocab sample : {list(sv.w2i.keys())[4:14]}")
    print(f"  Spanish vocab sample : {list(tv.w2i.keys())[4:14]}\n")

    # ── 2. Vocabulary comparison plot ─────────────────────────────────────────
    plot_vocab_comparison(
        sv, tv, n_words=40,
        save_path=str(OUTPUTS_DIR / "vocab_comparison.png"),
    )

    # ── 3. Datasets ───────────────────────────────────────────────────────────
    train_enc, val_enc = tok.encode_split()
    train_ds = TranslationDataset(train_enc)
    val_ds   = TranslationDataset(val_enc)
    print(f"  {train_ds}  (train)")
    print(f"  {val_ds}    (val)\n")

    # ── 4. Model ──────────────────────────────────────────────────────────────
    print("── Model ───────────────────────────────────────────")
    model = MiniTranslator(
        src_vocab_size = sv.size,
        tgt_vocab_size = tv.size,
        emb_dim        = EMB_DIM,
        n_heads        = N_HEADS,
        n_layers       = N_LAYERS,
        ff_dim         = FF_DIM,
        max_len        = MAX_LEN,
        src_pad_idx    = sv.pad_idx,
        tgt_pad_idx    = tv.pad_idx,
    )
    print(model, "\n")

    # ── 5. Train ──────────────────────────────────────────────────────────────
    print("── Training ────────────────────────────────────────")
    print("  Metrics: BLEU (translation quality) + perplexity")
    print("  BLEU: 0.0 = no match,  1.0 = perfect,  >0.5 = good on clean corpus\n")

    train_hist, val_hist = train(
        model, train_ds, val_ds,
        tgt_bos_idx  = tv.bos_idx,
        tgt_eos_idx  = tv.eos_idx,
        src_pad_idx  = sv.pad_idx,
        tgt_pad_idx  = tv.pad_idx,
        epochs       = EPOCHS,
        lr           = LR,
        min_lr       = MIN_LR,
        batch_size   = BATCH_SIZE,
        warmup_steps = WARMUP_STEPS,
        patience     = PATIENCE,
    )

    _, _, final_bleu = val_hist[-1][0], val_hist[-1][1], val_hist[-1][2]
    best_bleu        = max(h[2] for h in val_hist)
    best_epoch       = next(i+1 for i, h in enumerate(val_hist) if h[2] == best_bleu)
    print(f"\n  Best val BLEU = {best_bleu:.3f} at epoch {best_epoch}\n")

    # ── 6. Training curves ────────────────────────────────────────────────────
    plot_training(
        train_hist, val_hist,
        save_path=str(OUTPUTS_DIR / "training.png"),
    )

    # ── 7. Translation examples ───────────────────────────────────────────────
    show_translations(model, tok, n=12)

    # ── 8. Alignment matrices ─────────────────────────────────────────────────
    print("── Alignment matrices ──────────────────────────────")
    print("  Soft diagonal = English and Spanish share similar word order")
    print("  Brighter cell = decoder attended more to that English token\n")

    for i, sent in enumerate(INSPECT_SENTS):
        src_toks, tgt_toks, cross_ws = get_alignment(model, tok, sent)

        if not tgt_toks or cross_ws is None:
            continue

        # Last decoder layer, trimmed to actual lengths
        last_w = cross_ws[-1].squeeze(0)           # (n_heads, tgt_len, src_len)
        last_w = last_w[:, :len(tgt_toks), :len(src_toks)]

        plot_alignment(
            last_w, src_toks, tgt_toks,
            title     = sent,
            layer     = N_LAYERS - 1,
            save_path = str(OUTPUTS_DIR / f"alignment_{i}.png"),
        )

        # Layer-by-layer for first sentence only
        if i == 0:
            all_w_trimmed = [
                w.squeeze(0)[:, :len(tgt_toks), :len(src_toks)]
                for w in cross_ws
            ]
            plot_all_layers_alignment(
                all_w_trimmed, src_toks, tgt_toks,
                title     = sent,
                save_path = str(OUTPUTS_DIR / "alignment_layers.png"),
            )

    # ── 9. Interactive translation ────────────────────────────────────────────
    interactive_translate(model, tok)

    # ── 10. Save model ────────────────────────────────────────────────────────
    ckpt = OUTPUTS_DIR / "translator.pt"
    torch.save({
        "model_state":    model.state_dict(),
        "config": dict(
            src_vocab_size = sv.size,
            tgt_vocab_size = tv.size,
            emb_dim        = EMB_DIM,
            n_heads        = N_HEADS,
            n_layers       = N_LAYERS,
            ff_dim         = FF_DIM,
            max_len        = MAX_LEN,
        ),
        "best_val_bleu": best_bleu,
        "best_epoch":    best_epoch,
    }, ckpt)
    print(f"\nModel saved → {ckpt}")


if __name__ == "__main__":
    main()
