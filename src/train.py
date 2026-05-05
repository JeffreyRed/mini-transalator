"""
train.py — Training loop with BLEU score and perplexity evaluation.

New vs mini-cross-attention:
  BLEU score — the standard metric for translation quality.
  It measures how much the generated translation overlaps with the
  reference translation using n-gram precision.

  BLEU = 0.0  → no overlap at all
  BLEU = 1.0  → perfect match (rare even for human translators)
  BLEU > 0.3  → reasonable translation quality
  BLEU > 0.5  → good quality on a small clean corpus like this one

  We compute a simple sentence-level BLEU (1-gram + 2-gram) that is
  easy to understand without external libraries.
"""

import math
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.utils.data import DataLoader
from typing import List, Tuple

from src.model   import MiniTranslator
from src.dataset import TranslationDataset, make_collate


# ── Simple BLEU ───────────────────────────────────────────────────────────────

def simple_bleu(hypothesis: List[int], reference: List[int], max_n: int = 2) -> float:
    """
    Computes a simple BLEU score for one sentence pair.

    Measures how many n-grams in the hypothesis appear in the reference.
    Penalises hypotheses shorter than the reference (brevity penalty).

    Args:
        hypothesis: predicted token ids
        reference:  correct token ids
        max_n:      maximum n-gram order (default 2 = bigram BLEU)

    Returns:
        BLEU score between 0.0 and 1.0
    """
    if not hypothesis or not reference:
        return 0.0

    scores = []
    for n in range(1, max_n + 1):
        hyp_ngrams = Counter(
            tuple(hypothesis[i:i+n]) for i in range(len(hypothesis)-n+1)
        )
        ref_ngrams = Counter(
            tuple(reference[i:i+n]) for i in range(len(reference)-n+1)
        )
        matches = sum((hyp_ngrams & ref_ngrams).values())
        total   = sum(hyp_ngrams.values())
        if total == 0:
            return 0.0
        scores.append(matches / total)

    if not all(s > 0 for s in scores):
        return 0.0

    log_avg = sum(math.log(s) for s in scores) / len(scores)

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(reference) / max(len(hypothesis), 1)))

    return bp * math.exp(log_avg)


def cosine_lr(step, warmup, total, peak, min_lr=1e-5):
    if step < warmup:
        return peak * step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (peak - min_lr) * (1 + math.cos(math.pi * p))


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model          : MiniTranslator,
    dataset        : TranslationDataset,
    tgt_bos_idx    : int,
    tgt_eos_idx    : int,
    src_pad_idx    : int,
    tgt_pad_idx    : int,
    batch_size     : int = 32,
    n_bleu_samples : int = 50,
) -> Tuple[float, float, float]:
    """
    Evaluates loss, perplexity, and BLEU on a dataset.

    Returns:
        (mean_loss, perplexity, mean_bleu)
    """
    model.eval()
    loader  = DataLoader(dataset, batch_size=batch_size,
                         collate_fn=make_collate(src_pad_idx, tgt_pad_idx))
    loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
    total, n = 0.0, 0

    for src, dec_in, dec_tgt in loader:
        logits, _ = model(src, dec_in)
        loss = loss_fn(logits.transpose(1, 2), dec_tgt)
        total += loss.item(); n += 1

    mean_loss = total / max(n, 1)
    ppl       = math.exp(min(mean_loss, 30))

    # BLEU on a sample of the dataset
    bleu_scores = []
    sample = dataset.examples[:n_bleu_samples]
    for src_ids, dec_in_ids, dec_tgt_ids in sample:
        src    = torch.tensor([src_ids])
        pred, _ = model.translate(src, tgt_bos_idx, tgt_eos_idx, greedy=True)
        ref    = dec_tgt_ids  # already without BOS, includes EOS
        # strip EOS from reference for BLEU
        ref_clean = [t for t in ref if t != tgt_eos_idx]
        bleu_scores.append(simple_bleu(pred, ref_clean))

    mean_bleu = sum(bleu_scores) / max(len(bleu_scores), 1)
    return mean_loss, ppl, mean_bleu


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    model          : MiniTranslator,
    train_dataset  : TranslationDataset,
    val_dataset    : TranslationDataset,
    tgt_bos_idx    : int,
    tgt_eos_idx    : int,
    src_pad_idx    : int,
    tgt_pad_idx    : int,
    epochs         : int   = 80,
    lr             : float = 3e-3,
    min_lr         : float = 1e-5,
    batch_size     : int   = 16,
    warmup_steps   : int   = 200,
    patience       : int   = 15,
    verbose        : bool  = True,
) -> Tuple[list, list]:
    """
    Trains MiniTranslator with early stopping.

    Returns:
        train_history : list of (loss, ppl, bleu) per epoch
        val_history   : same for validation
    """
    loader    = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, collate_fn=make_collate(src_pad_idx, tgt_pad_idx),
    )
    loss_fn   = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    total_steps      = epochs * max(len(loader), 1)
    step             = 0
    best_val_bleu    = -1.0
    epochs_no_improve = 0
    train_hist, val_hist = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for src, dec_in, dec_tgt in loader:
            step += 1
            for pg in optimizer.param_groups:
                pg["lr"] = cosine_lr(step, warmup_steps, total_steps, lr, min_lr)

            optimizer.zero_grad()
            logits, _ = model(src, dec_in)
            loss = loss_fn(logits.transpose(1, 2), dec_tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        t_loss = total_loss / max(len(loader), 1)
        t_ppl  = math.exp(min(t_loss, 30))
        _, _, t_bleu = evaluate(model, train_dataset, tgt_bos_idx, tgt_eos_idx,
                                src_pad_idx, tgt_pad_idx, batch_size, n_bleu_samples=30)
        train_hist.append((t_loss, t_ppl, t_bleu))

        v_loss, v_ppl, v_bleu = evaluate(model, val_dataset, tgt_bos_idx, tgt_eos_idx,
                                         src_pad_idx, tgt_pad_idx, batch_size)
        val_hist.append((v_loss, v_ppl, v_bleu))

        if v_bleu > best_val_bleu:
            best_val_bleu    = v_bleu
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch % 5 == 0 or epoch == 1):
            stop_msg = f"  (stop in {patience - epochs_no_improve})" if epochs_no_improve > 0 else ""
            print(
                f"Epoch [{epoch:>3}/{epochs}]  "
                f"loss={t_loss:.3f}  ppl={t_ppl:.1f}  "
                f"train_bleu={t_bleu:.3f}  val_bleu={v_bleu:.3f}"
                f"{stop_msg}"
            )

        if epochs_no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch}. Best val BLEU = {best_val_bleu:.3f}")
            break

    return train_hist, val_hist
