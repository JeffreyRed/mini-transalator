"""
utils.py — Translation interface, alignment inspection, example display.
"""

import torch
from typing import List, Tuple

from src.model     import MiniTranslator
from src.tokenizer import TranslationTokenizer


def translate(
    model       : MiniTranslator,
    tokenizer   : TranslationTokenizer,
    sentence    : str,
    greedy      : bool  = True,
    temperature : float = 1.0,
    top_k       : int   = 0,
) -> Tuple[str, List[torch.Tensor]]:
    """
    Translates an English sentence to Spanish.

    Args:
        model:      trained MiniTranslator
        tokenizer:  TranslationTokenizer used at training
        sentence:   English sentence (plain text)
        greedy:     if True pick argmax at each step
        temperature: sampling temperature
        top_k:      top-k sampling

    Returns:
        translation:  Spanish sentence as a string
        cross_weights: list[n_layers] of (1, n_heads, tgt_len, src_len)
    """
    model.eval()
    sv     = tokenizer.src_vocab
    tv     = tokenizer.tgt_vocab
    tokens = sentence.strip().lower().split()
    src_ids = sv.encode(tokens, add_special=False)
    src     = torch.tensor([src_ids], dtype=torch.long)

    pred_ids, cross_ws = model.translate(
        src,
        bos_idx     = tv.bos_idx,
        eos_idx     = tv.eos_idx,
        greedy      = greedy,
        temperature = temperature,
        top_k       = top_k,
    )

    # Strip EOS
    pred_ids = [i for i in pred_ids if i != tv.eos_idx]
    translation = " ".join(tv.decode(pred_ids, strip_special=True))
    return translation, cross_ws


def get_alignment(
    model       : MiniTranslator,
    tokenizer   : TranslationTokenizer,
    sentence    : str,
) -> Tuple[List[str], List[str], List[torch.Tensor]]:
    """
    Returns source tokens, target tokens, and cross-attention weights
    for a translated sentence. Used for alignment plotting.
    """
    sv      = tokenizer.src_vocab
    tv      = tokenizer.tgt_vocab
    src_toks = sentence.strip().lower().split()

    translation, cross_ws = translate(model, tokenizer, sentence)
    tgt_toks = translation.split()

    return src_toks, tgt_toks, cross_ws


def show_translations(
    model     : MiniTranslator,
    tokenizer : TranslationTokenizer,
    n         : int = 10,
) -> None:
    """Shows n training examples with model translations vs references."""
    print("── Translation examples ─────────────────────────────")
    print(f"  {'English':<35}  {'Model output':<35}  {'Reference'}")
    print(f"  {'-'*35}  {'-'*35}  {'-'*30}")

    correct = 0
    for src_toks, tgt_toks in tokenizer.train_pairs[:n]:
        src_str = " ".join(src_toks)
        ref_str = " ".join(tgt_toks)
        out, _  = translate(model, tokenizer, src_str)

        match = out.strip() == ref_str.strip()
        correct += int(match)
        tick = "✓" if match else " "
        print(f"  {tick} {src_str:<34}  {out:<35}  {ref_str}")

    print(f"\n  Exact matches: {correct}/{n}\n")
    print("────────────────────────────────────────────────────\n")


def interactive_translate(
    model     : MiniTranslator,
    tokenizer : TranslationTokenizer,
) -> None:
    """
    Interactive English→Spanish translation loop.
    Flags: --temp=0.8  --topk=5  --sample
    Type 'quit' to exit.
    """
    sv             = tokenizer.src_vocab
    lower_to_vocab = {w.lower(): w for w in sv.w2i}

    print("── Interactive translation ──────────────────────────")
    print(f"  English vocabulary sample: {list(sv.w2i.keys())[4:20]} ...")
    print("  Type an English sentence to translate to Spanish.")
    print("  Flags: --temp=0.8  --topk=5  --sample")
    print("  Type 'quit' to exit.\n")

    while True:
        try:
            raw = input("  English: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Exiting.")
            break

        if raw.lower() in ("quit", "exit", "q"):
            break
        if not raw:
            continue

        parts  = raw.split()
        flags  = {p for p in parts if p.startswith("--")}
        words  = [p for p in parts if not p.startswith("--")]

        greedy  = "--sample" not in flags
        temp    = 1.0
        top_k   = 0
        for f in flags:
            if f.startswith("--temp="): temp  = float(f.split("=")[1])
            if f.startswith("--topk="): top_k = int(f.split("=")[1])

        # Check for unknown words
        unknown = [w for w in words if w.lower() not in sv.w2i]
        if unknown:
            print(f"  ✗ Unknown words: {unknown}\n")
            continue

        sentence = " ".join(words)
        out, _   = translate(model, tokenizer, sentence,
                             greedy=greedy, temperature=temp, top_k=top_k)
        mode = "greedy" if greedy else f"sample(temp={temp})"
        print(f"\n  [{mode}]\n  Spanish: {out}\n")

    print("────────────────────────────────────────────────────\n")
