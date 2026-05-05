"""
tokenizer.py — Separate source (English) and target (Spanish) vocabularies.

Why separate vocabularies?
  English and Spanish share some words (cognates: "animal", "hospital",
  "solar") but mostly have different words. Using separate vocabularies:
    - Each language gets its own embedding matrix
    - The encoder embedding is purely English
    - The decoder embedding is purely Spanish
    - Cross-attention learns to bridge the two spaces
  This is more realistic than shared vocabularies and makes the alignment
  more interpretable: source tokens are always English, target always Spanish.

Special tokens (same indices in both vocabularies):
    <PAD>  0
    <BOS>  1
    <EOS>  2
    <UNK>  3
"""

import random
from collections import Counter
from typing import List, Dict, Tuple


class Vocabulary:
    """
    Word-level vocabulary for one language.

    Args:
        words (List[str]): all word tokens from the corpus for this language
        min_freq (int):    minimum frequency to include a word
    """

    PAD, BOS, EOS, UNK = "<PAD>", "<BOS>", "<EOS>", "<UNK>"
    SPECIAL = [PAD, BOS, EOS, UNK]

    def __init__(self, words: List[str], min_freq: int = 1) -> None:
        counts = Counter(words)
        vocab  = sorted(w for w, c in counts.items() if c >= min_freq)
        all_tokens = self.SPECIAL + vocab

        self.w2i: Dict[str, int] = {t: i for i, t in enumerate(all_tokens)}
        self.i2w: Dict[int, str] = {i: t for t, i in self.w2i.items()}
        self.size = len(self.w2i)

        self.pad_idx = self.w2i[self.PAD]
        self.bos_idx = self.w2i[self.BOS]
        self.eos_idx = self.w2i[self.EOS]
        self.unk_idx = self.w2i[self.UNK]

    def encode(self, tokens: List[str], add_special: bool = True) -> List[int]:
        ids = [self.w2i.get(t, self.unk_idx) for t in tokens]
        if add_special:
            ids = [self.bos_idx] + ids + [self.eos_idx]
        return ids

    def decode(self, ids: List[int], strip_special: bool = True) -> List[str]:
        words = [self.i2w.get(i, self.UNK) for i in ids]
        if strip_special:
            words = [w for w in words if w not in self.SPECIAL]
        return words

    def __repr__(self) -> str:
        return f"Vocabulary(size={self.size})"


class TranslationTokenizer:
    """
    Loads a tab-separated bilingual corpus and builds separate
    source and target vocabularies.

    Corpus format (one pair per line, tab-separated):
        English sentence\\tSpanish sentence

    Args:
        path (str):       path to corpus file
        val_split (float): fraction held out for validation
        min_freq (int):   minimum token frequency
        seed (int):       random seed for split
    """

    def __init__(
        self,
        path      : str,
        val_split : float = 0.1,
        min_freq  : int   = 1,
        seed      : int   = 42,
    ) -> None:
        random.seed(seed)

        src_sents, tgt_sents = [], []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or "\t" not in line:
                    continue
                src, tgt = line.split("\t", 1)
                src_sents.append(src.strip().lower().split())
                tgt_sents.append(tgt.strip().lower().split())

        # Shuffle and split
        pairs = list(zip(src_sents, tgt_sents))
        random.shuffle(pairs)
        n_val = max(1, int(len(pairs) * val_split))

        self.val_pairs   = pairs[:n_val]
        self.train_pairs = pairs[n_val:]
        self.all_pairs   = pairs

        # Build vocabularies from FULL corpus so val words are known
        all_src = [w for s, _ in pairs for w in s]
        all_tgt = [w for _, t in pairs for w in t]
        self.src_vocab = Vocabulary(all_src, min_freq)
        self.tgt_vocab = Vocabulary(all_tgt, min_freq)

    def encode_pair(
        self, src: List[str], tgt: List[str]
    ) -> Tuple[List[int], List[int]]:
        """Encodes one (src, tgt) pair. Target gets BOS/EOS; source does not."""
        return (
            self.src_vocab.encode(src, add_special=False),
            self.tgt_vocab.encode(tgt, add_special=True),
        )

    def encode_split(
        self,
    ) -> Tuple[List[Tuple], List[Tuple]]:
        """Returns (train_encoded, val_encoded) as lists of (src_ids, tgt_ids)."""
        train = [self.encode_pair(s, t) for s, t in self.train_pairs]
        val   = [self.encode_pair(s, t) for s, t in self.val_pairs]
        return train, val

    def __repr__(self) -> str:
        return (
            f"TranslationTokenizer(\n"
            f"  src_vocab={self.src_vocab},\n"
            f"  tgt_vocab={self.tgt_vocab},\n"
            f"  train_pairs={len(self.train_pairs)},\n"
            f"  val_pairs={len(self.val_pairs)}\n"
            f")"
        )
