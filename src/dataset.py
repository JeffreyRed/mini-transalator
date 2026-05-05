"""
dataset.py — Translation dataset for encoder-decoder training.

Each example is a (source_ids, decoder_input, decoder_target) triple:

    source:         English token indices  (no BOS/EOS — encoder sees full source)
    decoder_input:  [BOS] + Spanish indices  (teacher forcing input)
    decoder_target: Spanish indices + [EOS]  (what the model must predict)

Example:
    English:  "the cat sits"     → [4, 7, 12]
    Spanish:  "el gato se sienta" → BOS=1, EOS=2
    dec_in:   [1, 5, 9, 15, 18]
    dec_tgt:  [5, 9, 15, 18, 2]
"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple


class TranslationDataset(Dataset):

    def __init__(
        self,
        encoded_pairs : List[Tuple[List[int], List[int]]],
        min_len       : int = 2,
    ) -> None:
        self.examples = []
        for src, tgt in encoded_pairs:
            if len(src) < min_len or len(tgt) < 2:
                continue
            self.examples.append((src, tgt[:-1], tgt[1:]))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx):
        src, dec_in, dec_tgt = self.examples[idx]
        return (
            torch.tensor(src,     dtype=torch.long),
            torch.tensor(dec_in,  dtype=torch.long),
            torch.tensor(dec_tgt, dtype=torch.long),
        )

    def __repr__(self) -> str:
        return f"TranslationDataset(pairs={len(self.examples)})"


def collate_fn(batch, src_pad=0, tgt_pad=0):
    srcs, dec_ins, dec_tgts = zip(*batch)

    def pad(seqs, pad_idx):
        max_len = max(s.size(0) for s in seqs)
        out = torch.full((len(seqs), max_len), pad_idx, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :s.size(0)] = s
        return out

    return pad(srcs, src_pad), pad(dec_ins, tgt_pad), pad(dec_tgts, tgt_pad)


def make_collate(src_pad: int, tgt_pad: int):
    def _fn(batch):
        return collate_fn(batch, src_pad, tgt_pad)
    return _fn
