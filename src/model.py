"""
model.py — MiniTranslator: English→Spanish encoder-decoder.

Key difference from mini-cross-attention:
  - Separate embedding matrices for source (English) and target (Spanish)
    encoder_embedding: (src_vocab_size × emb_dim)
    decoder_embedding: (tgt_vocab_size × emb_dim)
  - Both languages share emb_dim so cross-attention dimensions match
  - 4 encoder blocks + 4 decoder blocks

Architecture:
    English tokens
        │
        ▼
    src_embedding  (English vocab → emb_dim)
        │
        ▼
    PositionalEncoding
        │
        ▼
    EncoderBlock × 4   (self-attention, no mask)
        │
        ▼ encoder_output  (batch, src_len, emb_dim)
        │
        │    Spanish tokens (teacher forcing)
        │         │
        │         ▼
        │    tgt_embedding  (Spanish vocab → emb_dim)
        │         │
        │         ▼
        │    PositionalEncoding
        │         │
        │    DecoderBlock × 4
        │      1. causal self-attention
        └──→  2. cross-attention  Q=decoder, K/V=encoder_output
              3. feedforward
        │
        ▼
    Linear → Spanish vocab logits
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from src.attention import MultiHeadSelfAttention, MultiHeadCrossAttention


class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim: int, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, emb_dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, emb_dim, 2).float()
                        * -(math.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class FeedForward(nn.Module):
    def __init__(self, emb_dim: int, ff_dim: int = None, dropout: float = 0.1):
        super().__init__()
        ff_dim = ff_dim or 4 * emb_dim
        self.net = nn.Sequential(
            nn.Linear(emb_dim, ff_dim), nn.GELU(),
            nn.Linear(ff_dim, emb_dim), nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    """Self-attention encoder block — reads English, no mask."""
    def __init__(self, emb_dim, n_heads, ff_dim=None, dropout=0.1):
        super().__init__()
        self.attn  = MultiHeadSelfAttention(emb_dim, n_heads, dropout)
        self.ff    = FeedForward(emb_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        a, _ = self.attn(self.norm1(x), src_mask)
        x    = x + self.drop(a)
        x    = x + self.drop(self.ff(self.norm2(x)))
        return x


class DecoderBlock(nn.Module):
    """
    Three sub-layers:
      1. Causal self-attention  (Spanish tokens attend to past Spanish tokens)
      2. Cross-attention        (Spanish decoder queries English encoder output)
      3. FeedForward
    """
    def __init__(self, emb_dim, n_heads, ff_dim=None, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadSelfAttention(emb_dim, n_heads, dropout)
        self.cross_attn = MultiHeadCrossAttention(emb_dim, n_heads, dropout)
        self.ff         = FeedForward(emb_dim, ff_dim, dropout)
        self.norm1      = nn.LayerNorm(emb_dim)
        self.norm2      = nn.LayerNorm(emb_dim)
        self.norm3      = nn.LayerNorm(emb_dim)
        self.drop       = nn.Dropout(dropout)

    def forward(self, x, encoder_out, causal_mask=None, src_pad_mask=None):
        # 1. Causal self-attention (Spanish)
        sa, _      = self.self_attn(self.norm1(x), causal_mask)
        x          = x + self.drop(sa)

        # 2. Cross-attention (Spanish → English)
        ca, cross_w = self.cross_attn(self.norm2(x), encoder_out, src_pad_mask)
        x           = x + self.drop(ca)

        # 3. FeedForward
        x = x + self.drop(self.ff(self.norm3(x)))
        return x, cross_w


class MiniTranslator(nn.Module):
    """
    English→Spanish encoder-decoder transformer.

    Args:
        src_vocab_size (int): English vocabulary size
        tgt_vocab_size (int): Spanish vocabulary size
        emb_dim (int):        shared embedding dimension
        n_heads (int):        attention heads (must divide emb_dim)
        n_layers (int):       encoder AND decoder layers (same count)
        ff_dim (int):         feedforward inner dimension
        max_len (int):        max sequence length
        dropout (float):      dropout probability
        src_pad_idx (int):    English padding index
        tgt_pad_idx (int):    Spanish padding index
    """

    def __init__(
        self,
        src_vocab_size : int,
        tgt_vocab_size : int,
        emb_dim        : int   = 128,
        n_heads        : int   = 4,
        n_layers       : int   = 4,
        ff_dim         : int   = None,
        max_len        : int   = 64,
        dropout        : float = 0.1,
        src_pad_idx    : int   = 0,
        tgt_pad_idx    : int   = 0,
    ) -> None:
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.n_layers    = n_layers
        self.emb_dim     = emb_dim

        # Separate embeddings for each language
        self.src_emb = nn.Embedding(src_vocab_size, emb_dim, padding_idx=src_pad_idx)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, emb_dim, padding_idx=tgt_pad_idx)
        self.src_pe  = PositionalEncoding(emb_dim, max_len, dropout)
        self.tgt_pe  = PositionalEncoding(emb_dim, max_len, dropout)

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(emb_dim, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(emb_dim, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])

        self.encoder_norm = nn.LayerNorm(emb_dim)
        self.decoder_norm = nn.LayerNorm(emb_dim)

        # Output projection: decoder hidden → Spanish vocab logits
        self.head = nn.Linear(emb_dim, tgt_vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def _causal_mask(self, T, device):
        return torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1
        ).unsqueeze(0).unsqueeze(0)

    def _src_pad_mask(self, src):
        return (src == self.src_pad_idx).unsqueeze(1).unsqueeze(2)

    def encode(self, src):
        """Encodes English source. Returns (encoder_output, src_pad_mask)."""
        mask = self._src_pad_mask(src)
        x    = self.src_pe(self.src_emb(src))
        for block in self.encoder_blocks:
            x = block(x, mask)
        return self.encoder_norm(x), mask

    def decode(self, tgt, encoder_out, src_pad_mask):
        """Decodes Spanish target conditioned on encoder output."""
        T            = tgt.size(1)
        causal_mask  = self._causal_mask(T, tgt.device)
        x            = self.tgt_pe(self.tgt_emb(tgt))
        all_cross_w  = []
        for block in self.decoder_blocks:
            x, cross_w = block(x, encoder_out, causal_mask, src_pad_mask)
            all_cross_w.append(cross_w)
        return self.decoder_norm(x), all_cross_w

    def forward(self, src, tgt):
        """
        Args:
            src: (batch, src_len) — English token indices
            tgt: (batch, tgt_len) — Spanish token indices (decoder input)

        Returns:
            logits:      (batch, tgt_len, tgt_vocab_size)
            all_cross_w: list[n_layers] of (batch, n_heads, tgt_len, src_len)
        """
        enc_out, src_mask       = self.encode(src)
        dec_out, all_cross_w    = self.decode(tgt, enc_out, src_mask)
        return self.head(dec_out), all_cross_w

    @torch.no_grad()
    def translate(
        self,
        src         : torch.Tensor,
        bos_idx     : int,
        eos_idx     : int,
        max_steps   : int   = 40,
        temperature : float = 1.0,
        top_k       : int   = 0,
        greedy      : bool  = True,
    ) -> Tuple[List[int], List[torch.Tensor]]:
        """
        Greedy (or sampled) autoregressive translation.

        Args:
            src:         (1, src_len) English token indices
            bos_idx:     Spanish BOS index
            eos_idx:     Spanish EOS index
            max_steps:   maximum Spanish tokens to generate
            temperature: sampling temperature (1.0 = neutral)
            top_k:       if >0 use top-k sampling
            greedy:      if True always pick argmax

        Returns:
            generated_ids: list of Spanish token indices (without BOS)
            cross_weights: list[n_layers] of (1, n_heads, tgt_len, src_len)
                           at the final decoding step
        """
        self.eval()
        enc_out, src_mask = self.encode(src)
        dec_ids           = [bos_idx]
        final_cross_w     = None

        for _ in range(max_steps):
            tgt           = torch.tensor([dec_ids], dtype=torch.long, device=src.device)
            dec_out, cws  = self.decode(tgt, enc_out, src_mask)
            logits        = self.head(dec_out)[0, -1, :]  # last position

            if greedy:
                next_id = logits.argmax().item()
            else:
                logits = logits / max(temperature, 1e-8)
                if top_k > 0:
                    vals, _ = torch.topk(logits, top_k)
                    logits[logits < vals[-1]] = float("-inf")
                probs   = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1).item()

            dec_ids.append(next_id)
            final_cross_w = cws

            if next_id == eos_idx:
                break

        return dec_ids[1:], final_cross_w  # strip BOS

    def __repr__(self):
        params = sum(p.numel() for p in self.parameters())
        return (
            f"MiniTranslator(\n"
            f"  src_vocab={self.src_emb.num_embeddings}, "
            f"tgt_vocab={self.tgt_emb.num_embeddings},\n"
            f"  emb_dim={self.emb_dim}, n_layers={self.n_layers},\n"
            f"  n_heads={self.decoder_blocks[0].cross_attn.n_heads},\n"
            f"  parameters={params:,}\n"
            f")"
        )
