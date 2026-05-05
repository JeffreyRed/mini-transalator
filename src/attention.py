"""
attention.py — Self-attention and cross-attention.
Self-contained copy for repository independence.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k    = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights, nan=0.0)
    return torch.matmul(weights, V), weights


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert emb_dim % n_heads == 0
        self.emb_dim  = emb_dim
        self.n_heads  = n_heads
        self.head_dim = emb_dim // n_heads
        self.W_Q  = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_K  = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_V  = nn.Linear(emb_dim, emb_dim, bias=False)
        self.W_O  = nn.Linear(emb_dim, emb_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        Q = self._split(self.W_Q(x), B, T)
        K = self._split(self.W_K(x), B, T)
        V = self._split(self.W_V(x), B, T)
        out, w = scaled_dot_product_attention(Q, K, V, mask)
        out = self.drop(out)
        merged = out.transpose(1, 2).contiguous().view(B, T, self.emb_dim)
        return self.W_O(merged), w

    def _split(self, x, B, T):
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)


class MultiHeadCrossAttention(nn.Module):
    """
    Cross-attention: Q from target (decoder), K/V from source (encoder).

    Weight matrix shape: (batch, n_heads, tgt_len, src_len)
    This is the alignment matrix — shows which source (English) tokens
    each target (Spanish) token attended to.
    """
    def __init__(self, emb_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert emb_dim % n_heads == 0
        self.emb_dim  = emb_dim
        self.n_heads  = n_heads
        self.head_dim = emb_dim // n_heads
        self.W_Q  = nn.Linear(emb_dim, emb_dim, bias=False)  # target → Q
        self.W_K  = nn.Linear(emb_dim, emb_dim, bias=False)  # source → K
        self.W_V  = nn.Linear(emb_dim, emb_dim, bias=False)  # source → V
        self.W_O  = nn.Linear(emb_dim, emb_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, context, mask=None):
        B, T_tgt, _ = x.shape
        _, T_src, _ = context.shape
        Q = self._split(self.W_Q(x),       B, T_tgt)
        K = self._split(self.W_K(context), B, T_src)
        V = self._split(self.W_V(context), B, T_src)
        out, w = scaled_dot_product_attention(Q, K, V, mask)
        out = self.drop(out)
        merged = out.transpose(1, 2).contiguous().view(B, T_tgt, self.emb_dim)
        return self.W_O(merged), w   # w: (B, n_heads, tgt_len, src_len)

    def _split(self, x, B, T):
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
