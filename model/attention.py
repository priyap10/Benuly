"""
model/attention.py

The core mechanism of a transformer, built from raw operations —
no nn.MultiheadAttention, no shortcuts. This is the part you should
be able to explain line by line in an interview.

Self-attention, conceptually:
    For every token, ask: "which other tokens (including myself) should
    I pay attention to, and how much?" Do this by projecting each token
    into a Query, a Key, and a Value vector. A token's Query is compared
    against every other token's Key (dot product) to get attention
    scores; those scores are turned into weights (softmax); the output
    is a weighted sum of every token's Value vector.

Causal masking: since this is a language model (predict the next token),
a token must NOT be allowed to attend to tokens that come after it.
We enforce that with a lower-triangular mask.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """A single self-attention head."""

    def __init__(self, d_model, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)

        # Lower-triangular mask, registered as a buffer (not a trainable
        # parameter, but moves with the model to whatever device it's on)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size

    def forward(self, x):
        # x: (batch, time, d_model)
        B, T, C = x.shape

        k = self.key(x)     # (B, T, head_size)
        q = self.query(x)   # (B, T, head_size)
        v = self.value(x)   # (B, T, head_size)

        # Attention scores: how much does each query "match" each key?
        # Scale by sqrt(head_size) to keep the dot products from
        # growing too large and pushing softmax into a near-one-hot state.
        scores = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)  # (B, T, T)

        # Causal mask: block attention to future positions
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)  # (B, T, T), rows sum to 1
        weights = self.dropout(weights)

        out = weights @ v  # (B, T, head_size) — weighted sum of values
        return out


class MultiHeadAttention(nn.Module):
    """Multiple attention heads running in parallel, concatenated and projected back."""

    def __init__(self, d_model, n_heads, block_size, dropout):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        head_size = d_model // n_heads

        self.heads = nn.ModuleList([
            Head(d_model, head_size, block_size, dropout) for _ in range(n_heads)
        ])
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Run every head, concatenate their outputs along the last dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, d_model)
        out = self.proj(out)
        out = self.dropout(out)
        return out