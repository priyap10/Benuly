"""
model/transformer_block.py

One transformer block = the repeating unit that gets stacked N_LAYERS
times to form the full model. Each block has two sub-layers:

    1. Multi-head self-attention (mixes information across tokens —
       "what should I know about other tokens in the sequence?")
    2. A position-wise feed-forward network (processes each token
       independently — "given what I now know, transform my representation")

Both sub-layers are wrapped with:
    - A residual connection (add the input back to the output) — this is
      what makes deep stacks of these blocks trainable at all; without it,
      gradients struggle to flow back through many layers.
    - Layer normalization — stabilizes activations, applied BEFORE each
      sub-layer here (this is the "pre-norm" convention used by GPT-2
      and most modern transformers, and trains more stably than post-norm).
"""

import torch.nn as nn

from model.attention import MultiHeadAttention


class FeedForward(nn.Module):
    """Simple two-layer MLP applied independently to every token position."""

    def __init__(self, d_model, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),  # expand
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),  # project back down
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffwd = FeedForward(d_model, dropout)

    def forward(self, x):
        # Pre-norm + residual: x = x + sublayer(norm(x))
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x