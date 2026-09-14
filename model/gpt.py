"""
model/gpt.py

The full model. Wires together:
    1. Token embeddings — turn integer token IDs into learned vectors
    2. Positional embeddings — since attention has no built-in sense of
       order, we explicitly add a learned vector per position so the
       model knows "this token is 1st, this one is 2nd," etc.
    3. A stack of N_LAYERS TransformerBlocks
    4. A final LayerNorm
    5. A linear "head" that projects back up to vocab_size — producing,
       for every position, a score (logit) for every possible next token

forward() also computes the loss when targets are given (training mode),
using cross-entropy between predicted next-token distributions and the
actual next tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.transformer_block import TransformerBlock


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, block_size, dropout):
        super().__init__()
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)

        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, block_size, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)          # final layer norm
        self.lm_head = nn.Linear(d_model, vocab_size)  # projects to vocab logits

        # Weight init (GPT-2 style) — helps training stability
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx: (B, T) integer token ids
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)                              # (B, T, d_model)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))  # (T, d_model)
        x = tok_emb + pos_emb                                             # broadcast over batch

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            return logits, None

        # Cross-entropy expects (N, C) predictions vs (N,) targets, so flatten
        B, T, C = logits.shape
        logits_flat = logits.view(B * T, C)
        targets_flat = targets.view(B * T)
        loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Autoregressive generation: repeatedly predict the next token and append it."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]  # crop to last block_size tokens
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]              # take only the last position's logits
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # sample, not argmax
            idx = torch.cat([idx, next_token], dim=1)
        return idx