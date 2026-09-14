#the core mechanism of the transformer

import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
  
    def __init__(self, d_model, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size

    def forward(self, x):
      
        B, T, C = x.shape

        k = self.key(x)    
        q = self.query(x)  
        v = self.value(x)  
        scores = q @ k.transpose(-2, -1) * (self.head_size ** -0.5) 

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
        out = torch.cat([h(x) for h in self.heads], dim=-1) 
        out = self.proj(out)
        out = self.dropout(out)
        return out