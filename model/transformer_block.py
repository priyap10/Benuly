"""
One transformer block = the repeating unit that gets stacked N_LAYERS
times to form the full model. 
"""

import torch.nn as nn

from model.attention import MultiHeadAttention


class FeedForward(nn.Module):


    def __init__(self, d_model, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),  
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),  
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
        
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x